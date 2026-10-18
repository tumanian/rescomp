"""How much does the compression change the model? `model` is any rescomp.model.ModelAdapter.

Three measurements, all against the uncompressed model on the same tokens:

  kl_ppl          teacher-forced: mean next-token KL(clean || patched), top-1 agreement, and the perplexity ratio
                  exp(nll_patched - nll_clean).
  evaluate_file   patches the LAST stored layer (by default) with its reconstruction READ FROM THE FILE: what a
                  consumer of the file sees. Patching earlier stored layers as well would change nothing, since
                  every stored reconstruction was made from the clean activations.
  evaluate_file_per_layer
                  patches one stored layer at a time: how much replacing each layer alone changes the model.
  evaluate_live   patches with a live encoder in layer order, so each delta layer is coded against the patched
                  previous layer (errors compound as they would in a running pipeline).
  coherence       generates continuations with the live codec on, then scores each continuation with the clean
                  model: paired log-perplexity difference vs the clean model's own continuation for the same prompt
                  and seed, plus repetition statistics. (A blind LLM judge was tried before and only caught gross
                  damage; this paired perplexity test was the useful signal.)
"""
import math

import torch
import torch.nn.functional as F

from .codec import Encoder
from .container import Reader
from .spec import Spec


class Chain:
    """Live hook: called per stored layer in order; encodes against its own reconstruction of the previous layer."""

    def __init__(self, spec, calib, layers):
        self.layers = list(layers)
        self.enc = Encoder(spec, calib, len(self.layers))

    def hook(self, layer):
        i = self.layers.index(layer)

        def fn(t):
            if i == 0:
                self.enc.reset()
            _, xhat = self.enc.encode_layer(i, t)
            return xhat.to(t.dtype) if t.dtype != torch.float32 else xhat.clone()   # never hand out the encoder's state
        return fn

    def patch(self):
        return {l: self.hook(l) for l in self.layers}


@torch.no_grad()
def kl_ppl(model, tokens, layers, patch_for_batch, batch=8):
    """patch_for_batch(start, tokens_batch) -> patch dict for capture.forward. Returns summary dict."""
    kls, agree, nll_c, nll_p = [], [], [], []
    for a in range(0, tokens.shape[0], batch):
        tb = tokens[a: a + batch]
        clean, _, _ = model.forward(tb)
        patched, _, _ = model.forward(tb, patch=patch_for_batch(a, tb))
        tgt = tb[:, 1:].to(clean.device)
        for j in range(len(tb)):
            lp = F.log_softmax(clean[j].float(), -1)
            lq = F.log_softmax(patched[j].float(), -1)
            kls.append((lp.exp() * (lp - lq)).sum(-1).cpu())
            agree.append((clean[j].argmax(-1) == patched[j].argmax(-1)).float().cpu())
            nll_c.append(-lp[:-1].gather(-1, tgt[j][:, None]).squeeze(-1).cpu())
            nll_p.append(-lq[:-1].gather(-1, tgt[j][:, None]).squeeze(-1).cpu())
        del clean, patched
    k = torch.cat(kls)
    nc, npat = torch.cat(nll_c), torch.cat(nll_p)
    d_seq = torch.stack([(p - c).mean() for p, c in zip(nll_p, nll_c)])     # one log-ppl difference per sequence
    return {"kl_mean": float(k.mean()), "kl_median": float(k.median()), "kl_p95": float(k.quantile(0.95)),
            "top1_agree": float(torch.cat(agree).mean()), "n_positions": int(k.numel()), "n_seq": len(nll_c),
            "nll_clean": float(nc.mean()), "nll_patched": float(npat.mean()),
            "ppl_clean": math.exp(float(nc.mean())), "ppl_patched": math.exp(float(npat.mean())),
            "ppl_ratio": math.exp(float(npat.mean() - nc.mean())),
            # standard error of the mean per-sequence log-perplexity difference (sequences treated as independent)
            "log_ppl_diff_se": float(d_seq.std() / math.sqrt(len(d_seq))) if len(d_seq) > 1 else None}


@torch.no_grad()
def evaluate_file(path, model, batch=8, n_seq=None, layers=None):
    """KL / perplexity with the last stored layer (or `layers`) replaced by its reconstruction from the file."""
    with Reader(path) as r:
        layers = list(layers or r.layers[-1:])
        n = min(n_seq or r.n_seq, r.n_seq)
        tokens = r.tokens(slice(0, n))

        def patch(a, tb):                                   # the adapter moves each tensor to its layer's device
            stack = r.read_stack(layers, slice(a, a + len(tb)))
            return {l: stack[j] for j, l in enumerate(layers)}
        out = kl_ppl(model, tokens, layers, patch, batch)
        out.update(mode="file", layers=layers, spec=r.spec.name, ratio_vs_bf16=r.header["bytes"]["ratio_vs_bf16"])
    return out


@torch.no_grad()
def evaluate_file_per_layer(path, model, batch=8, n_seq=None, layers=None):
    """{layer: kl_ppl summary} with that layer alone replaced by its reconstruction from the file."""
    out = {}
    with Reader(path) as r:
        n = min(n_seq or r.n_seq, r.n_seq)
        tokens = r.tokens(slice(0, n))
        for l in layers or r.layers:
            out[l] = kl_ppl(model, tokens, [l], lambda a, tb, l=l: {l: r.read(l, slice(a, a + len(tb)))}, batch)
    return out


@torch.no_grad()
def evaluate_live(model, tokens, layers, spec, calib=None, batch=8):
    spec = Spec.parse(spec)
    layers = sorted(layers)
    chain = Chain(spec, calib, layers)
    out = kl_ppl(model, tokens, layers, lambda a, tb: chain.patch(), batch)
    out.update(mode="live", layers=layers, spec=spec.name, n_seq=int(tokens.shape[0]))
    return out


@torch.no_grad()
def identity_check(model, tokens, layers, batch=8):
    """Patching with the unmodified output must give KL exactly 0 (a hook-plumbing check)."""
    return kl_ppl(model, tokens, layers, lambda a, tb: {l: (lambda t: t) for l in layers}, batch)


# ---------------------------------------------------------------- generation and coherence

PROMPTS = [
    "The history of the Roman Empire", "In a small village at the edge of the forest, an old woman",
    "Photosynthesis is the process by which", "The best way to learn a new programming language is",
    "When the spacecraft finally reached the orbit of Jupiter,", "The city council met on Tuesday to discuss",
    "My grandfather always told me that", "The main causes of the French Revolution were",
    "To make a simple tomato sauce, first", "Scientists have recently discovered that",
    "The museum's newest exhibit focuses on", "Before the invention of the printing press,",
    "The detective looked at the muddy footprints and", "Climate change affects coastal cities because",
    "The first step in training a neural network is", "She opened the letter and read the first line:",
    "The economy of ancient Egypt depended on", "A healthy breakfast should include",
    "The rules of chess are simple:", "After years of drought, the farmers decided to",
]


@torch.no_grad()
def generate(model, ids, n_new, seed, patch=None, temperature=0.8, top_k=50):
    """Sampled continuation (no KV cache, so the hooks see the full prefix every step). ids [B, P] -> [B, P + n_new]."""
    dev = model.device
    ids = ids.to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(n_new):
        logits, _, _ = model.forward(ids, patch=patch)
        p = F.softmax(logits[:, -1].float() / temperature, -1)
        v, i = p.topk(top_k, -1)
        nxt = i.gather(-1, torch.multinomial(v / v.sum(-1, keepdim=True), 1, generator=gen))
        ids = torch.cat([ids, nxt], 1)
    return ids


def ngram_stats(toks):
    bi = list(zip(toks, toks[1:]))
    four = [tuple(toks[i: i + 4]) for i in range(len(toks) - 3)]
    seen, rep = set(), 0
    for g in four:
        rep += g in seen
        seen.add(g)
    return {"distinct2": len(set(bi)) / max(len(bi), 1), "rep4": rep / max(len(four), 1)}


@torch.no_grad()
def continuation_nll(model, ids, n_prompt):
    """Mean NLL of the continuation tokens under the clean model, given the prompt. ids [B, P + n]."""
    logits, _, _ = model.forward(ids)
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    nll = -lp.gather(-1, ids[:, 1:, None].to(lp.device)).squeeze(-1)
    return nll[:, n_prompt - 1:].mean(-1).cpu()


@torch.no_grad()
def coherence(model, prompts, layers, spec, calib=None, tokenizer=None, n_new=80, seeds=3, temperature=0.8, top_k=50):
    """prompts: strings (needs tokenizer) or lists of token ids. For each prompt and seed, generate once clean and
    once with the live codec, then score both continuations with the clean model.
    Returns {"records": [...], "summary": {...}}: log_ppl_diff is mean(nll_codec - nll_clean) over all pairs, so
    ppl_ratio = exp(log_ppl_diff). The seeds of one prompt share that prompt, so the standard error and t use one
    averaged difference per prompt (n_prompts observations), not every pair."""
    spec = Spec.parse(spec)
    layers = sorted(layers)
    chain = Chain(spec, calib, layers)
    recs = []
    for pi, prompt in enumerate(prompts):
        p_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        ids = torch.tensor([p_ids] * seeds)
        clean = generate(model, ids, n_new, 1000 + pi, None, temperature, top_k)
        coded = generate(model, ids, n_new, 1000 + pi, chain.patch(), temperature, top_k)
        nc, nk = continuation_nll(model, clean, len(p_ids)), continuation_nll(model, coded, len(p_ids))
        for s in range(seeds):
            tc, tk = clean[s, len(p_ids):].tolist(), coded[s, len(p_ids):].tolist()
            recs.append({"prompt_id": pi, "seed": s, "prompt": prompt if isinstance(prompt, str) else None,
                         "clean": {"tokens": tc, "nll": float(nc[s]), **ngram_stats(tc),
                                   "text": tokenizer.decode(tc) if tokenizer else None},
                         "codec": {"tokens": tk, "nll": float(nk[s]), **ngram_stats(tk),
                                   "text": tokenizer.decode(tk) if tokenizer else None}})
    d = torch.tensor([r["codec"]["nll"] - r["clean"]["nll"] for r in recs]).reshape(len(prompts), seeds)
    d_prompt = d.mean(1)
    se = float(d_prompt.std() / math.sqrt(len(d_prompt))) if len(d_prompt) > 1 else None
    summary = {"spec": spec.name, "layers": layers, "n_pairs": len(recs), "n_prompts": len(prompts),
               "log_ppl_diff": float(d.mean()), "log_ppl_diff_se": se, "ppl_ratio": math.exp(float(d.mean())),
               "t": float(d.mean()) / se if se else None,
               "rep4_clean": sum(r["clean"]["rep4"] for r in recs) / len(recs),
               "rep4_codec": sum(r["codec"]["rep4"] for r in recs) / len(recs),
               "distinct2_clean": sum(r["clean"]["distinct2"] for r in recs) / len(recs),
               "distinct2_codec": sum(r["codec"]["distinct2"] for r in recs) / len(recs)}
    return {"records": recs, "summary": summary}
