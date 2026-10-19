"""Instrumentation: feed reconstructions back into a running model and measure what changes.

Separate from the production flow (`rescomp.sink`). Same module-object convention: a `Patcher` hooks the modules
of a `Layers` and replaces their outputs; the model is otherwise run by the caller.

    patcher = Patcher(layers, source=rc.open("runs/run1"))              # reconstructions read from a store
    patcher = Patcher(layers, source=LiveCodec("int4_g384+layerK4+out12", channels))   # encode-decode on the fly
    with patcher.sequences([0, 1]):                                     # which stored sequences this batch is
        logits = model(**batch)

Measurements take a plain callable `run(tokens) -> logits [B, T, V]` (any framework) and a Patcher:
    kl_ppl(run, tokens, patcher, seq_ids)      KL(clean || patched), top-1 agreement, perplexity ratio
    per_layer(run, tokens, layers, source)     one layer at a time
    coherence(run, prompts, patcher_live, ...) generation with the codec on, paired perplexity per prompt
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .bf16_wrapper import OutlierChannels
from .codec import Encoder
from .hooks import CallHooks, Layers
from .spec import Spec


class LiveCodec:
    """Source for a Patcher: encode each hooked layer against the previous patched one, return its reconstruction."""

    def __init__(self, spec, channels=None):
        self.spec, self.channels = Spec.parse(spec), channels
        self.enc = None

    def bind(self, order, all_layers=None):
        """order: the layers actually patched (a subset re-indexes keyframes: each patched run starts with one).
        Outlier channels chosen for `all_layers` are cut down to the rows of `order`."""
        self.order = list(order)
        ch = self.channels
        o = None if ch is None else ch.ids
        if o is not None and all_layers and len(o) == len(all_layers) and len(self.order) != len(all_layers):
            ch = OutlierChannels(o[[list(all_layers).index(l) for l in self.order]], ch.source)
        self.enc = Encoder(self.spec, ch, len(self.order))

    def __call__(self, layer, x, seq_ids, starts):
        i = self.order.index(layer)
        if i == 0:
            self.enc.reset()
        _, xhat = self.enc.encode_layer(i, x)
        return xhat


class StoreSource:
    """Source for a Patcher: the stored reconstruction of the given sequences (whole, or a token window)."""

    def __init__(self, reader):
        self.reader = reader

    def bind(self, order, all_layers=None):
        missing = [l for l in order if l not in self.reader.layers]
        if missing:
            raise ValueError(f"layers {missing} are not stored; file has {self.reader.layers}")

    def __call__(self, layer, x, seq_ids, starts):
        B, T, d = x.shape
        if seq_ids is None:
            raise ValueError("a store-backed Patcher needs patcher.sequences([...]) around the forward call")
        if len(seq_ids) != B:
            raise ValueError(f"{len(seq_ids)} sequence ids for a batch of {B}")
        rows = []
        for sid, st in zip(seq_ids, starts):
            full = self.reader.read(layer, int(sid))
            if full.shape[0] < st + T:
                raise ValueError(f"sequence {sid} has {full.shape[0]} stored tokens, need {st + T}")
            rows.append(full[st: st + T])
        return torch.stack(rows)


class Patcher:
    """Replace the hooked layers' outputs with reconstructions from `source` (a Reader, LiveCodec, or any
    callable (layer, x, seq_ids, starts) -> replacement). Active inside `with patcher:`."""

    def __init__(self, layers, source, only: Optional[list] = None):
        L = layers if isinstance(layers, Layers) else Layers(modules=dict(layers))
        if only is not None:
            missing = [l for l in only if l not in L.modules]
            if missing:
                raise ValueError(f"layers {missing} are not hooked; have {sorted(L.modules)}")
            L = L.subset([l for l in sorted(L.modules) if l in only])
        self.hooks = CallHooks(L, on_layer=self._on_layer, tokens=False)
        self.order = self.hooks.order
        if hasattr(source, "read") and hasattr(source, "layers"):
            source = StoreSource(source)
        self.source = source
        if hasattr(source, "bind"):
            source.bind(self.order, sorted(layers.modules) if isinstance(layers, Layers) else sorted(layers))
        self._seq_ids, self._starts = None, None

    def sequences(self, seq_ids, starts=None):
        """Declare which stored sequences the next forward call's batch rows are (store-backed source)."""
        self._seq_ids = list(seq_ids)
        self._starts = list(starts) if starts is not None else [0] * len(self._seq_ids)
        return self

    def _on_layer(self, l, x, call):
        return self.source(l, x, self._seq_ids, self._starts)

    def __enter__(self):
        self.hooks.attach()
        return self

    def __exit__(self, *exc):
        self.hooks.detach()


# ---------------------------------------------------------------- measurements

@torch.no_grad()
def kl_ppl(run, tokens, patcher, seq_ids=None, batch=8):
    """KL(clean || patched) per position, top-1 agreement, teacher-forced perplexity ratio. `run(tokens)` gives
    logits; `seq_ids` (one per token row) are declared to a store-backed patcher per batch."""
    kls, agree, nll_c, nll_p = [], [], [], []
    for a in range(0, tokens.shape[0], batch):
        tb = tokens[a: a + batch]
        clean = run(tb)
        if seq_ids is not None:
            patcher.sequences(seq_ids[a: a + len(tb)])
        with patcher:
            patched = run(tb)
        tgt = tb[:, 1:].to(clean.device)
        for j in range(len(tb)):
            lp = F.log_softmax(clean[j].float(), -1)
            lq = F.log_softmax(patched[j].float(), -1)
            kls.append((lp.exp() * (lp - lq)).sum(-1).cpu())
            agree.append((clean[j].argmax(-1) == patched[j].argmax(-1)).float().cpu())
            nll_c.append(-lp[:-1].gather(-1, tgt[j][:, None]).squeeze(-1).cpu())
            nll_p.append(-lq[:-1].gather(-1, tgt[j][:, None]).squeeze(-1).cpu())
    k = torch.cat(kls)
    nc, npat = torch.cat(nll_c), torch.cat(nll_p)
    d_seq = torch.stack([(p - c).mean() for p, c in zip(nll_p, nll_c)])
    return {"kl_mean": float(k.mean()), "kl_median": float(k.median()), "kl_p95": float(k.quantile(0.95)),
            "top1_agree": float(torch.cat(agree).mean()), "n_positions": int(k.numel()), "n_seq": len(nll_c),
            "nll_clean": float(nc.mean()), "nll_patched": float(npat.mean()),
            "ppl_ratio": math.exp(float(npat.mean() - nc.mean())),
            "log_ppl_diff_se": float(d_seq.std() / math.sqrt(len(d_seq))) if len(d_seq) > 1 else None}


@torch.no_grad()
def per_layer(run, tokens, layers, source, seq_ids=None, batch=8):
    """kl_ppl with one layer patched at a time: {layer: summary}."""
    L = layers if isinstance(layers, Layers) else Layers(modules=dict(layers))
    return {l: kl_ppl(run, tokens, Patcher(L, source, only=[l]), seq_ids, batch) for l in sorted(L.modules)}


def ngram_stats(toks):
    bi = list(zip(toks, toks[1:]))
    four = [tuple(toks[i: i + 4]) for i in range(len(toks) - 3)]
    seen, rep = set(), 0
    for g in four:
        rep += g in seen
        seen.add(g)
    return {"distinct2": len(set(bi)) / max(len(bi), 1), "rep4": rep / max(len(four), 1)}


@torch.no_grad()
def generate(run, ids, n_new, seed, temperature=0.8, top_k=50):
    """Sampled continuation through `run` (full prefix every step, so patch hooks see whole sequences)."""
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    for _ in range(n_new):
        logits = run(ids)
        p = F.softmax(logits[:, -1].float() / temperature, -1)
        v, i = p.topk(min(top_k, p.shape[-1]), -1)
        nxt = i.gather(-1, torch.multinomial(v / v.sum(-1, keepdim=True), 1, generator=gen)).to(ids.device)
        ids = torch.cat([ids, nxt], 1)
    return ids


@torch.no_grad()
def continuation_nll(run, ids, n_prompt):
    logits = run(ids)
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    nll = -lp.gather(-1, ids[:, 1:, None].to(lp.device)).squeeze(-1)
    return nll[:, n_prompt - 1:].mean(-1).cpu()


@torch.no_grad()
def coherence(run, prompts, patcher, tokenizer=None, n_new=80, seeds=3, temperature=0.8, top_k=50, device="cpu"):
    """Generate once clean and once with the patcher on (a LiveCodec source), score both with the clean model;
    the standard error and t use one averaged log-perplexity difference per prompt."""
    recs = []
    for pi, prompt in enumerate(prompts):
        p_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        ids = torch.tensor([p_ids] * seeds, device=device)
        clean = generate(run, ids, n_new, 1000 + pi, temperature, top_k)
        with patcher:
            coded = generate(run, ids, n_new, 1000 + pi, temperature, top_k)
        nc, nk = continuation_nll(run, clean, len(p_ids)), continuation_nll(run, coded, len(p_ids))
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
    summary = {"n_pairs": len(recs), "n_prompts": len(prompts), "log_ppl_diff": float(d.mean()), "log_ppl_diff_se": se,
               "ppl_ratio": math.exp(float(d.mean())), "t": float(d.mean()) / se if se else None,
               "rep4_clean": sum(r["clean"]["rep4"] for r in recs) / len(recs),
               "rep4_codec": sum(r["codec"]["rep4"] for r in recs) / len(recs),
               "distinct2_clean": sum(r["clean"]["distinct2"] for r in recs) / len(recs),
               "distinct2_codec": sum(r["codec"]["distinct2"] for r in recs) / len(recs)}
    return {"records": recs, "summary": summary}
