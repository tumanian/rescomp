"""Capture residual-stream activations from any ModelAdapter into an RSC1 file, plus token packing helpers.

    from rescomp.hf import load_model            # or rescomp.model.FunctionAdapter / TLAdapter
    model, tok = load_model("gpt2")
    tokens = wikitext_sequences(tok, n_seq=200, seq_len=256)
    capture(model, tokens, layers=[8, 9, 10, 11], spec="int4_g384+layerK4+out12", out="gpt2.rsc")
"""
import time

import numpy as np
import torch

from .codec import calibrate
from .container import Writer
from .spec import Spec


# ---------------------------------------------------------------- sequences

def pack_sequences(tokenizer, texts, n_seq, seq_len, bos=False):
    """Concatenate texts (one newline between them) and cut n_seq windows of seq_len tokens. Texts are encoded
    without special tokens, so no BOS lands mid-window; bos=True starts every window with one BOS token (what
    Gemma/Llama-style models and their SAEs expect). `tokenizer` needs encode(text, add_special_tokens=) and,
    for bos=True, bos_token_id."""
    body = seq_len - 1 if bos else seq_len
    need, ids = n_seq * body, []
    for t in texts:
        ids.extend(tokenizer.encode(t + "\n", add_special_tokens=False))
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"texts give {len(ids)} tokens, need {need} for {n_seq} x {seq_len}")
    tokens = torch.tensor(ids[:need], dtype=torch.long).view(n_seq, body)
    if bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("tokenizer has no BOS token")
        tokens = torch.cat([torch.full((n_seq, 1), tokenizer.bos_token_id, dtype=torch.long), tokens], 1)
    return tokens


def wikitext_sequences(tokenizer, n_seq, seq_len, seed=0, bos=False, split="train"):
    """wikitext-103 paragraphs in a seeded random order, packed into windows (headings dropped). Needs `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    lines = [t.strip() for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    order = np.random.default_rng(seed).permutation(len(lines))
    return pack_sequences(tokenizer, (lines[i] for i in order), n_seq, seq_len, bos=bos)


# ---------------------------------------------------------------- capture

@torch.no_grad()
def capture(model, tokens, layers, spec, out, *, batch=8, calib_seqs=32, level=3, annotations=None, extra=None,
            site="resid_post", routers=None, log=print):
    """Run `model` (a ModelAdapter) over tokens [N, T] and write the stored layers' activations to `out`.
    Outlier channels (if the spec has any) are calibrated on the first calib_seqs sequences (0: on the first chunk).
    annotations: name -> array with first dim N, split per chunk.
    routers: MoE layers whose routing to store as annotations "experts.<l>" [N, T, k] (int16 expert ids) and
    "expert_weights.<l>" [N, T, k] (fp16); "stored" = the stored layers that are MoE, "all" = every MoE layer.
    Returns the written header."""
    spec = Spec.parse(spec)
    layers = sorted(int(l) for l in layers)
    if layers and not 0 <= layers[0] <= layers[-1] < model.n_layers:
        raise ValueError(f"layers {layers} outside [0, {model.n_layers})")
    N = tokens.shape[0]
    moe = model.moe_layers()
    if routers == "all":
        routers = moe
    elif routers == "stored":
        routers = [l for l in layers if l in moe]
    routers = sorted(int(l) for l in (routers or []))
    if any(l not in moe for l in routers):
        raise ValueError(f"routers {routers} must be MoE layers {moe}")
    t0 = time.time()
    calib = None
    if spec.outliers and calib_seqs > 0:
        n = min(calib_seqs, N)
        xs = []
        for i in range(0, n, batch):
            _, acts, _ = model.forward(tokens[i: min(i + batch, n)], capture=layers)
            xs.append(torch.stack([acts[l].to(torch.bfloat16).cpu() for l in layers]))
        calib = calibrate(torch.cat(xs, 1), spec, source=f"first {n} sequences")
    ex = {"corpus_tokens": int(tokens.numel()), "batch": batch, "calib_seqs": calib_seqs if spec.outliers else 0}
    if routers:
        info = model.info().get("config", {})
        ex["routers"] = {"layers": routers, "top_k": info.get("num_experts_per_tok"),
                         "num_experts": info.get("num_experts") or info.get("num_local_experts")}
    ex.update(extra or {})
    with Writer(out, model=model.info(), layers=layers, spec=spec, calib=calib, site=site, level=level,
                hook_names={l: model.hook_names(l) for l in layers}, extra=ex) as w:
        for i in range(0, N, batch):
            tb = tokens[i: i + batch]
            ann = {k: v[i: i + batch] for k, v in (annotations or {}).items()}
            _, acts, routing = model.forward(tb, capture=layers, routers=routers)
            for l, r in routing.items():
                ann[f"experts.{l}"], ann[f"expert_weights.{l}"] = r["experts"], r["weights"]
            w.write([acts[l] for l in layers], tb, annotations=ann or None)
            if log and ((i // batch) % 10 == 0):
                log(f"  {i + len(tb)}/{N} seqs, {time.time() - t0:.0f}s")
    h = w.written
    if log:
        b = h["bytes"]
        log(f"wrote {out}: {b['ratio_vs_bf16']:.2f}x vs bf16, rel L2 {h['fidelity']['rel_l2']:.2e}, "
            f"{time.time() - t0:.0f}s")
    return h
