"""Capture residual-stream activations from a Hugging Face causal LM into an RSC1 file.

The site is "resid_post": the output of decoder block l (HF `model.layers[l]` / `transformer.h[l]`, which is
TransformerLens `blocks.l.hook_resid_post` when the model is loaded without weight processing).

    model, tok = load_model("gpt2")
    tokens = wikitext_sequences(tok, n_seq=200, seq_len=256)
    capture(model, tok, tokens, layers=[8, 9, 10, 11], spec="int4_g384+layerK4+out12", out="gpt2.rsc")
"""
import time
from contextlib import contextmanager

import numpy as np
import torch

from .codec import calibrate
from .container import Writer
from .spec import Spec

BLOCK_ATTRS = ["model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers", "layers"]


def load_model(model_id, dtype="bfloat16", device=None, revision=None, attn_implementation=None, device_map=None, **kw):
    """device_map="auto" spreads a model that does not fit one GPU over all of them (hooks and patches follow
    each block's own device); otherwise the whole model goes to `device`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if device_map:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype), **kw)
    if not device_map:
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.eval(), tok


def decoder_blocks(model):
    for path in BLOCK_ATTRS:
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
        except AttributeError:
            continue
        if isinstance(m, torch.nn.ModuleList):
            return m, path
    raise ValueError(f"cannot find the decoder block list on {type(model).__name__}; tried {BLOCK_ATTRS}")


def hook_names(model, layer):
    _, path = decoder_blocks(model)
    return {"hf": f"{path}.{layer}", "transformer_lens": f"blocks.{layer}.hook_resid_post"}


def moe_layers(model):
    """Layers whose MLP is a routed mixture of experts (HF layout: block.mlp.gate + block.mlp.experts)."""
    blocks, _ = decoder_blocks(model)
    return [l for l, b in enumerate(blocks) if hasattr(getattr(b, "mlp", None), "gate") and hasattr(b.mlp, "experts")]


def model_info(model, tokenizer=None, model_id=None, revision=None):
    c = model.config
    blocks, _ = decoder_blocks(model)
    keys = ["hidden_size", "n_embd", "num_hidden_layers", "n_layer", "num_attention_heads", "n_head",
            "num_key_value_heads", "head_dim", "rope_theta", "intermediate_size", "vocab_size", "max_position_embeddings",
            "n_positions", "tie_word_embeddings", "hidden_act", "rms_norm_eps", "layer_norm_epsilon", "model_type",
            "num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size", "norm_topk_prob",
            "decoder_sparse_step", "mlp_only_layers"]
    cfg = {k: getattr(c, k) for k in keys if getattr(c, k, None) is not None}
    moe = moe_layers(model)
    if moe:
        cfg["moe_layers"] = moe
    return {"id": model_id or getattr(c, "_name_or_path", None), "revision": revision or getattr(c, "_commit_hash", None),
            "architecture": (getattr(c, "architectures", None) or [type(model).__name__])[0],
            "n_layers": len(blocks), "d_model": getattr(c, "hidden_size", None) or getattr(c, "n_embd", None),
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "tokenizer": getattr(tokenizer, "name_or_path", None), "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "config": cfg}


def _first(o):
    return o[0] if isinstance(o, tuple) else o


@contextmanager
def hooks(model, layers, store=None, patch=None, routers=(), routing=None):
    """store: dict to fill with layer -> output tensor. patch: layer -> tensor or fn(output) -> replacement.
    routers: MoE layers whose router output (top-k expert ids and weights, [B*T, k]) goes into `routing`."""
    blocks, _ = decoder_blocks(model)
    handles = []
    for l in layers:
        if store is not None:
            handles.append(blocks[l].register_forward_hook(
                lambda mod, inp, out, l=l: store.__setitem__(l, _first(out).detach())))
    for l, v in (patch or {}).items():
        def repl(mod, inp, out, v=v):
            o = _first(out)
            new = v(o) if callable(v) else v.to(o.device, o.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        handles.append(blocks[l].register_forward_hook(repl))
    for l in routers:
        def route(mod, inp, out, l=l):
            _, weights, experts = out                   # HF TopKRouter: (logits, top-k weights, top-k expert ids)
            routing[l] = (weights.detach(), experts.detach())
        handles.append(blocks[l].mlp.gate.register_forward_hook(route))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def forward(model, tokens, layers=(), patch=None, routers=()):
    """-> (logits, {layer: resid_post [B, T, d]}, {layer: {"experts": [B, T, k] int16, "weights": [B, T, k] fp16}})
    with optional replacement of some layers' outputs and router capture for MoE layers."""
    acts, routing = {}, {}
    dev = model.get_input_embeddings().weight.device
    B, T = tokens.shape
    with hooks(model, layers, store=acts, patch=patch, routers=routers, routing=routing):
        logits = model(tokens.to(dev), use_cache=False).logits
    routing = {l: {"experts": e.reshape(B, T, -1).to(torch.int16).cpu(), "weights": w.reshape(B, T, -1).to(torch.float16).cpu()}
               for l, (w, e) in routing.items()}
    return (logits, acts, routing) if routers else (logits, acts)


# ---------------------------------------------------------------- sequences

def pack_sequences(tokenizer, texts, n_seq, seq_len, bos=False):
    """Concatenate texts (one newline between them) and cut n_seq windows of seq_len tokens. Texts are encoded
    without special tokens, so no BOS lands mid-window; bos=True starts every window with one BOS token (what
    Gemma/Llama-style models and their SAEs expect)."""
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
    """wikitext-103 paragraphs in a seeded random order, packed into windows (headings dropped)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    lines = [t.strip() for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    order = np.random.default_rng(seed).permutation(len(lines))
    return pack_sequences(tokenizer, (lines[i] for i in order), n_seq, seq_len, bos=bos)


# ---------------------------------------------------------------- capture

@torch.no_grad()
def capture(model, tokenizer, tokens, layers, spec, out, *, batch=8, calib_seqs=32, model_id=None, revision=None,
            level=3, annotations=None, extra=None, site="resid_post", routers=None, log=print):
    """Run the model over tokens [N, T] and write the stored layers' resid_post to `out`.
    Outlier channels (if the spec has any) are calibrated on the first calib_seqs sequences.
    annotations: name -> array with first dim N, split per chunk.
    routers: MoE layers whose routing to store as annotations "experts.<l>" [N, T, k] (int16 expert ids) and
    "expert_weights.<l>" [N, T, k] (fp16); "stored" = the stored layers that are MoE, "all" = every MoE layer.
    Returns the written header."""
    spec = Spec.parse(spec)
    layers = sorted(int(l) for l in layers)
    N = tokens.shape[0]
    moe = moe_layers(model)
    if routers == "all":
        routers = moe
    elif routers == "stored":
        routers = [l for l in layers if l in moe]
    routers = sorted(int(l) for l in (routers or []))
    if any(l not in moe for l in routers):
        raise ValueError(f"routers {routers} must be MoE layers {moe}")
    t0 = time.time()
    calib = None
    if spec.outliers and calib_seqs > 0:                 # calib_seqs=0: the Writer calibrates on its first chunk
        n = min(calib_seqs, N)
        xs = []
        for i in range(0, n, batch):
            _, acts = forward(model, tokens[i: min(i + batch, n)], layers)
            xs.append(torch.stack([acts[l].to(torch.bfloat16).cpu() for l in layers]))
        calib = calibrate(torch.cat(xs, 1), spec, source=f"first {n} sequences")
    info = model_info(model, tokenizer, model_id, revision)
    ex = {"corpus_tokens": int(tokens.numel()), "batch": batch, "calib_seqs": calib_seqs if spec.outliers else 0}
    if routers:
        ex["routers"] = {"layers": routers, "top_k": int(model.config.num_experts_per_tok),
                         "num_experts": int(getattr(model.config, "num_experts", None) or model.config.num_local_experts)}
    ex.update(extra or {})
    with Writer(out, model=info, layers=layers, spec=spec, calib=calib, site=site, level=level,
                hook_names={l: hook_names(model, l) for l in layers}, extra=ex) as w:
        for i in range(0, N, batch):
            tb = tokens[i: i + batch]
            ann = {k: v[i: i + batch] for k, v in (annotations or {}).items()}
            if routers:
                _, acts, routing = forward(model, tb, layers, routers=routers)
                for l, r in routing.items():
                    ann[f"experts.{l}"], ann[f"expert_weights.{l}"] = r["experts"], r["weights"]
            else:
                _, acts = forward(model, tb, layers)
            w.write([acts[l] for l in layers], tb, annotations=ann or None)
            if log and ((i // batch) % 10 == 0):
                log(f"  {i + len(tb)}/{N} seqs, {time.time() - t0:.0f}s")
    h = w.written
    if log:
        b = h["bytes"]
        log(f"wrote {out}: {b['ratio_vs_bf16']:.2f}x vs bf16, rel L2 {h['fidelity']['rel_l2']:.2e}, "
            f"{time.time() - t0:.0f}s")
    return h
