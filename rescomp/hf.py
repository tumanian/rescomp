"""Hugging Face causal LMs: which modules a Sink hooks, what the header should say, and a loader.

    model, tok = load_model("gpt2")
    sink = Sink("runs/x", layers=hf_layers(model, [8, 9, 10, 11], tok), spec="balanced")

The site is "resid_post": the output of decoder block l (`model.layers[l]` / `transformer.h[l]`), which is
TransformerLens `blocks.l.hook_resid_post` when that library loads the model without weight processing.
Only this module imports transformers.
"""
import torch

from .sink import Layers, extract_tokens_generic

BLOCK_ATTRS = ["model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers", "layers"]
CONFIG_KEYS = ["hidden_size", "n_embd", "num_hidden_layers", "n_layer", "num_attention_heads", "n_head",
               "num_key_value_heads", "head_dim", "rope_theta", "intermediate_size", "vocab_size",
               "max_position_embeddings", "n_positions", "tie_word_embeddings", "hidden_act", "rms_norm_eps",
               "layer_norm_epsilon", "model_type", "num_experts", "num_local_experts", "num_experts_per_tok",
               "moe_intermediate_size", "norm_topk_prob", "decoder_sparse_step", "mlp_only_layers"]


def decoder_blocks(model):
    """(the ModuleList of decoder blocks, its attribute path)."""
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


def moe_layers(model):
    """Layers whose MLP is a routed mixture of experts (HF layout: block.mlp.gate + block.mlp.experts)."""
    blocks, _ = decoder_blocks(model)
    return [l for l, b in enumerate(blocks) if hasattr(getattr(b, "mlp", None), "gate") and hasattr(b.mlp, "experts")]


def model_info(model, tokenizer=None, model_id=None, revision=None):
    """What goes into the file header: id, revision, architecture, sizes, dtype, a config summary."""
    c = model.config
    blocks, _ = decoder_blocks(model)
    cfg = {k: getattr(c, k) for k in CONFIG_KEYS if getattr(c, k, None) is not None}
    moe = moe_layers(model)
    if moe:
        cfg["moe_layers"] = moe
    return {"id": model_id or getattr(c, "_name_or_path", None), "revision": revision or getattr(c, "_commit_hash", None),
            "loader": "transformers", "architecture": (getattr(c, "architectures", None) or [type(model).__name__])[0],
            "n_layers": len(blocks), "d_model": getattr(c, "hidden_size", None) or getattr(c, "n_embd", None),
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "tokenizer": getattr(tokenizer, "name_or_path", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None), "config": cfg}


def hf_layers(model, layers, tokenizer=None, model_id=None, revision=None):
    """Modules to hook for a Sink: the decoder blocks of `layers`, with header info and the HF token extractor."""
    blocks, path = decoder_blocks(model)
    layers = sorted(int(l) for l in layers)
    if any(not 0 <= l < len(blocks) for l in layers):
        raise ValueError(f"layers {layers} outside [0, {len(blocks)})")
    return Layers(modules={l: blocks[l] for l in layers}, root=model,
                  hook_names={l: {"hf": f"{path}.{l}", "transformer_lens": f"blocks.{l}.hook_resid_post"} for l in layers},
                  model_info=model_info(model, tokenizer, model_id, revision), extract=extract_tokens_generic,
                  n_layers=len(blocks))


def load_model(model_id, dtype="bfloat16", device=None, revision=None, attn_implementation=None, device_map=None, **kw):
    """-> (model in eval mode, tokenizer). device_map="auto" spreads a model that does not fit one GPU over all of
    them (the Sink follows each block's own device); otherwise the whole model goes to `device`."""
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
