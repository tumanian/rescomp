"""Hugging Face causal LMs as a ModelAdapter (optional dependency: transformers).

The site is "resid_post": the output of decoder block l (`model.layers[l]` / `transformer.h[l]`), which is
TransformerLens `blocks.l.hook_resid_post` when that library loads the model without weight processing.

    model, tok = load_model("gpt2")                       # -> (HFAdapter, tokenizer)
    logits, acts, _ = model.forward(tokens, capture=[8, 9])
"""
from contextlib import contextmanager

import torch

from .model import apply_patch

BLOCK_ATTRS = ["model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers", "layers"]
CONFIG_KEYS = ["hidden_size", "n_embd", "num_hidden_layers", "n_layer", "num_attention_heads", "n_head",
               "num_key_value_heads", "head_dim", "rope_theta", "intermediate_size", "vocab_size",
               "max_position_embeddings", "n_positions", "tie_word_embeddings", "hidden_act", "rms_norm_eps",
               "layer_norm_epsilon", "model_type", "num_experts", "num_local_experts", "num_experts_per_tok",
               "moe_intermediate_size", "norm_topk_prob", "decoder_sparse_step", "mlp_only_layers"]


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


def _first(o):
    return o[0] if isinstance(o, tuple) else o


class HFAdapter:
    def __init__(self, model, tokenizer=None, model_id=None, revision=None):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.blocks, self.block_path = decoder_blocks(model)
        c = model.config
        self.model_id = model_id or getattr(c, "_name_or_path", None)
        self.revision = revision or getattr(c, "_commit_hash", None)
        self.n_layers = len(self.blocks)
        self.d_model = getattr(c, "hidden_size", None) or getattr(c, "n_embd", None)

    def info(self):
        c = self.model.config
        cfg = {k: getattr(c, k) for k in CONFIG_KEYS if getattr(c, k, None) is not None}
        if self.moe_layers():
            cfg["moe_layers"] = self.moe_layers()
        return {"id": self.model_id, "revision": self.revision, "loader": "transformers",
                "architecture": (getattr(c, "architectures", None) or [type(self.model).__name__])[0],
                "n_layers": self.n_layers, "d_model": self.d_model,
                "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
                "tokenizer": getattr(self.tokenizer, "name_or_path", None),
                "bos_token_id": getattr(self.tokenizer, "bos_token_id", None), "config": cfg}

    def hook_names(self, layer):
        return {"hf": f"{self.block_path}.{layer}", "transformer_lens": f"blocks.{layer}.hook_resid_post"}

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def moe_layers(self):
        """Layers whose MLP is a routed mixture of experts (HF layout: block.mlp.gate + block.mlp.experts)."""
        return [l for l, b in enumerate(self.blocks)
                if hasattr(getattr(b, "mlp", None), "gate") and hasattr(b.mlp, "experts")]

    @contextmanager
    def _hooks(self, capture, patch, routers, acts, routing):
        handles = []
        for l in capture:
            handles.append(self.blocks[l].register_forward_hook(
                lambda mod, inp, out, l=l: acts.__setitem__(l, _first(out).detach())))
        for l in (patch or {}):
            def repl(mod, inp, out, l=l):
                new = apply_patch(l, _first(out), patch)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            handles.append(self.blocks[l].register_forward_hook(repl))
        for l in routers:
            def route(mod, inp, out, l=l):
                _, weights, experts = out               # HF TopKRouter: (logits, top-k weights, top-k expert ids)
                routing[l] = (weights.detach(), experts.detach())
            handles.append(self.blocks[l].mlp.gate.register_forward_hook(route))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def forward(self, tokens, capture=(), patch=None, routers=()):
        moe = self.moe_layers()
        if any(l not in moe for l in routers):
            raise ValueError(f"routers {list(routers)} must be MoE layers {moe}")
        acts, routing = {}, {}
        B, T = tokens.shape
        with self._hooks(capture, patch, routers, acts, routing):
            logits = self.model(tokens.to(self.device), use_cache=False).logits
        routing = {l: {"experts": e.reshape(B, T, -1).to(torch.int16).cpu(),
                       "weights": w.reshape(B, T, -1).to(torch.float16).cpu()} for l, (w, e) in routing.items()}
        return logits, acts, routing


def load_model(model_id, dtype="bfloat16", device=None, revision=None, attn_implementation=None, device_map=None, **kw):
    """-> (HFAdapter, tokenizer). device_map="auto" spreads a model that does not fit one GPU over all of them
    (hooks and patches follow each block's own device); otherwise the whole model goes to `device`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if device_map:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype), **kw)
    if not device_map:
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return HFAdapter(model, tok, model_id=model_id, revision=revision), tok
