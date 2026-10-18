"""The model interface the library needs: run tokens to logits while capturing or replacing layer outputs.

Anything that implements `ModelAdapter` works with capture and evaluation; nothing else in the library knows what
a model is. Adapters shipped here:

    rescomp.hf.HFAdapter              Hugging Face causal LMs (optional dependency: transformers)
    rescomp.model.TLAdapter           TransformerLens HookedTransformer (optional dependency)
    rescomp.model.FunctionAdapter     any model: you supply forward(tokens, capture, patch) -> (logits, acts)

Conventions: tokens are int64 [B, T]; a layer's activation is the residual stream after that layer, [B, T, d];
`patch` maps layer -> tensor or fn(actual) -> replacement, applied in place of that layer's output so later layers
see it. Logits are [B, T, vocab]. `routing` (MoE only) maps layer -> {"experts": [B, T, k] int16,
"weights": [B, T, k] fp16}; adapters without routers return {}.
"""
from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch


@runtime_checkable
class ModelAdapter(Protocol):
    n_layers: int
    d_model: int

    def info(self) -> dict:
        """What goes into the file header's "model" field (id, revision, config...). Must be JSON-serializable."""
        ...

    def hook_names(self, layer: int) -> Optional[dict]:
        """How this layer's site is named in the model's own framework(s), or None."""
        ...

    @property
    def device(self) -> torch.device:
        """Where input tokens should go."""
        ...

    def forward(self, tokens: torch.Tensor, capture: Sequence[int] = (), patch: Optional[dict] = None,
                routers: Sequence[int] = ()) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, dict]]:
        """-> (logits, {layer: activation for layer in capture}, routing)."""
        ...

    def moe_layers(self) -> list:
        """Layers whose MLP is a routed mixture of experts ([] for dense models)."""
        ...


def apply_patch(layer, actual, patch):
    """Helper for adapters: the replacement for `actual` if `patch` has one for `layer`, on actual's device/dtype."""
    if not patch or layer not in patch:
        return actual
    v = patch[layer]
    new = v(actual) if callable(v) else v
    return new.to(actual.device, actual.dtype)


class FunctionAdapter:
    """Wrap any model with one function.

        def run(tokens, capture, patch):   # -> (logits [B, T, V], {layer: [B, T, d] for layer in capture})
            ...                            # apply `patch` with rescomp.model.apply_patch(layer, x, patch)
        m = FunctionAdapter(run, n_layers=12, d_model=768, info={"id": "my-model"}, device="cpu")
    """

    def __init__(self, run: Callable, n_layers: int, d_model: int, info: Optional[dict] = None, device="cpu",
                 hook_names: Optional[Callable[[int], dict]] = None, moe_layers: Sequence[int] = ()):
        self._run, self.n_layers, self.d_model = run, int(n_layers), int(d_model)
        self._info = dict(info or {})
        self._device = torch.device(device)
        self._hook_names = hook_names
        self._moe = list(moe_layers)

    def info(self):
        return {"n_layers": self.n_layers, "d_model": self.d_model, **self._info}

    def hook_names(self, layer):
        return self._hook_names(layer) if self._hook_names else None

    @property
    def device(self):
        return self._device

    @torch.no_grad()
    def forward(self, tokens, capture=(), patch=None, routers=()):
        if routers:
            raise ValueError("FunctionAdapter has no router capture")
        logits, acts = self._run(tokens.to(self._device), list(capture), patch or {})
        return logits, {l: a.detach() for l, a in acts.items()}, {}

    def moe_layers(self):
        return self._moe


class TLAdapter:
    """TransformerLens HookedTransformer: the site is blocks.<l>.hook_resid_post."""

    def __init__(self, model, model_id=None):
        self.model = model
        self.model_id = model_id or getattr(getattr(model, "cfg", None), "model_name", None)
        self.n_layers, self.d_model = model.cfg.n_layers, model.cfg.d_model

    def info(self):
        cfg = self.model.cfg
        return {"id": self.model_id, "loader": "transformer_lens", "n_layers": self.n_layers, "d_model": self.d_model,
                "dtype": str(cfg.dtype).removeprefix("torch."), "config": {"n_heads": cfg.n_heads, "d_head": cfg.d_head,
                "d_mlp": cfg.d_mlp, "d_vocab": cfg.d_vocab, "n_ctx": cfg.n_ctx, "act_fn": cfg.act_fn}}

    def hook_names(self, layer):
        return {"transformer_lens": f"blocks.{layer}.hook_resid_post"}

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, tokens, capture=(), patch=None, routers=()):
        if routers:
            raise ValueError("TLAdapter has no router capture")
        acts, hooks = {}, []
        for l in capture:
            hooks.append((f"blocks.{l}.hook_resid_post", lambda t, hook, l=l: acts.__setitem__(l, t.detach())))
        for l in (patch or {}):
            hooks.append((f"blocks.{l}.hook_resid_post", lambda t, hook, l=l: apply_patch(l, t, patch)))
        logits = self.model.run_with_hooks(tokens.to(self.device), fwd_hooks=hooks)
        return logits, acts, {}

    def moe_layers(self):
        return []


def tl_layers(model, layers, model_id=None):
    """Modules to hook for a Sink on a TransformerLens HookedTransformer: blocks.<l>.hook_resid_post (HookPoints
    are modules). TransformerLens passes tokens positionally, so the generic extractor works."""
    from .sink import Layers, extract_tokens_generic
    a = TLAdapter(model, model_id=model_id)
    layers = sorted(int(l) for l in layers)
    if any(not 0 <= l < a.n_layers for l in layers):
        raise ValueError(f"layers {layers} outside [0, {a.n_layers})")
    return Layers(modules={l: model.blocks[l].hook_resid_post for l in layers}, root=model,
                  hook_names={l: a.hook_names(l) for l in layers}, model_info=a.info(),
                  extract=extract_tokens_generic, n_layers=a.n_layers)
