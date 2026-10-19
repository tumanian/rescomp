"""How a model is hooked: one ModelBinding per framework, chosen by `bind(model, layers)`.

    layers = bind(model, [8, 9, 10, 11])          # Hugging Face or TransformerLens, detected by type
    layers = bind({8: blk8, 9: blk9}, root=model) # any model: the modules whose outputs are the residual stream
    sink = Sink("runs/x", layers, "balanced")

A binding answers four questions about a model: which modules carry the residual stream after each layer, where
the token pre-hook goes, what the file header should record, and how to read tokens off a forward call. It
produces the `Layers` record the Sink and the Patcher hook.
"""
from abc import ABC, abstractmethod

import torch

from .hooks import Layers, extract_tokens_generic


class ModelBinding(ABC):
    loader = "unknown"

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def blocks(self):
        """{layer: module} for every layer: the module whose output is the residual stream after that layer."""

    @property
    def n_layers(self):
        return len(self.blocks())

    def root(self):
        """The module whose forward receives the tokens (None: no token pre-hook)."""
        return self.model

    @abstractmethod
    def info(self, tokenizer=None, model_id=None, revision=None):
        """What goes into the file header's "model" field (JSON-serializable)."""

    def hook_name(self, layer):
        return None

    def extract_tokens(self, args, kwargs):
        return extract_tokens_generic(args, kwargs)

    def moe_layers(self):
        return []

    def layers(self, layers, tokenizer=None, model_id=None, revision=None):
        """The Layers record for a subset of layers."""
        blocks = self.blocks()
        layers = sorted(int(l) for l in layers)
        bad = [l for l in layers if l not in blocks]
        if bad:
            raise ValueError(f"layers {bad} outside [0, {len(blocks)})")
        return Layers(modules={l: blocks[l] for l in layers}, root=self.root(),
                      hook_names={l: self.hook_name(l) for l in layers},
                      model_info=self.info(tokenizer, model_id, revision), extract=self.extract_tokens,
                      n_layers=len(blocks))


class ModuleBinding(ModelBinding):
    """Any model: you name the modules."""
    loader = "modules"

    def __init__(self, modules, root=None, info=None, hook_names=None):
        super().__init__(root)
        self._modules = {int(l): m for l, m in modules.items()}
        self._info = dict(info or {})
        self._hook_names = dict(hook_names or {})

    def blocks(self):
        return self._modules

    def root(self):
        return self.model

    def info(self, tokenizer=None, model_id=None, revision=None):
        out = {"loader": self.loader, "n_layers": max(self._modules) + 1 if self._modules else 0}
        out.update(self._info)
        if model_id:
            out["id"] = model_id
        return out

    def hook_name(self, layer):
        return self._hook_names.get(layer)

    def layers(self, layers=None, tokenizer=None, model_id=None, revision=None):
        return super().layers(sorted(self._modules) if layers is None else layers, tokenizer, model_id, revision)


def binding_for(model):
    """The binding class for a model instance, by type, importing a framework only when it is present."""
    name = type(model).__mro__
    if any(c.__module__.startswith("transformers.") for c in name):
        from .hf import HuggingFaceBinding
        return HuggingFaceBinding(model)
    if any(c.__module__.startswith("transformer_lens") for c in name):
        from .tl import TransformerLensBinding
        return TransformerLensBinding(model)
    raise ValueError(f"no binding for {type(model).__name__}: pass the modules to hook, bind({{layer: module}}, root=model)")


def bind(model, layers=None, *, root=None, tokenizer=None, model_id=None, revision=None, info=None, hook_names=None):
    """-> Layers. `model` is a Hugging Face or TransformerLens model (framework detected) or a {layer: module}
    dict (then `root` is the module that receives the tokens, for the pre-hook)."""
    if isinstance(model, dict):
        return ModuleBinding(model, root=root, info=info, hook_names=hook_names).layers(layers, tokenizer, model_id, revision)
    if layers is None:
        raise ValueError("bind(model, layers): say which layers to hook")
    return binding_for(model).layers(layers, tokenizer, model_id, revision)
