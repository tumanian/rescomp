"""Hook mechanics shared by the Sink and the Patcher: attach to a model's layer modules, observe each layer's
output as the forward pass produces it, and track the lifecycle of one forward call.

    hooks = CallHooks(layers, on_layer=fn, on_end=done)
    with hooks:
        model(**batch)

Per forward call: the pre-hook on the root module (if there is one) opens a `call` record and takes the tokens
from the call's arguments (`input_ids` keyword or first positional tensor, `attention_mask`, and the KV-cache
signal that tells a generation step from a new prompt; with tokens=False only the ids are left out); each hooked layer's forward hook passes (layer, output,
call) to `on_layer`, whose return value, if not None, replaces the layer's output in the model; when the last
hooked layer fires, `on_end(call)` runs. A call whose last hooked layer never fires counts as incomplete.
`give_tokens(ids, mask)` supplies the next call's tokens explicitly instead of the pre-hook.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch


@dataclass
class Layers:
    """Which modules to hook, plus what the file header should know about them."""
    modules: Dict[int, torch.nn.Module]
    root: Optional[torch.nn.Module] = None                      # for the token pre-hook
    hook_names: Dict[int, dict] = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    extract: Optional[Callable] = None                          # (args, kwargs) -> (ids, mask, continuation|None)
    n_layers: Optional[int] = None

    def subset(self, layers):
        missing = [l for l in layers if l not in self.modules]
        if missing:
            raise ValueError(f"layers {missing} are not hooked; have {sorted(self.modules)}")
        return Layers({l: self.modules[l] for l in layers}, self.root, {l: self.hook_names.get(l) for l in layers},
                      self.model_info, self.extract, self.n_layers)


def extract_tokens_generic(args, kwargs):
    """Default extractor: input_ids kwarg or first positional tensor; attention_mask kwarg; a cache hint from
    past_key_values / cache_position (Hugging Face) when present."""
    ids = kwargs.get("input_ids")
    if ids is None and args and isinstance(args[0], torch.Tensor):
        ids = args[0]
    mask = kwargs.get("attention_mask")
    cont = None
    cache = kwargs.get("past_key_values")
    if cache is not None and hasattr(cache, "get_seq_length"):
        try:
            cont = int(cache.get_seq_length()) > 0
        except Exception:
            cont = None
    if cont is None and kwargs.get("cache_position") is not None:
        cont = int(kwargs["cache_position"].reshape(-1)[0]) > 0
    return ids, mask, cont


def first_output(o):
    return o[0] if isinstance(o, tuple) else o


class CallHooks:
    def __init__(self, layers, on_layer, on_end=None, tokens=True):
        L = layers if isinstance(layers, Layers) else Layers(modules=dict(layers))
        self.layers = L
        self.order = sorted(int(l) for l in L.modules)
        if not self.order:
            raise ValueError("no layers to hook")
        self.on_layer, self.on_end, self.record_tokens = on_layer, on_end, tokens
        self._handles = []
        self.call = None
        self._given = None
        self.n_calls = self.n_incomplete = 0

    # -- lifecycle
    def attach(self):
        if self._handles:
            return
        if self.layers.root is not None:
            self._handles.append(self.layers.root.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for l in self.order:
            self._handles.append(self.layers.modules[l].register_forward_hook(
                lambda mod, inp, out, l=l: self._layer_hook(l, out)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *exc):
        self.detach()

    def give_tokens(self, ids, mask=None):
        self._given = (torch.as_tensor(ids).cpu(), None if mask is None else torch.as_tensor(mask).cpu())

    # -- hooks
    def _begin(self):
        if self.call is not None and self.call["seen"]:
            self.n_incomplete += 1                     # the previous call never reached the last hooked layer
        self.call = {"ids": None, "mask": None, "cont": None, "given": False, "seen": []}
        if self._given is not None:
            self.call["ids"], self.call["mask"] = self._given
            self.call["given"] = True
            self._given = None

    def _pre_hook(self, mod, args, kwargs):
        """The attention mask and the cache signal are always read (pooling needs the mask to skip padding); the
        token ids only when tokens are recorded."""
        self._begin()
        if not self.call["given"]:
            extract = self.layers.extract or extract_tokens_generic
            ids, mask, cont = extract(args, kwargs)
            self.call.update(ids=None if ids is None or not self.record_tokens else ids.detach().cpu(),
                             mask=None if mask is None else mask.detach().cpu(), cont=cont)

    def _layer_hook(self, l, out):
        if self.call is None or (l == self.order[0] and self.call["seen"]):
            self._begin()                              # no root pre-hook: the first hooked layer opens a call
        call = self.call
        call["seen"].append(l)
        x = first_output(out)
        new = self.on_layer(l, x.detach(), call)
        if l == self.order[-1]:
            self.call = None
            if len(call["seen"]) == len(self.order):
                self.n_calls += 1
                if self.on_end is not None:
                    self.on_end(call)
            else:
                self.n_incomplete += 1
        if new is None:
            return None
        new = new.to(x.device, x.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
