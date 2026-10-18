"""The Sink: hooks on a running model, activations flow into a format-2 store. The model is not wrapped.

    sink = Sink("runs/run1", layers=hf_layers(model, [8, 9, 10, 11]), spec="int4_g384+layerK4+out12")
    with sink:
        model(**batch)              # one forward call = one chunk
        model.generate(**prompt)    # each step = one chunk; the reader returns the sequence whole
    sink.stats()

`layers` is a `Layers` (from hf_layers / tl_layers) or a plain {layer: nn.Module} dict; the hook on each module
takes its output (the residual stream after that layer). Tokens: given with `sink.tokens(ids, mask)` before a
forward, or taken by a forward pre-hook on the root model (`input_ids` keyword or first positional argument);
`tokens="off"` records none.

What happens where: the hook quantizes on the model's device (cheap map-and-reduce work, closed-loop deltas
included), starts asynchronous copies of the encoded streams to pinned host memory and returns without
synchronizing; the slow parts, zstd and the file writes, run at flush() (also forced when the host buffer
exceeds `buffer_bytes`, and at close). The forward pass pays only for the quantizer kernels and the copies.

Sequences: a call with T > 1 tokens starts B new sequences; a call with T == 1 continues the B open ones when
the model says it is using a cache (HF: past_key_values / cache_position) or, lacking that signal, when the
previous call had the same batch size.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

from .codec import Encoder, calibrate
from .spec import Spec
from .store import Store


@dataclass
class Layers:
    """Which modules to hook, plus what the file header should know about them."""
    modules: Dict[int, torch.nn.Module]
    root: Optional[torch.nn.Module] = None                      # for the token pre-hook
    hook_names: Dict[int, dict] = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    extract: Optional[Callable] = None                          # (args, kwargs) -> (ids, mask, continuation|None)
    n_layers: Optional[int] = None


def extract_tokens_generic(args, kwargs):
    """Default pre-hook extractor: input_ids kwarg or first positional tensor; attention_mask kwarg; a cache
    hint from past_key_values / cache_position (Hugging Face) when present."""
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


def _first(o):
    return o[0] if isinstance(o, tuple) else o


class Sink:
    def __init__(self, stem, layers, spec, *, tokens=None, calib=None, buffer_bytes=2 << 30, rotate_bytes=4 << 30,
                 level=3, site="resid_post", extra=None, model_info=None):
        if tokens not in (None, "off"):
            raise ValueError("tokens must be None (pre-hook) or 'off'")
        self.spec = Spec.parse(spec)
        L = layers if isinstance(layers, Layers) else Layers(modules=dict(layers))
        self.layers = L
        self.order = sorted(int(l) for l in L.modules)
        if not self.order:
            raise ValueError("no layers to hook")
        self.record_tokens = tokens is None
        self.calib = calib
        self.buffer_bytes, self.site = buffer_bytes, site
        info = dict(L.model_info)
        info.update(model_info or {})
        info.setdefault("n_layers", L.n_layers)
        self.store = Store(stem, model=info, layers=self.order, spec=self.spec, calib=calib, site=site, level=level,
                           hook_names=L.hook_names, extra=dict(extra or {}), rotate_bytes=rotate_bytes)
        self.enc = None
        self._handles = []
        self._call = None                 # the forward call being observed: {"acts", "ids", "mask", "cont"}
        self._given = None                # tokens given explicitly for the next call
        self._pending_ann = {}
        self._open, self._len = [], {}    # open sequence ids and their current lengths
        self._buffer, self._buffered = [], 0     # encoded chunks waiting for zstd + write
        self._events = []                         # CUDA events marking the end of each chunk's host copies
        self.n_calls = self.n_incomplete = 0
        self.t0 = time.time()

    # -- user-facing
    def tokens(self, ids, mask=None):
        """Tokens for the next forward call (overrides the pre-hook)."""
        self._given = (torch.as_tensor(ids).cpu(), None if mask is None else torch.as_tensor(mask).cpu())

    def annotate(self, **cols):
        """Labels for the sequences the next new-sequence call starts: each value a list/tensor of length B, or a
        scalar for all of them."""
        self._pending_ann.update(cols)

    def flush(self):
        """zstd + write everything encoded so far; the files are readable afterwards."""
        self._write_buffer()
        self.store.flush()

    def stats(self):
        s = self.store.stats()
        s.update(calls=self.n_calls, incomplete_calls=self.n_incomplete, buffered_bytes=self._buffered,
                 seconds=round(time.time() - self.t0, 1))
        return s

    def close(self):
        self._remove_hooks()
        self._write_buffer()
        self.store.close()

    def __enter__(self):
        self._add_hooks()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- hooks
    def _add_hooks(self):
        if self._handles:
            return
        if self.layers.root is not None:
            self._handles.append(self.layers.root.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for l in self.order:
            self._handles.append(self.layers.modules[l].register_forward_hook(
                lambda mod, inp, out, l=l: self._layer_hook(l, _first(out))))

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre_hook(self, mod, args, kwargs):
        self._begin_call()
        if self.record_tokens and not self._call["given"]:
            extract = self.layers.extract or extract_tokens_generic
            ids, mask, cont = extract(args, kwargs)
            self._call.update(ids=None if ids is None else ids.detach().cpu(),
                              mask=None if mask is None else mask.detach().cpu(), cont=cont)

    def _begin_call(self):
        if self._call is not None and self._call["acts"]:
            self.n_incomplete += 1                     # a previous call never reached the last stored layer
        self._call = {"acts": {}, "ids": None, "mask": None, "cont": None, "given": False}
        if self._given is not None:
            self._call["ids"], self._call["mask"] = self._given
            self._call["given"] = True
            self._given = None

    def _layer_hook(self, l, x):
        if self._call is None or (l == self.order[0] and self._call["acts"]):
            self._begin_call()                         # no root pre-hook: the first stored layer opens a call
        self._call["acts"][l] = x.detach()
        if l == self.order[-1]:
            self._complete_call()

    # -- a complete forward call
    def _complete_call(self):
        call, self._call = self._call, None
        acts = [call["acts"][l] for l in self.order]
        if any(a is None for a in acts) or len(call["acts"]) != len(self.order):
            self.n_incomplete += 1
            return
        B, T, d = acts[0].shape
        self.n_calls += 1
        cont = call["cont"]
        if cont is None:
            cont = T == 1 and len(self._open) == B
        if cont and len(self._open) == B:
            seq_ids = list(self._open)
            starts = [self._len[s] for s in seq_ids]
            if call["ids"] is not None:
                for j, s in enumerate(seq_ids):
                    ids_row = call["ids"][j, -T:] if call["ids"].dim() == 2 else call["ids"][-T:]
                    mask_row = None if call["mask"] is None else call["mask"][j, -T:]
                    self.store.extend_sequence(s, ids_row, mask_row)
        else:
            ann = self._take_annotations(B)
            seq_ids = []
            for j in range(B):
                ids_row = None if call["ids"] is None else call["ids"][j]
                mask_row = None if call["mask"] is None else call["mask"][j]
                seq_ids.append(self.store.new_sequence(ids_row, mask_row, **{k: v[j] for k, v in ann.items()}))
            starts = [0] * B
            self._open = seq_ids
        for s in seq_ids:
            self._len[s] = self._len.get(s, 0) + T
        self.store.note_raw(B * T * d * 2 * len(acts), d)
        self._encode_call(seq_ids, starts, acts)
        if self._buffered >= self.buffer_bytes:
            self.flush()

    def _take_annotations(self, B):
        out = {}
        for k, v in self._pending_ann.items():
            if isinstance(v, (list, tuple)) or (isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B):
                if len(v) != B:
                    raise ValueError(f"annotation {k!r} has {len(v)} values for a batch of {B}")
                out[k] = list(v)
            else:
                out[k] = [v] * B
        self._pending_ann = {}
        return out

    # -- encoding (in the hook, on the model's device)
    def _encoder(self, acts):
        if self.enc is None:
            if self.calib is None and self.spec.outliers:
                self.calib = calibrate(torch.stack([a.float() for a in acts]), self.spec, source="first_chunk")
            if self.spec.outliers:
                self.calib.check(self.spec, len(self.order))
                self.store.calib = self.calib
            self.enc = Encoder(self.spec, self.calib, len(self.order))
        self.enc.reset()
        return self.enc

    @staticmethod
    def _to_host(t):
        """Start a copy to pinned host memory and return the destination (synchronized later at flush)."""
        if t.device.type == "cpu":
            return t
        dst = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        dst.copy_(t, non_blocking=True)
        return dst

    def _encode_call(self, seq_ids, starts, acts):
        enc = self._encoder(acts)
        streams, err = [], []
        for i, x in enumerate(acts):
            st, xhat = enc.encode_layer(i, x)
            xf = x.float()
            e = xf - xhat
            # error statistics stay on the device as 0-d tensors: reducing them here would synchronize
            err.append(torch.stack([e.double().pow(2).sum(), xf.double().pow(2).sum(), e.abs().max().double()]))
            streams.append({k: self._to_host(t) for k, t in st.items()})
        dev = acts[0].device
        event = None
        if dev.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(dev))
        self._buffer.append((list(seq_ids), list(starts), streams, torch.stack(err), event))
        self._buffered += sum(t.numel() * t.element_size() for st in streams for t in st.values())

    def _write_buffer(self):
        """The slow part, off the forward path: wait for the copies, then zstd + write each chunk."""
        for seq_ids, starts, streams, err, event in self._buffer:
            if event is not None:
                event.synchronize()
            self.store.add_chunk(seq_ids, starts, streams, err.cpu().tolist())
        self._buffer, self._buffered = [], 0
