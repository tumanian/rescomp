"""The Sink: hooks on a running model, activations flow into a format-2 store. The model is not wrapped.

    sink = Sink("runs/run1", layers=hf_layers(model, [8, 9, 10, 11]), spec="balanced")   # a regime or a spec string
    with sink:
        model(**batch)              # one forward call = one chunk
        model.generate(**prompt)    # each step = one chunk; the reader returns the sequence whole
    sink.stats()

`layers` is a `Layers` (from hf_layers / tl_layers) or a plain {layer: nn.Module} dict; the hook on each module
takes its output (the residual stream after that layer). Tokens: given with `sink.tokens(ids, mask)` before a
forward, or taken by a forward pre-hook on the root model (`input_ids` keyword or first positional argument);
`tokens="off"` records none.

What happens where: each layer's hook quantizes that layer on the model's device as soon as it is produced
(cheap map-and-reduce work, closed-loop deltas included) and returns; the encoded streams stay on the device,
so the model frees its activations exactly as it would without the sink and the forward pass pays only for the
quantizer kernels. flush() moves the encoded streams to the host, runs zstd and writes the files; it is also
forced when the on-device buffer exceeds `buffer_bytes` (default 1 GiB), and at close.

Sequences: a call with T > 1 tokens starts B new sequences; a call with T == 1 continues the B open ones when
the model says it is using a cache (HF: past_key_values / cache_position) or, lacking that signal, when the
previous call had the same batch size.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

from .codec import Calib, Encoder
from .regimes import is_regime, needs_width, resolve
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
    def __init__(self, stem, layers, spec, *, tokens=None, calib=None, buffer_bytes=1 << 30, rotate_bytes=None,
                 level=3, site="resid_post", extra=None, model_info=None, fidelity=False, compile=False,
                 bucket=None, prefix=None, s3=None):
        """stem: a local path stem, an s3://bucket/prefix/name URI, or a name with bucket= and prefix= (see
        rescomp.storage). fidelity=True also measures each layer's reconstruction error in the hook (an extra
        reconstruction and three reductions per layer, in fp32): useful once, costly on every forward.
        compile=True runs the quantizer through torch.compile (fused kernels; a one-time compile per shape class)."""
        if tokens not in (None, "off"):
            raise ValueError("tokens must be None (pre-hook) or 'off'")
        self._spec_arg = spec
        self.spec = None if needs_width(spec) else resolve(spec)     # regimes resolve on the first call (need d)
        L = layers if isinstance(layers, Layers) else Layers(modules=dict(layers))
        self.layers = L
        self.order = sorted(int(l) for l in L.modules)
        if not self.order:
            raise ValueError("no layers to hook")
        self.record_tokens = tokens is None
        self.fidelity, self.compile = fidelity, compile
        self.calib = calib
        self.buffer_bytes, self.site = buffer_bytes, site
        info = dict(L.model_info)
        info.update(model_info or {})
        info.setdefault("n_layers", L.n_layers)
        self.store = Store(stem, model=info, layers=self.order, spec=self.spec, calib=calib, site=site, level=level,
                           hook_names=L.hook_names, extra=dict(extra or {}), rotate_bytes=rotate_bytes,
                           bucket=bucket, prefix=prefix, s3=s3)
        self.store.regime = spec if is_regime(spec) else None
        self.enc = None
        self._handles = []
        self._call = None                 # the forward call being observed: {"acts", "ids", "mask", "cont"}
        self._given = None                # tokens given explicitly for the next call
        self._pending_ann = {}
        self._open, self._len = [], {}    # open sequence ids and their current lengths
        self._buffer, self._buffered = [], 0     # encoded chunks waiting for zstd + write
        self._calibrating = False
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
        s.update(calls=self.n_calls, incomplete_calls=self.n_incomplete, buffered_bytes=self._buffered,   # on the device
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
        if self._call is not None and self._call["streams"]:
            self.n_incomplete += 1                     # a previous call never reached the last stored layer
        self._call = {"streams": {}, "err": {}, "shape": None, "ids": None, "mask": None, "cont": None, "given": False}
        if self._given is not None:
            self._call["ids"], self._call["mask"] = self._given
            self._call["given"] = True
            self._given = None

    def _layer_hook(self, l, x):
        """Encode this layer now, on its device, so the model can free the activation as usual."""
        if self._call is None or (l == self.order[0] and self._call["streams"]):
            self._begin_call()                         # no root pre-hook: the first stored layer opens a call
        x = x.detach()
        i = self.order.index(l)
        if i == 0:
            self._call["shape"] = tuple(x.shape)
            self._encoder_for(x)
        if self._call["shape"] is None or tuple(x.shape) != self._call["shape"]:
            return                                     # a layer fired without the first one (or shapes changed): drop the call
        if self._calibrating:
            self._calibrate_layer(i, x)
        st, xhat = self.enc.encode_layer(i, x, reconstruct=self.fidelity)
        if self.fidelity:
            xf = x.float()
            e = xf - xhat
            # fp32 and kept on the device as a 0-d tensor: reducing to Python here would synchronize
            self._call["err"][l] = torch.stack([e.pow(2).sum(), xf.pow(2).sum(), e.abs().max()])
        self._call["streams"][l] = st                   # stays on the device until flush
        if l == self.order[-1]:
            if self._calibrating:
                self._calibrating = False
                self.store.calib = self.calib
            self._complete_call(x.device)

    # -- a complete forward call
    def _complete_call(self, dev):
        call, self._call = self._call, None
        if len(call["streams"]) != len(self.order):
            self.n_incomplete += 1
            return
        B, T, d = call["shape"]
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
        self.store.note_raw(B * T * d * 2 * len(self.order), d)
        streams = [call["streams"][l] for l in self.order]
        err = torch.stack([call["err"][l] for l in self.order]) if self.fidelity else None
        self._buffer.append((list(seq_ids), list(starts), streams, err))
        self._buffered += sum(t.numel() * t.element_size() for st in streams for t in st.values())
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
    def _encoder_for(self, x0):
        """Resolve the spec on the first call's first layer and reset the encoder per call. Without a given
        calibration, the first call calibrates each layer's outlier channels in that layer's own hook, from what
        the encoder is about to quantize (x at keyframes, x - xhat_prev at delta layers)."""
        if self.enc is None:
            if self.spec is None:
                self.spec = resolve(self._spec_arg, x0.shape[-1])
                self.store.set_spec(self.spec)
            if self.calib is None and self.spec.outliers:
                n, L = self.spec.outliers, len(self.order)
                self.calib = Calib(torch.arange(n).repeat(L, 1), source="first_call")   # placeholder rows, filled per layer
                self._calibrating = True
            elif self.spec.outliers:
                self.calib.check(self.spec, len(self.order))
                self.store.calib = self.calib
            self.enc = Encoder(self.spec, self.calib, len(self.order), compile=self.compile)
        self.enc.reset()

    def _calibrate_layer(self, i, x):
        v = x.float() if self.spec.is_key(i) else x.float() - self.enc.prev.to(x.device)
        energy = v.pow(2).sum(tuple(range(v.dim() - 1)))
        ids = energy.topk(self.spec.outliers).indices.sort().values
        self.enc._outl_cache[(i, str(x.device))] = ids          # the encoder uses this row from now on
        self.calib.outliers[i] = ids.cpu()                      # first call only: one sync per layer

    def _write_buffer(self):
        """The slow part, off the forward path: move the encoded streams to the host, zstd + write each chunk."""
        for seq_ids, starts, streams, err in self._buffer:
            host = [{k: t.cpu() for k, t in st.items()} for st in streams]
            self.store.add_chunk(seq_ids, starts, host, None if err is None else err.cpu().tolist())
        self._buffer, self._buffered = [], 0
