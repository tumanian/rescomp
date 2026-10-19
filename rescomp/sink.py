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

import torch

from .bf16_wrapper import OutlierChannels, check_channels, choose_layer_channels
from .codec import Encoder
from .hooks import CallHooks, Layers, extract_tokens_generic  # noqa: F401  (re-exported for callers)
from .regimes import is_regime, needs_width, resolve
from .store import Store


class Sink:
    def __init__(self, stem, layers, spec, *, tokens=None, channels=None, buffer_bytes=1 << 30, rotate_bytes=None,
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
        self.hooks = CallHooks(layers, on_layer=self._on_layer, on_end=self._complete_call, tokens=tokens is None)
        L, self.order = self.hooks.layers, self.hooks.order
        self.layers = L
        self.fidelity, self.compile = fidelity, compile
        self.channels = channels
        self.buffer_bytes, self.site = buffer_bytes, site
        info = dict(L.model_info)
        info.update(model_info or {})
        info.setdefault("n_layers", L.n_layers)
        self.store = Store(stem, model=info, layers=self.order, spec=self.spec, channels=channels, site=site, level=level,
                           hook_names=L.hook_names, extra=dict(extra or {}), rotate_bytes=rotate_bytes,
                           bucket=bucket, prefix=prefix, s3=s3)
        self.store.regime = spec if is_regime(spec) else None
        self.enc = None
        self._pending_ann = {}
        self._open, self._len = [], {}    # open sequence ids and their current lengths
        self._buffer, self._buffered = [], 0     # encoded chunks waiting for zstd + write
        self._calibrating = False
        self.t0 = time.time()

    @property
    def n_calls(self):
        return self.hooks.n_calls

    @property
    def n_incomplete(self):
        return self.hooks.n_incomplete

    # -- user-facing
    def tokens(self, ids, mask=None):
        """Tokens for the next forward call (overrides the pre-hook)."""
        self.hooks.give_tokens(ids, mask)

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
        self.hooks.detach()
        self._write_buffer()
        self.store.close()

    def __enter__(self):
        self.hooks.attach()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- per hooked layer, inside the model's forward
    def _on_layer(self, l, x, call):
        """Encode this layer now, on its device, so the model can free the activation as usual."""
        i = self.order.index(l)
        if i == 0:
            call["shape"], call["streams"], call["err"] = tuple(x.shape), {}, {}
            self._encoder_for(x)
        if "shape" not in call or tuple(x.shape) != call["shape"]:
            return                                     # a layer fired without the first one (or shapes changed): drop the call
        if self._calibrating:
            self._calibrate_layer(i, x)
        st, xhat = self.enc.encode_layer(i, x, reconstruct=self.fidelity)
        if self.fidelity:
            xf = x.float()
            e = xf - xhat
            # fp32 and kept on the device as a 0-d tensor: reducing to Python here would synchronize
            call["err"][l] = torch.stack([e.pow(2).sum(), xf.pow(2).sum(), e.abs().max()])
        call["streams"][l] = st                        # stays on the device until flush
        if l == self.order[-1] and self._calibrating:
            self._calibrating = False
            self.store.channels = self.channels

    # -- a complete forward call
    def _complete_call(self, call):
        if len(call.get("streams", {})) != len(self.order):
            self.hooks.n_incomplete += 1
            self.hooks.n_calls -= 1
            return
        B, T, d = call["shape"]
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
        """Resolve the spec on the first call's first layer and reset the encoder per call. Without given
        outlier channels, the first call chooses each layer's in that layer's own hook, from what the encoder is
        about to quantize (x at keyframes, x - xhat_prev at delta layers); see bf16_wrapper."""
        if self.enc is None:
            if self.spec is None:
                self.spec = resolve(self._spec_arg, x0.shape[-1])
                self.store.set_spec(self.spec)
            if self.channels is None and self.spec.outliers:
                self.channels = OutlierChannels.placeholder(len(self.order), self.spec.outliers)
                self._calibrating = True
            elif self.spec.outliers:
                check_channels(self.channels, self.spec, len(self.order))
                self.store.channels = self.channels
            self.enc = Encoder(self.spec, self.channels, len(self.order), compile=self.compile)
        self.enc.reset()

    def _calibrate_layer(self, i, x):
        v = x.float() if self.spec.is_key(i) else x.float() - self.enc.prev.to(x.device)
        self.enc.set_channel_ids(i, choose_layer_channels(v, self.spec.outliers))   # first call only: one sync per layer

    def _write_buffer(self):
        """The slow part, off the forward path: move the encoded streams to the host, zstd + write each chunk."""
        for seq_ids, starts, streams, err in self._buffer:
            host = [{k: t.cpu() for k, t in st.items()} for st in streams]
            self.store.add_chunk(seq_ids, starts, host, None if err is None else err.cpu().tolist())
        self._buffer, self._buffered = [], 0
