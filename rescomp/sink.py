"""The Sink: hooks on a running model, activations flow into a store. The model is not wrapped.

    sink = Sink("runs/run1", layers=bind(model, [8, 9, 10, 11]), codec="balanced")  # a regime, a spec string or a Codec
    with sink:
        model(**batch)              # one forward call = one chunk
        model.generate(**prompt)    # each step = one chunk; the reader returns the sequence whole
    sink.stats()

`layers` is a `Layers` (from `bind`) or a plain {layer: nn.Module} dict; the hook on each module
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
import warnings

import torch

from .bf16_wrapper import OutlierChannels, check_channels, choose_layer_channels
from .codec import as_codec
from .hooks import CallHooks, Layers, extract_tokens_generic  # noqa: F401  (re-exported for callers)
from .quant import fp16_inexact
from .store import Store, check_annotation_names


class Sink:
    def __init__(self, stem, layers, codec="balanced", *, tokens=None, buffer_bytes=1 << 30, rotate_bytes=None,
                 site="resid_post", extra=None, model_info=None, fidelity=False, check_exact=None, overwrite=False,
                 bucket=None, prefix=None, s3=None):
        """stem: a local path stem, an s3://bucket/prefix/name URI, or a name with bucket= and prefix= (see
        rescomp.store.location); an existing capture there is an error unless overwrite=True. codec: a Codec, or a
        regime name / spec string for Codec() (regimes resolve on the first call, when the width is known).
        fidelity=True also measures each layer's reconstruction error in the hook (an extra reconstruction and three
        reductions per layer, in fp32): useful once, costly on every forward. check_exact (default: on for the `exact`
        regime) checks every call that fp16 holds this model's values (bf16 / fp16 models: exactly; wider: within
        range) and raises otherwise; see rescomp.quant.fp16_inexact."""
        if tokens not in (None, "off"):
            raise ValueError("tokens must be None (pre-hook) or 'off'")
        self.codec = as_codec(codec)
        self.check_exact = (self.codec.regime == "exact") if check_exact is None else bool(check_exact)
        self.hooks = CallHooks(layers, on_layer=self._on_layer, on_end=self._complete_call, tokens=tokens is None)
        L, self.order = self.hooks.layers, self.hooks.order
        self.layers = L
        self.fidelity = fidelity
        self.buffer_bytes, self.site = buffer_bytes, site
        info = dict(L.model_info)
        info.update(model_info or {})
        info.setdefault("n_layers", L.n_layers)
        self.store = Store(stem, model=info, layers=self.order, codec=self.codec, site=site, hook_names=L.hook_names,
                           extra=dict(extra or {}), rotate_bytes=rotate_bytes, overwrite=overwrite, bucket=bucket, prefix=prefix, s3=s3)
        self.enc = None
        self._pending_ann = {}
        self._open, self._len = [], {}    # open sequence ids and their current lengths
        self._buffer, self._buffered = [], 0     # encoded chunks waiting for zstd + write
        self._choosing = False
        self.t0 = time.time()

    @property
    def spec(self):
        return self.codec.spec

    @property
    def channels(self):
        return self.codec.channels

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
        check_annotation_names(cols)
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
        """Take the hooks off, write what is buffered and finish the capture."""
        self.hooks.detach()
        self._write_buffer()
        self.store.close()

    def discard(self):
        """Take the hooks off and stop without writing: the buffered calls are dropped and the stem reads as it did
        after the last flush (or holds no capture at all). What `with` does when its body raises, so a run that
        fails (the exact check, an OOM, an interrupt) never leaves a capture that looks complete."""
        self.hooks.detach()
        self._buffer, self._buffered = [], 0
        self.store.discard()

    def __enter__(self):
        self.hooks.attach()
        return self

    def __exit__(self, exc_type, *_):
        (self.close if exc_type is None else self.discard)()

    # -- the hook side. Two extension points: `_layer_step` (one layer's activation -> what the call keeps) and
    # `_call_step` (a complete call -> what goes into the buffer). `_on_layer` / `_complete_call` are the fixed
    # entry points the hooks call; they keep the call record and the sequence bookkeeping.
    def _on_layer(self, l, x, call):
        """Hook entry, inside the model's forward: the first stored layer opens the call record; a layer that
        fires without the first one, or with a different shape, is dropped (the call ends up incomplete)."""
        i = self.order.index(l)
        if i == 0:
            call["shape"], call["n_done"] = tuple(x.shape), 0
        if "shape" not in call or tuple(x.shape) != call["shape"]:
            warnings.warn(f"rescomp: layer {l} fired with shape {tuple(x.shape)} "
                          + ("before the first hooked layer" if "shape" not in call else f"after shape {call['shape']}")
                          + "; this forward call is not recorded", stacklevel=2)
            return
        self._layer_step(i, l, x, call)
        call["n_done"] += 1

    def _layer_step(self, i, l, x, call):
        """Extension point 1, per stored layer (i: index in storage order, l: layer number, x: the activation
        [B, T, d] on its device). Turns the activation into what the call record keeps for `_call_step`. Runs
        inside the forward, so it must be cheap and must not synchronize. The Sink encodes the layer now, on its
        device, so the model can free the activation as usual."""
        if i == 0:
            call["streams"], call["err"], call["inexact"] = {}, {}, {}
            self._encoder_for(x)
        if self._choosing:
            self._choose_channels(i, x)
        if self.check_exact:                           # a 0-d flag on the device; read once per call in _complete_call
            call["inexact"][l] = fp16_inexact(x)
        st, xhat = self.enc.encode_layer(i, x, reconstruct=self.fidelity)
        if self.fidelity:
            xf = x.float()
            e = xf - xhat
            # fp32 and kept on the device as a 0-d tensor: reducing to Python here would synchronize
            call["err"][l] = torch.stack([e.pow(2).sum(), xf.pow(2).sum(), e.abs().max()])
        call["streams"][l] = st                        # stays on the device until flush
        if l == self.order[-1] and self._choosing:
            self._choosing = False

    def _complete_call(self, call):
        """The last stored layer fired: a call every layer went through becomes a chunk; one with a dropped layer
        counts as incomplete. Decides whether the call continues the B open sequences (a generation step: T == 1
        with the model's cache signal, or lacking that, after a call of the same batch size) or starts new ones."""
        if call.get("n_done") != len(self.order):
            self.hooks.n_incomplete += 1
            self.hooks.n_calls -= 1
            return
        if self.check_exact:
            bad = [l for l in self.order if bool(call.get("inexact", {}).get(l, False))]
            if bad:
                raise ValueError(f"the exact regime (fp16) cannot hold layer(s) {bad} of this call: a value is above 65504 "
                                 "in magnitude or too small for fp16; use codec='bf16' (exact for any bf16 model, at a "
                                 "lower ratio) or check_exact=False to accept fp16's clamping")
        B, T, d = call["shape"]
        cont = call["cont"]
        if cont is None:
            cont = T == 1 and len(self._open) == B
        self._call_step(call, cont and len(self._open) == B)

    def _call_step(self, call, cont):
        """Extension point 2, per complete call: register its sequences (`_register`) and put what `_layer_step`
        kept into the buffer; `cont` says the call continues the open sequences rather than starting B new ones.
        The Sink buffers one encoded chunk per call and flushes when the on-device buffer is full."""
        seq_ids, starts = self._register(call, cont)
        B, T, d = call["shape"]
        self.store.note_raw(B * T * d * 2 * len(self.order), d)
        streams = [call["streams"][l] for l in self.order]
        err = torch.stack([call["err"][l] for l in self.order]) if self.fidelity else None
        self._buffer.append((seq_ids, starts, streams, err))
        self._buffered += sum(t.numel() * t.element_size() for st in streams for t in st.values())
        if self._buffered >= self.buffer_bytes:
            self.flush()

    def _register(self, call, cont):
        """The call's sequences in the store: the B open ones extended by its T tokens, or B new ones with the
        pending annotations. Returns (seq_ids, start positions)."""
        B, T, d = call["shape"]
        if cont:
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
        return seq_ids, starts

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
            c = self.codec
            spec = c.resolve(x0.shape[-1])
            if c.channels is None and spec.outliers:
                c.channels = OutlierChannels.placeholder(len(self.order), spec.outliers)
                self._choosing = True
            elif spec.outliers:
                check_channels(c.channels, spec, len(self.order))
            self.enc = c.encoder(len(self.order))
        self.enc.reset()

    def _choose_channels(self, i, x):
        """First call only: this layer's outlier channels from what is about to be quantized (one sync per layer)."""
        v = x.float() if self.spec.is_key(i) else x.float() - self.enc.prev.to(x.device)
        self.enc.set_channel_ids(i, choose_layer_channels(v, self.spec.outliers))

    def _write_buffer(self):
        """The slow part, off the forward path: move the encoded streams to the host, zstd + write each chunk."""
        for seq_ids, starts, streams, err in self._buffer:
            host = [{k: t.cpu() for k, t in st.items()} for st in streams]
            self.store.add_chunk(seq_ids, starts, host, None if err is None else err.cpu().tolist())
        self._buffer, self._buffered = [], 0
