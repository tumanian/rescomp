"""The codec: one object for how activations are stored, lossy or lossless.

    codec = Codec("balanced")                                  # a regime (resolved when the width is known) ...
    codec = Codec("int4_g96+layerK4+out8", level=5)            # ... or any spec string, with the zstd level
    codec = Codec("bf16")                                      # lossless for a bf16 model
    enc = codec.encoder(n_layers); enc.encode_layer(i, x)      # on the model's device, one layer at a time
    blob, raw = codec.pack(stream)                             # on the host, at flush: byte-shuffle + zstd
    codec.to_dict()                                            # what the file header records

Inside: a quantizer (rescomp.quant), the closed-loop layer deltas and keyframes (below), the outlier channels and
sink rows kept exactly (rescomp.bf16_wrapper) and the lossless stage (rescomp.lossless). Which of them run on the
device and which on the host is this module's business: `encode_layer` is the device half, `pack` the host half.

Order per layer (fp32 throughout):
    v = x                       at keyframes
    v = x - xhat_prev           otherwise (closed loop: the delta is against the DECODED previous stored layer)
    sink rows out (bf16) -> outlier channels out (bf16) -> quantize the rest -> reconstruct
    xhat = vhat (+ xhat_prev)

Encoder and Decoder run the same reconstruction ops in the same order, so the decoder returns exactly the
encoder's xhat. `layer_values` holds the tensor work of one layer with no Python state, so torch.compile can fuse
it; `_CodecState` holds what carries over between layers of one chunk.
"""
import json

import torch

from .bf16_wrapper import OutlierChannels, check_channels, restore, split
from .lossless import lossless
from .quant import CODEC_VERSION, Cast, quantizer
from .regimes import is_regime, needs_width, resolve
from .spec import Spec


# ---------------------------------------------------------------- validators

def check_layer_index(i, n_layers):
    if not 0 <= i < n_layers:
        raise IndexError(f"stored layer index {i} out of range [0, {n_layers})")


def check_order(spec, i, expected):
    """A delta layer must follow the layer it is coded against; a keyframe may start a new run."""
    if not spec.is_key(i) and i != expected:
        raise ValueError(f"layer {i} is a delta layer: code layers in order (expected {expected})")


def check_width(spec, i, x, seen):
    """x is [B, T, d] with the width this codec has seen so far and a d the quantizer can block."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, T, d], got {tuple(x.shape)}")
    d = x.shape[-1]
    if seen is not None and d != seen:
        raise ValueError(f"d changed from {seen} to {d}")
    quantizer(spec.quant_for(i)).check_dim(d)
    if spec.sink > x.shape[1]:
        raise ValueError(f"sink{spec.sink} > T={x.shape[1]}")
    return d


# ---------------------------------------------------------------- the per-layer tensor work

def _layer_values(x, prev, O, sink, q, reconstruct):
    """fp32 cast, closed-loop delta, sink rows and outlier channels out, scales, codes and (if asked) the
    reconstruction with rows and channels restored. q is a Quantizer. Returns (scales, codes, sink_rows,
    outliers, xhat)."""
    v = x.to(torch.float32, copy=True) if prev is None else x.float() - prev
    sink_rows, outl = split(v, O, sink)
    S = q.scales(v)
    codes = q.encode(v, S)
    if not reconstruct:
        return S, codes, sink_rows, outl, None
    vhat = restore(q.decode(codes, S), O, sink, sink_rows, outl)
    return S, codes, sink_rows, outl, (vhat if prev is None else prev + vhat)


_COMPILED = {}


def layer_values(x, prev, O, sink, q, reconstruct, compile=False):
    if not compile:
        return _layer_values(x, prev, O, sink, q, reconstruct)
    fn = _COMPILED.get("fn")
    if fn is None:
        # Inductor keeps the intermediates of a fused kernel in fp32 by default, which would skip the fp16 rounding
        # of the scales and codes on the way to xhat; the emulation keeps the compiled bytes equal to the eager ones.
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True
        fn = _COMPILED["fn"] = torch.compile(_layer_values, dynamic=True)
    return fn(x, prev, O, sink, q, reconstruct)


def reconstruct(streams, q, spec, O):
    """Decoder side: streams -> vhat fp32 [B, T, d], the same ops as the encoder's reconstruction."""
    S = [streams[k] for k in sorted(streams) if k.startswith("scale")]
    return restore(q.decode(streams["codes"], S), O, spec.sink, streams.get("sink"), streams.get("outliers"))


# ---------------------------------------------------------------- state across the layers of one chunk

class _CodecState:
    """What carries over from one stored layer to the next within a chunk: the spec, the outlier channels (with
    a per-device copy of their ids), the previous layer's reconstruction and the next expected layer index.
    Built by Codec.encoder() / Codec.decoder(); the channel ids it chooses go into the codec's record."""

    def __init__(self, codec, n_layers):
        if codec.spec is None:
            raise ValueError(f"codec {codec.given!r} needs the model width first: codec.resolve(d)")
        self.spec, self.n_layers, self.compile = codec.spec, n_layers, codec.compile
        self.channels = codec.channels if codec.channels is not None else OutlierChannels(None, "none")
        check_channels(self.channels, self.spec, n_layers)
        self.d = None
        self._ids_on = {}
        self.reset()

    def reset(self):
        self.prev, self.expect = None, 0

    def channel_ids(self, i, dev):
        """Outlier channel ids of layer i on `dev`, copied once: a per-call host-to-device copy of a pageable
        tensor is synchronous and would stall the CPU behind the GPU at every hooked layer."""
        if self.spec.outliers == 0:
            return None
        key = (i, str(dev))
        o = self._ids_on.get(key)
        if o is None:
            o = self._ids_on[key] = self.channels.ids[i].to(dev)
        return o

    def set_channel_ids(self, i, ids):
        """First call: the ids for layer i, already on the device they were measured on."""
        self._ids_on[(i, str(ids.device))] = ids
        self.channels.ids[i] = ids.cpu()


class Encoder(_CodecState):
    """Call encode_layer(i, x) for i = 0, 1, ..., in order within one chunk (keyframes may be coded alone)."""

    @torch.no_grad()
    def encode_layer(self, i, x, reconstruct=True):
        """x [B, T, d] -> (streams: name -> tensor on x's device, xhat fp32 [B, T, d]).
        xhat is also the reference for the next delta layer: do not modify it in place.
        reconstruct=False skips the dequantization when no delta layer follows and the caller does not need
        xhat (returns None); a delta layer always reconstructs."""
        check_layer_index(i, self.n_layers)
        check_order(self.spec, i, self.expect)
        d = check_width(self.spec, i, x, self.d)
        if self.d is None:
            check_channels(self.channels, self.spec, self.n_layers, d)
            self.d = d
        self.expect = i + 1
        spec = self.spec
        prev = None if spec.is_key(i) else self.prev.to(x.device)        # layers may live on different GPUs
        needed = reconstruct or (spec.K > 1 and not spec.is_key(i + 1) and i + 1 < self.n_layers)
        S, codes, sink_rows, outl, xhat = layer_values(x, prev, self.channel_ids(i, x.device), spec.sink,
                                                       quantizer(spec.quant_for(i)), needed, self.compile)
        streams = {}
        if sink_rows is not None:
            streams["sink"] = sink_rows
        if outl is not None:
            streams["outliers"] = outl
        for j, s in enumerate(S):
            streams[f"scale{j}"] = s
        streams["codes"] = codes
        self.prev = xhat
        return streams, xhat


class Decoder(_CodecState):
    @torch.no_grad()
    def decode_layer(self, i, streams):
        check_layer_index(i, self.n_layers)
        check_order(self.spec, i, self.expect)
        self.expect = i + 1
        spec = self.spec
        vhat = reconstruct(streams, quantizer(spec.quant_for(i)), spec, self.channel_ids(i, streams["codes"].device))
        xhat = vhat if spec.is_key(i) else self.prev + vhat
        self.prev = xhat
        return xhat


class Codec:
    """How activations are stored. `spec`: a regime name, a spec template or a spec string (or a Spec);
    `channels`: the outlier channels when the spec keeps some (else chosen on the first call by the Sink, or from
    a sample by transcode); `lossless`/`level`/`shuffle`: the byte stage; `compile`: torch.compile the per-layer
    function (fused kernels, same bytes)."""

    def __init__(self, spec="balanced", *, channels=None, lossless="zstd", level=3, shuffle=True, compile=False):
        self.given = spec.name if isinstance(spec, Spec) else spec
        self.regime = spec if is_regime(spec) else None
        self.spec = None if needs_width(spec) else resolve(spec)       # a regime resolves once d is known
        self.channels = channels
        self.lossless = _lossless(lossless, level, shuffle)
        self.compile = compile

    @classmethod
    def from_header(cls, header):
        """The codec a file was written with (header: the meta row with the JSON fields decoded or not)."""
        def field(k):
            v = header[k]
            return json.loads(v) if isinstance(v, str) else v
        c = cls(header["spec"], channels=OutlierChannels.from_dict(field("outlier_channels")), lossless=field("lossless"))
        c.regime = header.get("regime")
        return c

    # -- the spec
    def resolve(self, d):
        """Fix the width-dependent sizes of a regime or template (idempotent) -> Spec."""
        if self.spec is None:
            self.spec = resolve(self.given, d)
        return self.spec

    @property
    def lossy(self):
        """False when every layer is a plain cast (fp32 / bf16 / fp16: exact for bf16 activations in range)."""
        spec = self.spec if self.spec is not None else resolve(self.given, 1 << 20)
        return not all(isinstance(quantizer(spec.quant_for(i)), Cast) for i in range(max(2, spec.K or 2)))

    # -- the device half: tensors in, streams of codes and scales out
    def encoder(self, n_layers):
        """A fresh per-chunk encoder: encode_layer(i, x) for i = 0, 1, ... in storage order."""
        return Encoder(self, n_layers)

    def decoder(self, n_layers):
        return Decoder(self, n_layers)

    @torch.no_grad()
    def encode(self, X):
        """X [L, B, T, d] -> (list of per-layer stream dicts, xhat [L, B, T, d] fp32)."""
        enc = self.encoder(X.shape[0])
        out, xh = [], []
        for i in range(X.shape[0]):
            s, x = enc.encode_layer(i, X[i])
            out.append(s)
            xh.append(x)
        return out, torch.stack(xh)

    @torch.no_grad()
    def decode(self, streams):
        dec = self.decoder(len(streams))
        return torch.stack([dec.decode_layer(i, s) for i, s in enumerate(streams)])

    # -- the host half: a stream tensor <-> compressed bytes
    def pack(self, t):
        """tensor -> (compressed bytes, raw byte count)."""
        return self.lossless.compress(t)

    def unpack(self, blob, dtype, shape, raw_nbytes):
        return self.lossless.decompress(blob, dtype, shape, raw_nbytes)

    # -- the header
    def to_dict(self):
        return {"version": CODEC_VERSION, "spec": self.spec.name if self.spec is not None else self.given,
                "regime": self.regime, "outlier_channels": (self.channels or OutlierChannels(None, "none")).to_dict(),
                "lossless": self.lossless.to_dict()}

    def __repr__(self):
        return f"Codec({self.given!r}{', lossy' if self.lossy else ', lossless'}, {self.lossless!r})"


def as_codec(x):
    """A Codec, a spec string / Spec / regime name -> Codec."""
    return x if isinstance(x, Codec) else Codec(x)


def _lossless(backend, level, shuffle):
    if isinstance(backend, (str, dict)) or backend is None:
        rec = {"codec": backend or "zstd"} if not isinstance(backend, dict) else dict(backend)
        if isinstance(backend, dict):
            return lossless(rec)
        return lossless(rec, level=level, shuffle=shuffle)
    return lossless(backend)
