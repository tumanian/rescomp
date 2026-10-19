"""Layer codec: turns one stored layer [B, T, d] into named streams and back, layer after layer.

Order per layer (fp32 throughout):
    v = x                       at keyframes
    v = x - xhat_prev           otherwise (closed loop: the delta is against the DECODED previous stored layer)
    sink rows out (bf16) -> outlier channels out (bf16) -> quantize the rest -> reconstruct
    xhat = vhat (+ xhat_prev)

Encoder and Decoder run the same reconstruction ops in the same order, so the decoder returns exactly the
encoder's xhat. `layer_values` holds the tensor work of one layer with no Python state, so torch.compile can fuse
it; `_CodecState` holds what carries over between layers of one chunk.
"""
import torch

from . import quant as Q
from .bf16_wrapper import OutlierChannels, check_channels, restore, split
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
    Q.check_dim(spec.quant_for(i), d)
    if spec.sink > x.shape[1]:
        raise ValueError(f"sink{spec.sink} > T={x.shape[1]}")
    return d


# ---------------------------------------------------------------- the per-layer tensor work

def _layer_values(x, prev, O, sink, q, reconstruct):
    """fp32 cast, closed-loop delta, sink rows and outlier channels out, scales, codes and (if asked) the
    reconstruction with rows and channels restored. Returns (scales, codes, sink_rows, outliers, xhat)."""
    v = x.to(torch.float32, copy=True) if prev is None else x.float() - prev
    sink_rows, outl = split(v, O, sink)
    S = Q.make_scales(v, q)
    codes = Q.quantize(v, S, q)
    if not reconstruct:
        return S, codes, sink_rows, outl, None
    vhat = restore(Q.dequantize(codes, S, q), O, sink, sink_rows, outl)
    return S, codes, sink_rows, outl, (vhat if prev is None else prev + vhat)


_COMPILED = {}


def layer_values(x, prev, O, sink, q, reconstruct, compile=False):
    if not compile:
        return _layer_values(x, prev, O, sink, q, reconstruct)
    fn = _COMPILED.get("fn")
    if fn is None:
        fn = _COMPILED["fn"] = torch.compile(_layer_values, dynamic=True)
    return fn(x, prev, O, sink, q, reconstruct)


def reconstruct(streams, q, spec, O):
    """Decoder side: streams -> vhat fp32 [B, T, d], the same ops as the encoder's reconstruction."""
    S = [streams[k] for k in sorted(streams) if k.startswith("scale")]
    return restore(Q.dequantize(streams["codes"], S, q), O, spec.sink, streams.get("sink"), streams.get("outliers"))


# ---------------------------------------------------------------- state across the layers of one chunk

class _CodecState:
    """What carries over from one stored layer to the next within a chunk: the spec, the outlier channels (with
    a per-device copy of their ids), the previous layer's reconstruction and the next expected layer index."""

    def __init__(self, spec, channels, n_layers, compile=False):
        self.spec, self.n_layers, self.compile = Spec.parse(spec), n_layers, compile
        self.channels = channels if channels is not None else OutlierChannels(None, "none")
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
        """First-call calibration: the ids for layer i, already on the device they were measured on."""
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
                                                       spec.quant_for(i), needed, self.compile)
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
        vhat = reconstruct(streams, spec.quant_for(i), spec, self.channel_ids(i, streams["codes"].device))
        xhat = vhat if spec.is_key(i) else self.prev + vhat
        self.prev = xhat
        return xhat


@torch.no_grad()
def encode(X, spec, channels=None):
    """X [L, B, T, d] -> (list of per-layer stream dicts, xhat [L, B, T, d] fp32)."""
    enc = Encoder(spec, channels, X.shape[0])
    out, xh = [], []
    for i in range(X.shape[0]):
        s, x = enc.encode_layer(i, X[i])
        out.append(s)
        xh.append(x)
    return out, torch.stack(xh)


@torch.no_grad()
def decode(streams, spec, channels=None):
    dec = Decoder(spec, channels, len(streams))
    return torch.stack([dec.decode_layer(i, s) for i, s in enumerate(streams)])
