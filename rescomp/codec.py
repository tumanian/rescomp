"""Layer codec: turns one stored layer [B, T, d] into named streams and back.

Order per layer (fp32 throughout):
    v = x                       at keyframes
    v = x - xhat_prev           otherwise (closed loop: the delta is against the DECODED previous stored layer)
    sink rows out (bf16) -> outlier channels out (bf16) -> quantize the rest -> reconstruct
    xhat = vhat (+ xhat_prev)

Encoder and Decoder share `_reconstruct`, so the encoder's reconstruction and the decoder's are the same tensor
op for op. The container stores the streams; `Reader.read` gives back exactly the encoder's xhat.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from . import quant as Q
from .spec import Spec


@dataclass
class Calib:
    """Per stored layer: the channel ids kept in bf16. outliers is [L, n] int64 or None."""
    outliers: Optional[torch.Tensor] = None
    source: str = "sample"

    def to_dict(self):
        return {"source": self.source,
                "outliers": None if self.outliers is None else self.outliers.tolist()}

    @staticmethod
    def from_dict(d):
        o = d.get("outliers")
        return Calib(None if o is None else torch.tensor(o, dtype=torch.int64), d.get("source", "sample"))

    def check(self, spec, n_layers, d=None):
        if spec.outliers == 0:
            return
        if self.outliers is None:
            raise ValueError(f"spec {spec} needs outlier calibration (rescomp.calibrate)")
        if tuple(self.outliers.shape) != (n_layers, spec.outliers):
            raise ValueError(f"calibration is {tuple(self.outliers.shape)}, need ({n_layers}, {spec.outliers})")
        if d is not None:
            o = self.outliers
            if o.numel() and (o.min() < 0 or o.max() >= d):
                raise ValueError(f"outlier channel ids must be in [0, {d})")
            if any(len(set(row)) != len(row) for row in o.tolist()):
                raise ValueError("outlier channel ids repeat within a layer")


@torch.no_grad()
def calibrate(X, spec, source="sample"):
    """X: [L, N, T, d] activations of the L layers to be stored (any float dtype, any device), in storage order.
    Picks, per stored layer, the spec.outliers channels with the most energy in what gets quantized:
    x at keyframes, x - x_prev at delta layers (open loop; close enough to pick channels). Sink rows (the first
    spec.sink tokens) are stored exactly anyway, so they do not count."""
    spec = Spec.parse(spec)
    if spec.outliers == 0:
        return Calib(None, source)
    L, d = X.shape[0], X.shape[-1]
    if spec.outliers > d:
        raise ValueError(f"out{spec.outliers} > d={d}")
    if spec.sink >= X.shape[2]:
        raise ValueError(f"sink{spec.sink} leaves no tokens to calibrate on (T={X.shape[2]})")
    body = X[:, :, spec.sink:]
    energy = torch.zeros(L, d, dtype=torch.float64)
    for i in range(L):
        v = body[i].double() if spec.is_key(i) else body[i].double() - body[i - 1].double()
        energy[i] = v.pow(2).reshape(-1, d).sum(0).cpu()
    idx = energy.topk(spec.outliers, dim=-1).indices.sort(-1).values
    return Calib(idx, source)


def _reconstruct(streams, q, spec, O):
    """Shared by encoder and decoder: streams -> vhat fp32 [B, T, d]."""
    S = [streams[k] for k in sorted(streams) if k.startswith("scale")]
    vhat = Q.dequantize(streams["codes"], S, q)
    if O is not None:
        vhat[..., O] = streams["outliers"].float()
    if spec.sink:
        vhat[:, : spec.sink, :] = streams["sink"].float()
    return vhat


def _layer_values(x, prev, O, sink, q, reconstruct):
    """The whole tensor work of one layer, free of Python state so torch.compile can fuse it: fp32 cast,
    closed-loop delta, sink rows and outlier channels out, scales, codes, and (if asked) the reconstruction
    with the rows and channels restored. Returns (scales, codes, sink_rows, outlier_values, xhat)."""
    v = x.to(torch.float32, copy=True) if prev is None else x.float() - prev
    sink_rows = outl = None
    if sink:
        sink_rows = v[:, :sink, :].to(torch.bfloat16)
        v[:, :sink, :] = 0
    if O is not None:
        outl = v[..., O].to(torch.bfloat16)
        v[..., O] = 0
    S = Q.make_scales(v, q)
    codes = Q.quantize(v, S, q)
    if not reconstruct:
        return S, codes, sink_rows, outl, None
    vhat = Q.dequantize(codes, S, q)
    if O is not None:
        vhat[..., O] = outl.float()
    if sink:
        vhat[:, :sink, :] = sink_rows.float()
    return S, codes, sink_rows, outl, (vhat if prev is None else prev + vhat)


_COMPILED = {}


def layer_values(x, prev, O, sink, q, reconstruct, compile=False):
    if not compile:
        return _layer_values(x, prev, O, sink, q, reconstruct)
    fn = _COMPILED.get("fn")
    if fn is None:
        fn = _COMPILED["fn"] = torch.compile(_layer_values, dynamic=True)
    return fn(x, prev, O, sink, q, reconstruct)


class _Walker:
    def __init__(self, spec, calib, n_layers, compile=False):
        self.spec, self.n_layers, self.compile = Spec.parse(spec), n_layers, compile
        self.calib = calib if calib is not None else Calib(None, "none")
        self.calib.check(self.spec, n_layers)
        self.d = None
        self.reset()

    def reset(self):
        self.prev, self.expect = None, 0
        if not hasattr(self, "_outl_cache"):
            self._outl_cache = {}

    def _check(self, i, d=None):
        """Order and range checks; with d (encoder side) also the outlier ids and a fixed channel dim."""
        if not 0 <= i < self.n_layers:
            raise IndexError(f"stored layer index {i} out of range [0, {self.n_layers})")
        if not self.spec.is_key(i) and i != self.expect:
            raise ValueError(f"layer {i} is a delta layer: code layers in order (expected {self.expect})")
        if d is not None:
            if self.d is None:
                self.calib.check(self.spec, self.n_layers, d)
                self.d = d
            elif d != self.d:
                raise ValueError(f"d changed from {self.d} to {d}")
        self.expect = i + 1

    def _outl(self, i, dev):
        """Outlier channel ids for layer i on `dev`, copied once: a per-call host-to-device copy of a pageable
        tensor is synchronous and would stall the CPU behind the GPU at every hooked layer."""
        if self.spec.outliers == 0:
            return None
        key = (i, str(dev))
        o = self._outl_cache.get(key)
        if o is None:
            o = self._outl_cache[key] = self.calib.outliers[i].to(dev)
        return o


class Encoder(_Walker):
    """Call encode_layer(i, x) for i = 0, 1, ..., in order within one chunk (keyframes may be coded alone)."""

    @torch.no_grad()
    def encode_layer(self, i, x, reconstruct=True):
        """x [B, T, d] -> (streams: name -> tensor on x's device, xhat fp32 [B, T, d]).
        xhat is also the reference for the next delta layer: do not modify it in place.
        reconstruct=False skips the dequantization when no delta layer follows and the caller does not need
        xhat (returns None); a delta layer always reconstructs. Non-finite inputs are not checked here (a check
        would synchronize the device); the Writer checks them."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, d], got {tuple(x.shape)}")
        self._check(i, x.shape[-1])
        spec = self.spec
        q = spec.quant_for(i)
        Q.check_dim(q, x.shape[-1])
        if spec.sink > x.shape[1]:
            raise ValueError(f"sink{spec.sink} > T={x.shape[1]}")
        key = spec.is_key(i)
        prev = None if key else self.prev.to(x.device)             # layers may live on different GPUs
        needed = reconstruct or (spec.K > 1 and not spec.is_key(i + 1) and i + 1 < self.n_layers)
        S, codes, sink_rows, outl, xhat = layer_values(x, prev, self._outl(i, x.device), spec.sink, q, needed, self.compile)
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


class Decoder(_Walker):
    @torch.no_grad()
    def decode_layer(self, i, streams):
        self._check(i)
        spec = self.spec
        q = spec.quant_for(i)
        dev = streams["codes"].device
        vhat = _reconstruct(streams, q, spec, self._outl(i, dev))
        xhat = vhat if spec.is_key(i) else self.prev + vhat
        self.prev = xhat
        return xhat


@torch.no_grad()
def encode(X, spec, calib=None):
    """X [L, B, T, d] -> (list of per-layer stream dicts, xhat [L, B, T, d] fp32)."""
    enc = Encoder(spec, calib, X.shape[0])
    out, xh = [], []
    for i in range(X.shape[0]):
        s, x = enc.encode_layer(i, X[i])
        out.append(s)
        xh.append(x)
    return out, torch.stack(xh)


@torch.no_grad()
def decode(streams, spec, calib=None):
    dec = Decoder(spec, calib, len(streams))
    return torch.stack([dec.decode_layer(i, s) for i, s in enumerate(streams)])

