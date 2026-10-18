"""Quantizers. v is an fp32 tensor [..., d]; the last dim is the channel dim.

Each quantizer is three functions: make_scales(v) -> list of scale tensors, quantize(v, scales) -> codes,
dequantize(codes, scales) -> fp32. Codes and scales are plain tensors whose dtype and shape are stored with them,
so decoding needs nothing but the tensors and the quantizer name. All arithmetic is fp32 and deterministic, so
dequantize(quantize(v)) is the same on the encoder and the decoder.

  fp32 / bf16 / fp16   plain casts (fp16 clamps to +-65504 first)
  fp8_e4m3             per-token fp16 scale, e4m3 codes
  int8_tok             per-token fp16 scale, int8 codes
  int{2,4,8}_g<G>      symmetric integer codes with one fp16 scale per group of G channels; 2/4-bit codes packed
  uabs<step>           fixed step in activation units: round(v / step), zigzag-coded, stored as 4 byte planes
                       (entropy coder friendly: the high planes are almost all zero); |v| must be < 2^30 steps
  mxfp4                e2m1 codes, one e8m0 (power of two) scale per 32 channels
  nvfp4                e2m1 codes, one e4m3 scale per 16 channels plus one fp32 tensor scale
"""
import re

import torch

FP16_TINY = 2.0 ** -14
FP16_MAX = 65504.0
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def qinfo(q):
    if q in ("fp32", "bf16", "fp16", "fp8_e4m3", "int8_tok"):
        return {"kind": q}
    if q == "nvfp4":
        return {"kind": "nvfp4", "bits": 4, "group": 16}
    if q == "mxfp4":
        return {"kind": "mxfp4", "bits": 4, "group": 32}
    if q.startswith("uabs"):
        return {"kind": "uabs", "step": float(q[4:])}
    m = re.fullmatch(r"int([248])_g(\d+)", q)
    if not m:
        raise ValueError(f"unknown quantizer {q!r}")
    return {"kind": "int", "bits": int(m[1]), "group": int(m[2])}


def check_dim(q, d):
    """Raise if quantizer q cannot code a channel dim of d."""
    i = qinfo(q)
    g = i.get("group")
    if g is not None and (g <= 0 or d % g != 0):
        raise ValueError(f"{q}: group size {g} must divide d={d}")
    per = 8 // i["bits"] if i["kind"] in ("int", "nvfp4", "mxfp4") else 1
    if d % per != 0:
        raise ValueError(f"{q}: {i['bits']}-bit codes pack {per} per byte, so d={d} must be a multiple of {per}")



def _half_scale(amax, qmax):
    return (amax / qmax).clamp(min=FP16_TINY).half()


def _groups(v, G):
    return v.unflatten(-1, (-1, G))


def _pack(q, bits):
    """uint8 codes < 2^bits along the last dim -> packed bytes (first code in the low bits)."""
    if bits == 8:
        return q
    per = 8 // bits
    q = q.unflatten(-1, (-1, per))
    out = q[..., 0].clone()
    for j in range(1, per):
        out |= q[..., j] << (bits * j)
    return out


def _unpack(c, bits):
    if bits == 8:
        return c
    per, mask = 8 // bits, (1 << bits) - 1
    return torch.stack([(c >> (bits * j)) & mask for j in range(per)], -1).flatten(-2)


def _fp4_encode(x):
    idx = torch.bucketize(x.abs().contiguous(), FP4_MIDS.to(x.device))
    return (idx | ((x < 0).to(idx.dtype) << 3)).to(torch.uint8)


def _fp4_decode(c):
    mag = FP4_GRID.to(c.device)[(c & 7).long()]
    return torch.where((c >> 3) == 1, -mag, mag)


def _fp4_scale(S, q):
    if q == "nvfp4":
        return S[1].float() * S[0]
    # 2^(e - 127) built from the exponent bits: exact on every device (exp2 is only guaranteed to 2 ulp on CUDA)
    return (S[0].to(torch.int32) << 23).view(torch.float32)


# ---------------------------------------------------------------- scales

def make_scales(v, q):
    i = qinfo(q)
    k = i["kind"]
    if k in ("fp32", "bf16", "fp16", "uabs"):
        return []
    a = v.abs()
    if k == "int8_tok":
        return [_half_scale(a.amax(-1, keepdim=True), 127)]
    if k == "fp8_e4m3":
        return [_half_scale(a.amax(-1, keepdim=True), 448)]
    if k == "int":
        return [_half_scale(_groups(a, i["group"]).amax(-1, keepdim=True), 2 ** (i["bits"] - 1) - 1)]
    if k == "nvfp4":
        gscale = (a.amax() / (448 * 6)).clamp(min=1e-30).reshape(1).float()
        sg = (_groups(a, 16).amax(-1, keepdim=True) / 6 / gscale).clamp(max=448).to(torch.float8_e4m3fn)
        return [gscale, sg]
    if k == "mxfp4":
        am = _groups(a, 32).amax(-1, keepdim=True)
        # floor(log2(am)) - 2, from the exponent bits so it is exact on every device (log2 is 1 ulp on CUDA)
        e = torch.where(am > 0, torch.frexp(am).exponent - 3, torch.full_like(am, -127, dtype=torch.int32))
        return [(e.clamp(-127, 127) + 127).to(torch.uint8)]
    raise ValueError(q)


# ---------------------------------------------------------------- codes

def quantize(v, S, q):
    i = qinfo(q)
    k = i["kind"]
    if k == "fp32":
        return v.float().clone()
    if k == "bf16":
        return v.to(torch.bfloat16)
    if k == "fp16":
        return v.clamp(-FP16_MAX, FP16_MAX).to(torch.float16)
    if k == "int8_tok":
        return torch.round(v / S[0].float()).clamp(-127, 127).to(torch.int8)
    if k == "fp8_e4m3":
        return (v / S[0].float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    if k == "uabs":
        n = torch.round(v / i["step"])
        if n.abs().max() > 2 ** 30:
            raise ValueError(f"{q}: a value exceeds 2^30 steps; use a larger step")
        n = n.to(torch.int64)
        z = (n << 1) ^ (n >> 63)                                   # zigzag: 0,-1,1,-2,.. -> 0,1,2,3,..
        return torch.stack([((z >> (8 * p)) & 0xFF).to(torch.uint8) for p in range(4)])
    g = _groups(v, i["group"])
    if k == "int":
        b = i["bits"]
        qm = 2 ** (b - 1) - 1
        n = torch.round(g / S[0].float()).clamp(-qm, qm)
        if b == 8:
            return n.to(torch.int8).flatten(-2)
        return _pack((n + qm + 1).to(torch.uint8).flatten(-2), b)
    s = _fp4_scale(S, q)
    x = (g / torch.where(s > 0, s, torch.ones_like(s))).clamp(-6, 6)
    return _pack(_fp4_encode(x).flatten(-2), 4)


def dequantize(c, S, q):
    i = qinfo(q)
    k = i["kind"]
    if k == "fp32":
        return c.clone()                                           # never alias the codes stream
    if k in ("bf16", "fp16"):
        return c.float()
    if k in ("int8_tok", "fp8_e4m3"):
        return c.float() * S[0].float()
    if k == "uabs":
        z = torch.zeros(c.shape[1:], dtype=torch.int64, device=c.device)
        for p in range(4):
            z |= c[p].to(torch.int64) << (8 * p)
        n = (z >> 1) ^ -(z & 1)
        return n.float() * i["step"]
    if k == "int":
        b = i["bits"]
        n = c.float() if b == 8 else _unpack(c, b).float() - 2 ** (b - 1)
        return (_groups(n, i["group"]) * S[0].float()).flatten(-2)
    s = _fp4_scale(S, q)
    return (_groups(_fp4_decode(_unpack(c, 4)), i["group"]) * s).flatten(-2)
