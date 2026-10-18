"""Quantizers. v is an fp32 tensor [..., d]; the last dim is the channel dim.

Each quantizer is three functions: make_scales(v) -> list of scale tensors, quantize(v, scales) -> codes,
dequantize(codes, scales) -> fp32. Codes and scales are plain tensors whose dtype and shape are stored with them.
dequantize(quantize(v)) is the same on the encoder and the decoder (codec version 2).

Block formats come from torchao and the on-disk format is pinned to its numerics:
  int{4,8}_g<G>        torchao affine quantization, MappingType.SYMMETRIC_NO_CLIPPING_ERR over the full range
                       [-2^(b-1), 2^(b-1)-1] with one fp16 scale per group of G channels (the scale is the larger
                       of -min/qmin and max/qmax, so nothing clips and the extra negative level is used when it
                       helps); 4-bit codes packed with torchao's pack_uint4, int8 stored as is
  mxfp4                OCP MX: e2m1 codes, one e8m0 scale per 32 channels (torchao to_mx / to_dtype)
  nvfp4                NVIDIA FP4: e2m1 codes, one e4m3 scale per 16 channels and one fp32 per-tensor scale
                       (torchao nvfp4_quantize; dequantized with its f4 table)
Torch-native:
  fp32 / bf16 / fp16   plain casts (fp16 clamps to +-65504 first)
  fp8_e4m3             per-token fp16 scale, e4m3 codes
  int8_tok             per-token fp16 scale, int8 codes in [-127, 127]
  uabs<step>           fixed step in activation units: round(v / step), zigzag-coded, stored as 4 byte planes
                       (entropy coder friendly: the high planes are almost all zero); |v| must be < 2^30 steps
"""
import re

import torch
from torchao.prototype.mx_formats import kernels as mxk
from torchao.prototype.mx_formats import mx_tensor, nvfp4_tensor
from torchao.quantization import quant_primitives as qp

CODEC_VERSION = 2
FP16_TINY = 2.0 ** -14
FP16_MAX = 65504.0
MX_ELEM = torch.float4_e2m1fn_x2


def qinfo(q):
    if q in ("fp32", "bf16", "fp16", "fp8_e4m3", "int8_tok"):
        return {"kind": q}
    if q == "nvfp4":
        return {"kind": "nvfp4", "bits": 4, "group": 16}
    if q == "mxfp4":
        return {"kind": "mxfp4", "bits": 4, "group": 32}
    if q.startswith("uabs"):
        return {"kind": "uabs", "step": float(q[4:])}
    m = re.fullmatch(r"int([48])_g(\d+)", q)
    if not m:
        raise ValueError(f"unknown quantizer {q!r}")
    b = int(m[1])
    return {"kind": "int", "bits": b, "group": int(m[2]), "qmin": -(2 ** (b - 1)), "qmax": 2 ** (b - 1) - 1}


def check_dim(q, d):
    """Raise if quantizer q cannot code a channel dim of d."""
    i = qinfo(q)
    g = i.get("group")
    if g is not None and (g <= 0 or d % g != 0):
        raise ValueError(f"{q}: group size {g} must divide d={d}")
    per = 2 if i.get("bits") == 4 else 1
    if d % per != 0:
        raise ValueError(f"{q}: {i['bits']}-bit codes pack {per} per byte, so d={d} must be a multiple of {per}")


def _half_scale(amax, qmax):
    return (amax / qmax).clamp(min=FP16_TINY).half()


def _block(v, G):
    return (1,) * (v.dim() - 1) + (G,)


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
        s, _ = qp.choose_qparams_affine(v, qp.MappingType.SYMMETRIC_NO_CLIPPING_ERR, _block(v, i["group"]), torch.int8,
                                        quant_min=i["qmin"], quant_max=i["qmax"], scale_dtype=torch.float16)
        return [s.reshape(*v.shape[:-1], -1)]                      # always [..., d / G], even when G == d
    if k == "nvfp4":
        gscale = nvfp4_tensor.per_tensor_amax_to_scale(a.amax()).reshape(1).float().clamp(min=1e-30)   # keeps torchao's reciprocal finite for an all-zero input
        s, _ = nvfp4_tensor.nvfp4_quantize(v.reshape(-1, v.shape[-1]).contiguous(), 16, gscale)
        return [gscale, s.reshape(*v.shape[:-1], -1)]
    if k == "mxfp4":
        s, _ = mx_tensor.to_mx(v, MX_ELEM, 32)
        return [s]
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
    if k == "int":
        b, qmin, qmax = i["bits"], i["qmin"], i["qmax"]
        zp = torch.zeros_like(S[0], dtype=torch.int32)
        c = qp.quantize_affine(v, _block(v, i["group"]), S[0], zp, torch.int8, qmin, qmax)
        if b == 8:
            return c
        return mxk.pack_uint4((c.to(torch.int16) - qmin).to(torch.uint8))
    if k == "nvfp4":
        _, lp = nvfp4_tensor.nvfp4_quantize(v.reshape(-1, v.shape[-1]).contiguous(), 16, S[0])
        return lp.reshape(*v.shape[:-1], -1)
    if k == "mxfp4":
        _, lp = mx_tensor.to_mx(v, MX_ELEM, 32)
        return lp
    raise ValueError(q)


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
        b, qmin, qmax = i["bits"], i["qmin"], i["qmax"]
        codes = c if b == 8 else (mxk.unpack_uint4(c).to(torch.int16) + qmin).to(torch.int8)
        zp = torch.zeros_like(S[0], dtype=torch.int32)
        return qp.dequantize_affine(codes, _block(codes, i["group"]), S[0], zp, torch.int8, qmin, qmax,
                                    output_dtype=torch.float32)
    if k == "nvfp4":
        x = mxk.f4_unpacked_to_f32(mxk.unpack_uint4(c))
        scales = (S[1].float() * S[0]).repeat_interleave(16, -1)
        return x * scales
    if k == "mxfp4":
        return mx_tensor.to_dtype(c, S[0], MX_ELEM, 32, torch.float32)
    raise ValueError(q)

