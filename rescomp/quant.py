"""Quantizers: one class per format, all built by `quantizer(name)`.

A quantizer turns one fp32 tensor v [..., d] (the channel dim last) into codes plus scales and back:
    scales(v) -> list of scale tensors      encode(v, scales) -> codes      decode(codes, scales) -> fp32
Codes and scales are plain tensors stored with their dtype and shape. decode(encode(v)) is the same on the
encoder and the decoder, and the on-disk bytes are pinned to the library that does the arithmetic (codec version 3):

| name           | class        | arithmetic                                                              |
|----------------|--------------|-------------------------------------------------------------------------|
| fp32/bf16/fp16 | Cast         | torch casts (fp16 clamps to +-65504)                                     |
| fp8_e4m3       | PerTokenFp8  | torch float8_e4m3fn cast, one fp16 scale per token                       |
| int8_tok       | PerTokenInt8 | torchao affine, symmetric no-clipping, [-128, 127], one fp16 scale/token |
| int{4,8}_g<G>  | GroupInt     | torchao affine, symmetric no-clipping, full range, fp16 scale per group  |
| mxfp4          | MXFP4        | torchao to_mx / to_dtype: e2m1 codes, e8m0 scale per 32 channels         |
| nvfp4          | NVFP4        | torchao nvfp4_quantize: e2m1 codes, e4m3 scale per 16, fp32 tensor scale |
"""
import re
from abc import ABC, abstractmethod

import torch
from torchao.prototype.mx_formats import kernels as mxk
from torchao.prototype.mx_formats import mx_tensor, nvfp4_tensor
from torchao.quantization import quant_primitives as qp

CODEC_VERSION = 3
FP16_TINY = 2.0 ** -14
FP16_MAX = 65504.0
MX_ELEM = torch.float4_e2m1fn_x2


class Quantizer(ABC):
    pattern: "re.Pattern"          # matches the names this class builds
    group = None                   # channels per scale (None: per token or none)
    codes_per_byte = 1

    def __init__(self, name):
        self.name = name

    @classmethod
    def matches(cls, name):
        return cls.pattern.fullmatch(name) is not None

    def check_dim(self, d):
        """Raise if this quantizer cannot block a channel dim of d."""
        if self.group is not None and (self.group <= 0 or d % self.group != 0):
            raise ValueError(f"{self.name}: group size {self.group} must divide d={d}")
        if d % self.codes_per_byte != 0:
            raise ValueError(f"{self.name}: {self.codes_per_byte} codes per byte, so d={d} must be a multiple of it")

    @abstractmethod
    def scales(self, v):
        ...

    @abstractmethod
    def encode(self, v, S):
        ...

    @abstractmethod
    def decode(self, codes, S):
        ...

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


def fp16_inexact(x):
    """A 0-d bool on x's device: whether an fp16 cast loses what the model has. bf16 / fp16 inputs: any value that
    changes (range, or bf16 values below fp16's normal range). Wider inputs: only values outside fp16's range
    (above 65504 in magnitude, or non-zero and flushed to zero); rounding to fp16's mantissa is the cast's job."""
    h = x.to(torch.float16).float()
    if x.dtype in (torch.bfloat16, torch.float16):
        return (h != x.float()).any()
    return ((x.abs() > FP16_MAX) | ((x != 0) & (h == 0))).any()


def _half_scale(amax, qmax):
    return (amax / qmax).clamp(min=FP16_TINY).half()


def _block(v, G):
    return (1,) * (v.dim() - 1) + (G,)


class Cast(Quantizer):
    pattern = re.compile(r"fp32|bf16|fp16")

    def scales(self, v):
        return []

    def encode(self, v, S):
        if self.name == "fp32":
            return v.float().clone()
        if self.name == "bf16":
            return v.to(torch.bfloat16)
        return v.clamp(-FP16_MAX, FP16_MAX).to(torch.float16)

    def decode(self, codes, S):
        return codes.clone() if self.name == "fp32" else codes.float()      # never alias the codes stream


class PerTokenFp8(Quantizer):
    pattern = re.compile(r"fp8_e4m3")

    def scales(self, v):
        return [_half_scale(v.abs().amax(-1, keepdim=True), 448)]

    def encode(self, v, S):
        return (v / S[0].float()).clamp(-448, 448).to(torch.float8_e4m3fn)

    def decode(self, codes, S):
        return codes.float() * S[0].float()


class _Affine(Quantizer):
    """torchao affine quantization, symmetric without clipping over the full signed range, fp16 scales."""
    bits = 8

    @property
    def qrange(self):
        return -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1

    def _block(self, v):
        return _block(v, self.group or v.shape[-1])

    def scales(self, v):
        qmin, qmax = self.qrange
        s, _ = qp.choose_qparams_affine(v, qp.MappingType.SYMMETRIC_NO_CLIPPING_ERR, self._block(v), torch.int8,
                                        quant_min=qmin, quant_max=qmax, scale_dtype=torch.float16)
        return [s.reshape(*v.shape[:-1], -1)]                     # always [..., d / G]

    def encode(self, v, S):
        qmin, qmax = self.qrange
        zp = torch.zeros_like(S[0], dtype=torch.int32)
        c = qp.quantize_affine(v, self._block(v), S[0], zp, torch.int8, qmin, qmax)
        return c if self.bits == 8 else mxk.pack_uint4((c.to(torch.int16) - qmin).to(torch.uint8))

    def decode(self, codes, S):
        qmin, qmax = self.qrange
        c = codes if self.bits == 8 else (mxk.unpack_uint4(codes).to(torch.int16) + qmin).to(torch.int8)
        zp = torch.zeros_like(S[0], dtype=torch.int32)
        return qp.dequantize_affine(c, self._block(c), S[0], zp, torch.int8, qmin, qmax, output_dtype=torch.float32)


class PerTokenInt8(_Affine):
    pattern = re.compile(r"int8_tok")


class GroupInt(_Affine):
    pattern = re.compile(r"int([48])_g([1-9]\d*)")

    def __init__(self, name):
        super().__init__(name)
        m = self.pattern.fullmatch(name)
        self.bits, self.group = int(m[1]), int(m[2])
        self.codes_per_byte = 8 // self.bits


class MXFP4(Quantizer):
    pattern = re.compile(r"mxfp4")
    group, codes_per_byte = 32, 2

    def scales(self, v):
        return [mx_tensor.to_mx(v, MX_ELEM, 32)[0]]

    def encode(self, v, S):
        return mx_tensor.to_mx(v, MX_ELEM, 32)[1]

    def decode(self, codes, S):
        return mx_tensor.to_dtype(codes, S[0], MX_ELEM, 32, torch.float32)


class NVFP4(Quantizer):
    pattern = re.compile(r"nvfp4")
    group, codes_per_byte = 16, 2

    def scales(self, v):
        # per-tensor scale clamped away from zero: torchao multiplies by its reciprocal
        gscale = nvfp4_tensor.per_tensor_amax_to_scale(v.abs().amax()).reshape(1).float().clamp(min=1e-30)
        s, _ = nvfp4_tensor.nvfp4_quantize(v.reshape(-1, v.shape[-1]).contiguous(), 16, gscale)
        return [gscale, s.reshape(*v.shape[:-1], -1)]

    def encode(self, v, S):
        _, lp = nvfp4_tensor.nvfp4_quantize(v.reshape(-1, v.shape[-1]).contiguous(), 16, S[0])
        return lp.reshape(*v.shape[:-1], -1)

    def decode(self, codes, S):
        x = mxk.f4_unpacked_to_f32(mxk.unpack_uint4(codes))
        return x * (S[1].float() * S[0]).repeat_interleave(16, -1)


QUANTIZERS = [Cast, PerTokenFp8, PerTokenInt8, GroupInt, MXFP4, NVFP4]
_CACHE = {}


def quantizer(name):
    """Factory: the Quantizer for a name such as "int4_g512" (instances are cached, one per name)."""
    q = _CACHE.get(name)
    if q is None:
        for cls in QUANTIZERS:
            if isinstance(name, str) and cls.matches(name):
                q = _CACHE[name] = cls(name)
                break
        else:
            raise ValueError(f"unknown quantizer {name!r}")
    return q
