import pytest
import torch

from rescomp import quant as Q
from rescomp.quant import quantizer


def roundtrip(v, name):
    q = quantizer(name)
    S = q.scales(v)
    c = q.encode(v, S)
    return q.decode(c, S), c, S


@pytest.fixture
def v():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 7, 64, generator=g) * 4
    x[..., 9] *= 300
    return x


def test_factory():
    assert type(quantizer("int4_g512")).__name__ == "GroupInt" and quantizer("int4_g512").group == 512
    assert quantizer("int4_g512") is quantizer("int4_g512")
    assert type(quantizer("int8_tok")).__name__ == "PerTokenInt8" and quantizer("int8_tok").group is None
    assert type(quantizer("fp16")).__name__ == "Cast" and type(quantizer("nvfp4")).__name__ == "NVFP4"
    for bad in ("int5_g8", "int2_g16", "uabs0.1", "int4_g0", "int4", "fp64", 12):
        with pytest.raises(ValueError):
            quantizer(bad)


def test_fp32_exact(v):
    y, c, S = roundtrip(v, "fp32")
    assert torch.equal(y, v) and S == [] and c.dtype == torch.float32 and y is not c


def test_bf16_fp16_relative_error(v):
    y, _, _ = roundtrip(v, "bf16")
    assert ((y - v).abs() <= v.abs() * 2 ** -8 + 1e-30).all()
    y, c, _ = roundtrip(v, "fp16")
    assert c.dtype == torch.float16
    assert ((y - v).abs() <= v.abs() * 2 ** -11 + 1e-30).all()


def test_fp16_clamps_instead_of_overflowing():
    v = torch.tensor([[1e5, -1e5, 3.0]])
    y, _, _ = roundtrip(v, "fp16")
    assert torch.isfinite(y).all() and y[0, 0] == Q.FP16_MAX and y[0, 1] == -Q.FP16_MAX


def test_int8_tok_bound(v):
    y, c, S = roundtrip(v, "int8_tok")
    assert c.dtype == torch.int8 and S[0].shape == (3, 7, 1) and S[0].dtype == torch.float16
    assert ((y - v).abs() <= S[0].float() / 2 + 1e-3 * S[0].float()).all()        # torchao: no clipping


def test_fp8_bound(v):
    y, c, S = roundtrip(v, "fp8_e4m3")
    assert c.dtype == torch.float8_e4m3fn
    # e4m3 has a 3-bit mantissa: relative step 2^-3, so error <= 1/16 of the value (plus the scale's own rounding)
    assert ((y - v).abs() <= v.abs() * 2 ** -4 + S[0].float() * 2 ** -6 + 1e-5).all()


@pytest.mark.parametrize("name,bits", [("int8_g64", 8), ("int4_g16", 4), ("int4_g64", 4)])
def test_int_group_bound_and_packing(v, name, bits):
    y, c, S = roundtrip(v, name)
    G = quantizer(name).group
    assert S[0].shape == (3, 7, 64 // G) and S[0].dtype == torch.float16
    assert c.shape == (3, 7, 64 * bits // 8) and c.dtype == (torch.int8 if bits == 8 else torch.uint8)
    step = S[0].float().repeat_interleave(G, -1)
    assert ((y - v).abs() <= step / 2 + 1e-3 * step).all()          # no clipping: every value within half a step
    # the grid is torchao's: scale = max(-min / qmin, max / qmax) per group, then fp16
    g = v.unflatten(-1, (-1, G))
    qmin, qmax = quantizer(name).qrange
    expect = torch.maximum(g.amin(-1) / qmin, g.amax(-1) / qmax).clamp(min=torch.finfo(torch.float32).eps).half()
    assert torch.equal(S[0], expect)


def test_pack_unpack():
    from torchao.prototype.mx_formats import kernels as mxk
    q = torch.randint(0, 16, (5, 32), dtype=torch.uint8)
    assert torch.equal(mxk.unpack_uint4(mxk.pack_uint4(q)), q)


@pytest.mark.parametrize("name", ["mxfp4", "nvfp4"])
def test_fp4_bound(v, name):
    y, c, S = roundtrip(v, name)
    assert c.shape == (3, 7, 32) and c.dtype == torch.uint8
    s = (S[1].float() * S[0]).repeat_interleave(16, -1) if name == "nvfp4" else S[0].float().repeat_interleave(32, -1)
    # e2m1 grid spacing is at most 2 (between 4 and 6) in scaled units, so error <= 1 scale unit; mxfp4's power-of-two
    # scale puts the group max anywhere in [4, 8), so values above 6 clip (error up to 2 units); nvfp4's e4m3 block
    # scale rounds, so allow its 1/16 relative step on top
    assert ((y - v).abs() <= s * (2.0 if name == "mxfp4" else 1.0 + 1 / 8) + 1e-4).all()
    assert ((y.abs()) <= s * 6 + 1e-4).all()
    assert S[0].dtype == (torch.float8_e8m0fnu if name == "mxfp4" else torch.float32)


def test_fp4_matches_torchao_reference(v):
    from torchao.prototype.mx_formats import mx_tensor
    y, c, S = roundtrip(v, "mxfp4")
    assert torch.equal(y, mx_tensor.to_dtype(c, S[0], Q.MX_ELEM, 32, torch.float32))
    y, c, S = roundtrip(v, "nvfp4")
    assert S[1].dtype == torch.float8_e4m3fn and S[1].shape == (3, 7, 4) and S[0].shape == (1,)


def test_check_dim():
    quantizer("int4_g16").check_dim(64)
    quantizer("bf16").check_dim(7)
    quantizer("int8_tok").check_dim(7)
    with pytest.raises(ValueError):
        quantizer("int4_g48").check_dim(64)
    with pytest.raises(ValueError):
        quantizer("mxfp4").check_dim(48)
    with pytest.raises(ValueError):
        quantizer("int4_g3").check_dim(9)        # 2 codes per byte: d must be even


def test_zero_group_is_exact_zero():
    v = torch.zeros(2, 3, 32)
    for name in ["int4_g16", "int8_tok", "fp8_e4m3", "mxfp4", "nvfp4"]:
        y, _, _ = roundtrip(v, name)
        assert torch.equal(y, v), name
