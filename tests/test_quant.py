import pytest
import torch

from rescomp import quant as Q


def roundtrip(v, q):
    S = Q.make_scales(v, q)
    c = Q.quantize(v, S, q)
    return Q.dequantize(c, S, q), c, S


@pytest.fixture
def v():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 7, 64, generator=g) * 4
    x[..., 9] *= 300
    return x


def test_fp32_exact(v):
    y, c, S = roundtrip(v, "fp32")
    assert torch.equal(y, v) and S == [] and c.dtype == torch.float32


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
    assert c.dtype == torch.int8 and S[0].shape == (3, 7, 1)
    assert ((y - v).abs() <= S[0].float() / 2 + 1e-5).all()


def test_fp8_bound(v):
    y, c, S = roundtrip(v, "fp8_e4m3")
    assert c.dtype == torch.float8_e4m3fn
    # e4m3 has a 3-bit mantissa: relative step 2^-3, so error <= 1/16 of the value (plus the scale's own rounding)
    assert ((y - v).abs() <= v.abs() * 2 ** -4 + S[0].float() * 2 ** -6 + 1e-5).all()


@pytest.mark.parametrize("q,bits", [("int8_g64", 8), ("int4_g16", 4), ("int2_g16", 2), ("int4_g64", 4)])
def test_int_group_bound_and_packing(v, q, bits):
    y, c, S = roundtrip(v, q)
    G = Q.qinfo(q)["group"]
    assert S[0].shape == (3, 7, 64 // G, 1) and S[0].dtype == torch.float16
    assert c.shape == (3, 7, 64 * bits // 8)
    step = S[0].float().expand(3, 7, 64 // G, G).reshape(3, 7, 64)
    assert ((y - v).abs() <= step / 2 + 1e-4).all()
    assert c.dtype == (torch.int8 if bits == 8 else torch.uint8)


def test_pack_unpack():
    q = torch.randint(0, 16, (5, 32), dtype=torch.uint8)
    assert torch.equal(Q._unpack(Q._pack(q, 4), 4), q)
    q = torch.randint(0, 4, (5, 32), dtype=torch.uint8)
    assert torch.equal(Q._unpack(Q._pack(q, 2), 2), q)


@pytest.mark.parametrize("step", [0.05, 0.5, 2.0])
def test_uabs_bound_and_planes(v, step):
    q = f"uabs{step}"
    y, c, S = roundtrip(v, q)
    assert S == [] and c.shape == (4, 3, 7, 64) and c.dtype == torch.uint8
    assert ((y - v).abs() <= step / 2 + 1e-5).all()
    # ordinary values use only the low plane(s): the top plane is all zero, and so is plane 2 outside the outlier channel
    assert int(c[3].sum()) == 0 and int(c[2][..., [i for i in range(64) if i != 9]].sum()) == 0


def test_uabs_negative_values_zigzag():
    v = torch.tensor([[0.0, -0.1, 0.1, -0.2, 0.2, -1000.0]])
    y, c, _ = roundtrip(v, "uabs0.1")
    assert torch.allclose(y, v, atol=0.05)


@pytest.mark.parametrize("q", ["mxfp4", "nvfp4"])
def test_fp4_bound(v, q):
    y, c, S = roundtrip(v, q)
    G = Q.qinfo(q)["group"]
    assert c.shape == (3, 7, 32)
    s = Q._fp4_scale(S, q).expand(3, 7, 64 // G, G).reshape(3, 7, 64)
    # e2m1 grid spacing is at most 2 (between 4 and 6) in scaled units, so error <= 1 scale unit; mxfp4's power-of-two
    # scale puts the group max anywhere in [4, 8), so values above 6 clip (error up to 2 units)
    assert ((y - v).abs() <= s * (2.0 if q == "mxfp4" else 1.0) + 1e-4).all()
    assert ((y.abs()) <= s * 6 + 1e-4).all()


def test_fp4_codes():
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0, 0.24, 0.26, 5.9, -2.6])
    y = Q._fp4_decode(Q._fp4_encode(x))
    assert torch.equal(y, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0, 0.0, 0.5, 6.0, -3.0]))


def test_check_dim():
    Q.check_dim("int4_g16", 64)
    Q.check_dim("bf16", 7)
    with pytest.raises(ValueError):
        Q.check_dim("int4_g48", 64)
    with pytest.raises(ValueError):
        Q.check_dim("mxfp4", 48)
    with pytest.raises(ValueError):
        Q.check_dim("int4_g0", 64)
    with pytest.raises(ValueError):
        Q.check_dim("int4_g3", 9)        # 2 codes per byte: d must be even
    with pytest.raises(ValueError):
        Q.check_dim("int2_g3", 6)        # 4 codes per byte


def test_mxfp4_scale_is_exact_power_of_two():
    e = torch.arange(1, 255, dtype=torch.uint8).reshape(-1, 1)
    assert torch.equal(Q._fp4_scale([e], "mxfp4"), torch.exp2(e.float() - 127))


def test_qinfo_rejects_unknown():
    with pytest.raises(ValueError):
        Q.qinfo("int5_g8")



def test_zero_group_is_exact_zero():
    v = torch.zeros(2, 3, 32)
    for q in ["int4_g16", "int8_tok", "fp8_e4m3", "mxfp4", "nvfp4", "uabs0.1"]:
        y, _, _ = roundtrip(v, q)
        assert torch.equal(y, v), q
