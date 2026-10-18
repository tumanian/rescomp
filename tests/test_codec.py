import pytest
import torch

import rescomp as rc
from rescomp.codec import Calib, Decoder, Encoder
from tests.conftest import ALL_SPECS, make_acts


@pytest.mark.parametrize("name", ALL_SPECS)
def test_encode_decode_bit_exact(acts, name):
    spec = rc.Spec.parse(name)
    cal = rc.calibrate(acts, spec)
    streams, xhat = rc.encode(acts, spec, cal)
    assert xhat.dtype == torch.float32 and xhat.shape == acts.shape
    xdec = rc.decode(streams, spec, cal)
    assert torch.equal(xhat, xdec)
    assert torch.isfinite(xhat).all()
    rel = ((xhat - acts).norm() / acts.norm()).item()
    assert rel < 0.2, rel


def test_lossless_specs_exact(acts):
    _, xh = rc.encode(acts, "fp32")
    assert torch.equal(xh, acts)
    xb = acts.to(torch.bfloat16)
    _, xh = rc.encode(xb, "bf16")
    assert torch.equal(xh, xb.float())
    _, xh = rc.encode(xb, "bf16+layerK2")           # bf16 deltas of bf16 data are not exactly representable
    assert torch.equal(xh[0], xb[0].float()) and torch.equal(xh[2], xb[2].float())
    assert ((xh - xb.float()).norm() / xb.float().norm()) < 1e-2


def test_calibrate_finds_massive_channels(acts):
    cal = rc.calibrate(acts, "int4_g16+out2")
    assert cal.outliers.shape == (4, 2)
    assert cal.outliers.tolist() == [[5, 17]] * 4
    assert rc.calibrate(acts, "bf16").outliers is None
    with pytest.raises(ValueError):
        rc.calibrate(acts, "int4_g16+out100")


def test_outliers_and_sink_are_exact_to_bf16(acts):
    spec = rc.Spec.parse("int4_g16+out2+sink1")
    cal = rc.calibrate(acts, spec)
    _, xh = rc.encode(acts, spec, cal)
    for c in (5, 17):
        assert torch.equal(xh[..., c], acts[..., c].to(torch.bfloat16).float())
    assert torch.equal(xh[:, :, 0, :], acts[:, :, 0, :].to(torch.bfloat16).float())
    # and the rest is coarse but bounded
    rest = [c for c in range(64) if c not in (5, 17)]
    assert ((xh[:, :, 1:, rest] - acts[:, :, 1:, rest]).abs().max()) < acts[:, :, 1:, rest].abs().max()


def test_outliers_help(acts):
    base = rc.encode(acts, "int4_g16")[1]
    spec = rc.Spec.parse("int4_g16+out2")
    better = rc.encode(acts, spec, rc.calibrate(acts, spec))[1]
    e0, e1 = (base - acts).norm(), (better - acts).norm()
    assert e1 < e0 / 5


def test_layer_delta_reduces_error_on_delta_layers(acts):
    spec = rc.Spec.parse("int8_tok+layerK4+out2")
    cal = rc.calibrate(acts, spec)
    xh = rc.encode(acts, spec, cal)[1]
    x0 = rc.encode(acts, "int8_tok+out2", rc.calibrate(acts, "int8_tok+out2"))[1]
    for i in (1, 2, 3):
        assert (xh[i] - acts[i]).norm() < (x0[i] - acts[i]).norm()


def test_hybrid_keyframes_lossless(acts):
    xb = acts.to(torch.bfloat16)
    spec = rc.Spec.parse("int4_g16+layerK2+key=bf16")
    streams, xh = rc.encode(xb, spec)
    assert torch.equal(xh[0], xb[0].float()) and torch.equal(xh[2], xb[2].float())
    assert not torch.equal(xh[1], xb[1].float())
    assert streams[0]["codes"].dtype == torch.bfloat16 and streams[1]["codes"].dtype == torch.uint8


def test_delta_layers_need_order(acts):
    spec = rc.Spec.parse("int4_g16+layerK2")
    enc = Encoder(spec, None, 4)
    with pytest.raises(ValueError):
        enc.encode_layer(1, acts[1])
    enc.encode_layer(0, acts[0])
    enc.encode_layer(1, acts[1])
    enc.encode_layer(2, acts[2])        # a keyframe can start a new group
    with pytest.raises(IndexError):
        enc.encode_layer(4, acts[0])


def test_decoder_can_start_at_any_keyframe(acts):
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    cal = rc.calibrate(acts, spec)
    streams, xh = rc.encode(acts, spec, cal)
    dec = Decoder(spec, cal, 4)
    assert torch.equal(dec.decode_layer(2, streams[2]), xh[2])
    assert torch.equal(dec.decode_layer(3, streams[3]), xh[3])


def test_fp32_codes_are_not_aliased(acts):
    spec = rc.Spec.parse("fp32+out2+sink1")
    cal = rc.calibrate(acts, spec)
    streams, xh = rc.encode(acts, spec, cal)
    codes = streams[0]["codes"]
    assert codes is not xh[0] and (codes[..., 5] == 0).all() and (codes[:, 0, :] == 0).all()
    assert torch.equal(rc.decode(streams, spec, cal), xh)


def test_calib_ids_validated(acts):
    with pytest.raises(ValueError, match="in \\[0, 64"):
        Encoder("bf16+out2", Calib(torch.tensor([[1, 64]] * 4)), 4).encode_layer(0, acts[0])
    with pytest.raises(ValueError, match="repeat"):
        Encoder("bf16+out2", Calib(torch.tensor([[3, 3]] * 4)), 4).encode_layer(0, acts[0])
    enc = Encoder("bf16", None, 4)
    enc.encode_layer(0, acts[0])
    with pytest.raises(ValueError, match="d changed"):
        enc.encode_layer(1, acts[1][..., :32])
    with pytest.raises(ValueError, match="non-finite"):
        Encoder("bf16", None, 1).encode_layer(0, acts[0] * float("inf"))


def test_calib_check():
    spec = rc.Spec.parse("int4_g16+out2")
    with pytest.raises(ValueError):
        Encoder(spec, None, 4)
    with pytest.raises(ValueError):
        Encoder(spec, Calib(torch.zeros(4, 3, dtype=torch.long)), 4)
    assert Calib.from_dict(Calib(torch.tensor([[1, 2]]), "x").to_dict()).outliers.tolist() == [[1, 2]]
    assert Calib.from_dict({"outliers": None}).outliers is None


def test_shape_errors(acts):
    with pytest.raises(ValueError):
        Encoder("int4_g16", None, 1).encode_layer(0, acts[0, 0])
    with pytest.raises(ValueError):
        Encoder("int4_g48", None, 1).encode_layer(0, acts[0])
    with pytest.raises(ValueError):
        Encoder("bf16+sink9", None, 1).encode_layer(0, acts[0])


def test_works_on_bf16_input_and_other_batch_sizes():
    x = make_acts(L=2, B=1, T=5, d=32).to(torch.bfloat16)
    spec = rc.Spec.parse("int4_g16+layerK2+out1")
    cal = rc.calibrate(x, spec)
    st, xh = rc.encode(x, spec, cal)
    assert torch.equal(rc.decode(st, spec, cal), xh)



def test_compiled_quantizer_matches_eager(acts):
    """torch.compile must not change the bytes (same ops, fused). CUDA only: inductor's CPU backend needs a
    working OpenMP C++ toolchain, which the development Macs do not have."""
    if not torch.cuda.is_available():
        pytest.skip("compiled quantizer is checked on CUDA")
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    cal = rc.calibrate(acts, spec)
    eager, xe = rc.encode(acts, spec, cal)
    enc = Encoder(spec, cal, acts.shape[0], compile=True)
    comp, xc = [], []
    for i in range(acts.shape[0]):
        s, x = enc.encode_layer(i, acts[i])
        comp.append(s)
        xc.append(x)
    assert torch.equal(torch.stack(xc), xe)
    assert all(torch.equal(eager[i][k], comp[i][k]) for i in range(len(eager)) for k in eager[i])
