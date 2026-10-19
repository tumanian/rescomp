import json

import pytest
import torch

import rescomp as rc
from rescomp.bf16_wrapper import OutlierChannels, choose_channels
from tests.conftest import ALL_SPECS, make_acts


@pytest.mark.parametrize("name", ALL_SPECS)
def test_encode_decode_bit_exact(acts, name):
    spec = rc.Spec.parse(name)
    cal = choose_channels(acts, spec)
    streams, xhat = rc.Codec(spec, channels=cal).encode(acts)
    assert xhat.dtype == torch.float32 and xhat.shape == acts.shape
    xdec = rc.Codec(spec, channels=cal).decode(streams)
    assert torch.equal(xhat, xdec)
    assert torch.isfinite(xhat).all()
    rel = ((xhat - acts).norm() / acts.norm()).item()
    assert rel < 0.2, rel


def test_lossless_specs_exact(acts):
    _, xh = rc.Codec("fp32").encode(acts)
    assert torch.equal(xh, acts)
    xb = acts.to(torch.bfloat16)
    _, xh = rc.Codec("bf16").encode(xb)
    assert torch.equal(xh, xb.float())
    _, xh = rc.Codec("bf16+layerK2").encode(xb)           # bf16 deltas of bf16 data are not exactly representable
    assert torch.equal(xh[0], xb[0].float()) and torch.equal(xh[2], xb[2].float())
    assert ((xh - xb.float()).norm() / xb.float().norm()) < 1e-2


def test_choose_channels_finds_massive_channels(acts):
    cal = choose_channels(acts, "int4_g16+out2")
    assert cal.ids.shape == (4, 2)
    assert cal.ids.tolist() == [[5, 17]] * 4
    assert choose_channels(acts, "bf16").ids is None
    with pytest.raises(ValueError):
        choose_channels(acts, "int4_g16+out100")


def test_outliers_and_sink_are_exact_to_bf16(acts):
    spec = rc.Spec.parse("int4_g16+out2+sink1")
    cal = choose_channels(acts, spec)
    _, xh = rc.Codec(spec, channels=cal).encode(acts)
    for c in (5, 17):
        assert torch.equal(xh[..., c], acts[..., c].to(torch.bfloat16).float())
    assert torch.equal(xh[:, :, 0, :], acts[:, :, 0, :].to(torch.bfloat16).float())
    # and the rest is coarse but bounded
    rest = [c for c in range(64) if c not in (5, 17)]
    assert ((xh[:, :, 1:, rest] - acts[:, :, 1:, rest]).abs().max()) < acts[:, :, 1:, rest].abs().max()


def test_outliers_help(acts):
    base = rc.Codec("int4_g16").encode(acts)[1]
    spec = rc.Spec.parse("int4_g16+out2")
    better = rc.Codec(spec, channels=choose_channels(acts, spec)).encode(acts)[1]
    e0, e1 = (base - acts).norm(), (better - acts).norm()
    assert e1 < e0 / 5


def test_layer_delta_reduces_error_on_delta_layers(acts):
    spec = rc.Spec.parse("int8_tok+layerK4+out2")
    cal = choose_channels(acts, spec)
    xh = rc.Codec(spec, channels=cal).encode(acts)[1]
    x0 = rc.Codec("int8_tok+out2", channels=choose_channels(acts, "int8_tok+out2")).encode(acts)[1]
    for i in (1, 2, 3):
        assert (xh[i] - acts[i]).norm() < (x0[i] - acts[i]).norm()


def test_hybrid_keyframes_lossless(acts):
    xb = acts.to(torch.bfloat16)
    spec = rc.Spec.parse("int4_g16+layerK2+key=bf16")
    streams, xh = rc.Codec(spec).encode(xb)
    assert torch.equal(xh[0], xb[0].float()) and torch.equal(xh[2], xb[2].float())
    assert not torch.equal(xh[1], xb[1].float())
    assert streams[0]["codes"].dtype == torch.bfloat16 and streams[1]["codes"].dtype == torch.uint8


def test_delta_layers_need_order(acts):
    spec = rc.Spec.parse("int4_g16+layerK2")
    enc = rc.Codec(spec).encoder(4)
    with pytest.raises(ValueError):
        enc.encode_layer(1, acts[1])
    enc.encode_layer(0, acts[0])
    enc.encode_layer(1, acts[1])
    enc.encode_layer(2, acts[2])        # a keyframe can start a new group
    with pytest.raises(IndexError):
        enc.encode_layer(4, acts[0])


def test_decoder_can_start_at_any_keyframe(acts):
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    cal = choose_channels(acts, spec)
    streams, xh = rc.Codec(spec, channels=cal).encode(acts)
    dec = rc.Codec(spec, channels=cal).decoder(4)
    assert torch.equal(dec.decode_layer(2, streams[2]), xh[2])
    assert torch.equal(dec.decode_layer(3, streams[3]), xh[3])


def test_fp32_codes_are_not_aliased(acts):
    spec = rc.Spec.parse("fp32+out2+sink1")
    cal = choose_channels(acts, spec)
    streams, xh = rc.Codec(spec, channels=cal).encode(acts)
    codes = streams[0]["codes"]
    assert codes is not xh[0] and (codes[..., 5] == 0).all() and (codes[:, 0, :] == 0).all()
    assert torch.equal(rc.Codec(spec, channels=cal).decode(streams), xh)


def test_channel_ids_validated(acts):
    with pytest.raises(ValueError, match="in \\[0, 64"):
        rc.Codec("bf16+out2", channels=OutlierChannels(torch.tensor([[1, 64]] * 4))).encoder(4).encode_layer(0, acts[0])
    with pytest.raises(ValueError, match="repeat"):
        rc.Codec("bf16+out2", channels=OutlierChannels(torch.tensor([[3, 3]] * 4))).encoder(4).encode_layer(0, acts[0])
    enc = rc.Codec("bf16").encoder(4)
    enc.encode_layer(0, acts[0])
    with pytest.raises(ValueError, match="d changed"):
        enc.encode_layer(1, acts[1][..., :32])


def test_check_channels():
    spec = rc.Spec.parse("int4_g16+out2")
    with pytest.raises(ValueError):
        rc.Codec(spec).encoder(4)
    with pytest.raises(ValueError):
        rc.Codec(spec, channels=OutlierChannels(torch.zeros(4, 3, dtype=torch.long))).encoder(4)
    assert OutlierChannels.from_dict(OutlierChannels(torch.tensor([[1, 2]]), "x").to_dict()).ids.tolist() == [[1, 2]]
    assert OutlierChannels.from_dict({"ids": None}).ids is None


def test_shape_errors(acts):
    with pytest.raises(ValueError):
        rc.Codec("int4_g16").encoder(1).encode_layer(0, acts[0, 0])
    with pytest.raises(ValueError):
        rc.Codec("int4_g48").encoder(1).encode_layer(0, acts[0])
    with pytest.raises(ValueError):
        rc.Codec("bf16+sink9").encoder(1).encode_layer(0, acts[0])


def test_works_on_bf16_input_and_other_batch_sizes():
    x = make_acts(L=2, B=1, T=5, d=32).to(torch.bfloat16)
    spec = rc.Spec.parse("int4_g16+layerK2+out1")
    cal = choose_channels(x, spec)
    st, xh = rc.Codec(spec, channels=cal).encode(x)
    assert torch.equal(rc.Codec(spec, channels=cal).decode(st), xh)



def test_compiled_quantizer_matches_eager(acts):
    """torch.compile must not change the bytes (same ops, fused). CUDA only: inductor's CPU backend needs a
    working OpenMP C++ toolchain, which the development Macs do not have."""
    if not torch.cuda.is_available():
        pytest.skip("compiled quantizer is checked on CUDA")
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    cal = choose_channels(acts, spec)
    eager, xe = rc.Codec(spec, channels=cal).encode(acts)
    enc = rc.Codec(spec, channels=cal, compile=True).encoder(acts.shape[0])
    comp, xc = [], []
    for i in range(acts.shape[0]):
        s, x = enc.encode_layer(i, acts[i])
        comp.append(s)
        xc.append(x)
    assert torch.equal(torch.stack(xc), xe)
    assert all(torch.equal(eager[i][k], comp[i][k]) for i in range(len(eager)) for k in eager[i])


def test_codec_object(acts, tmp_path):
    """Codec: lossy or lossless, resolves regimes, packs and unpacks, round-trips through a header."""
    c = rc.Codec("compact", level=5)
    assert c.spec is None and c.regime == "compact" and c.lossy and "lossy" in repr(c)
    with pytest.raises(ValueError, match="resolve"):
        c.encoder(4)
    assert c.resolve(64).name == "int4_g8+layerK4+out8" and c.resolve(1024).name == "int4_g8+layerK4+out8"   # idempotent
    assert not rc.Codec("bf16").lossy and not rc.Codec("exact").lossy and rc.Codec("fp16+layerK2+key=int8_tok").lossy
    rec = c.to_dict()
    assert rec["spec"] == "int4_g8+layerK4+out8" and rec["lossless"] == {"codec": "zstd", "level": 5, "shuffle": True}
    cal = choose_channels(acts, c.spec)
    c.channels = cal
    back = rc.Codec.from_header({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in c.to_dict().items()})
    assert back.spec == c.spec and back.regime == "compact" and back.lossless.level == 5
    assert torch.equal(back.channels.ids, cal.ids)
    streams, xhat = c.encode(acts)
    assert torch.equal(back.decode(streams), xhat)
    t = streams[1]["codes"]
    blob, raw = c.pack(t)
    assert raw == t.numel() * t.element_size() and torch.equal(c.unpack(blob, str(t.dtype).removeprefix("torch."), t.shape, raw), t)
    assert rc.Codec(c.spec).spec == c.spec                                    # a Spec object is accepted too
