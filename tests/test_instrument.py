"""Instrumentation: Patcher (store-backed and live), measurements through a plain run() callable."""
import math

import pytest
import torch

import rescomp as rc
from rescomp import instrument as I
from rescomp.sink import Layers, Sink
from tests.test_generic import D, L, V
from tests.test_sink import ToyWithForward, layers_of


@pytest.fixture(scope="module")
def lm():
    return ToyWithForward().eval()


@pytest.fixture(scope="module")
def toks():
    return torch.randint(0, V, (8, 7), generator=torch.Generator().manual_seed(4))


def run_of(lm):
    return lambda t: lm(t)


def test_identity_patch_is_exact(lm, toks):
    ident = lambda layer, x, s, st: x
    k = I.kl_ppl(run_of(lm), toks, I.Patcher(layers_of(lm), ident), batch=3)
    assert k["kl_mean"] == 0.0 and k["top1_agree"] == 1.0 and k["ppl_ratio"] == 1.0 and k["n_seq"] == 8


def test_live_codec_patcher(lm, toks):
    run = run_of(lm)
    kl = {}
    for spec in ("fp16", "int8_tok", "int2_g16+layerK2"):
        kl[spec] = I.kl_ppl(run, toks, I.Patcher(layers_of(lm), I.LiveCodec(spec)), batch=4)["kl_mean"]
    assert kl["fp16"] < kl["int8_tok"] < kl["int2_g16+layerK2"]
    per = I.per_layer(run, toks, layers_of(lm), I.LiveCodec("int8_tok"), batch=4)
    assert list(per) == [0, 1, 2] and all(0 < v["kl_mean"] < 1 for v in per.values())


def test_store_backed_patcher_matches_live(lm, toks, tmp_path):
    """Patching from the file gives the same logits as patching live with the same spec on the same data."""
    stem = str(tmp_path / "s")
    spec = "int4_g16+layerK2+out2"
    with Sink(stem, layers_of(lm), spec, encode="inline") as sink:
        lm(toks[:4])
        lm(toks[4:])
    r = rc.open(stem)
    run = run_of(lm)
    # stored reconstructions were made from clean inputs, so patching every layer from the file only changes the
    # logits through the last one: it equals a live patch of that layer alone (a keyframe, same spec and calib)
    from_file = I.kl_ppl(run, toks, I.Patcher(layers_of(lm), r), seq_ids=list(range(8)), batch=4)
    last_only = I.kl_ppl(run, toks, I.Patcher(layers_of(lm), I.LiveCodec(spec, r.calib), only=[2]), batch=4)
    live_all = I.kl_ppl(run, toks, I.Patcher(layers_of(lm), I.LiveCodec(spec, r.calib)), batch=4)
    assert abs(from_file["kl_mean"] - last_only["kl_mean"]) < 1e-6 and from_file["kl_mean"] > 0
    assert live_all["kl_mean"] > from_file["kl_mean"]                   # errors compound in the live chain
    # a token window of a stored sequence
    with I.Patcher(layers_of(lm), r).sequences([2], starts=[3]):
        out = lm(toks[2:3, 3:6])
    assert out.shape == (1, 3, V)
    with pytest.raises(ValueError, match="needs patcher.sequences"):
        with I.Patcher(layers_of(lm), r):
            lm(toks[:1])
    with pytest.raises(ValueError, match="stored tokens"):
        with I.Patcher(layers_of(lm), r).sequences([0], starts=[5]):
            lm(toks[:1])
    with pytest.raises(ValueError, match="not stored"):
        I.Patcher(Layers(modules={5: lm.blocks[0]}), r)


def test_patcher_only_and_cleanup(lm, toks):
    p = I.Patcher(layers_of(lm), I.LiveCodec("int2_g16"), only=[2])
    assert p.order == [2]
    with p:
        a = lm(toks[:2])
    b = lm(toks[:2])
    assert not torch.equal(a, b) and all(len(blk._forward_hooks) == 0 for blk in lm.blocks)
    with pytest.raises(ValueError):
        I.Patcher(layers_of(lm), I.LiveCodec("bf16"), only=[9])


def test_coherence(lm):
    out = I.coherence(run_of(lm), [[1, 2, 3], [4, 5]], I.Patcher(layers_of(lm), I.LiveCodec("int4_g16+layerK2")),
                      n_new=5, seeds=2)
    s = out["summary"]
    assert s["n_pairs"] == 4 and abs(s["ppl_ratio"] - math.exp(s["log_ppl_diff"])) < 1e-9
    z = I.coherence(run_of(lm), [[1, 2, 3]], I.Patcher(layers_of(lm), I.LiveCodec("fp32")), n_new=4, seeds=2)["summary"]
    assert z["log_ppl_diff"] == 0.0 and z["t"] is None


def test_hf_patcher(tiny_model, fake_tok, toks):
    """Same measurement on a Hugging Face model: run() is just the model call."""
    L = rc.hf_layers(tiny_model, [1, 2, 3], fake_tok)
    run = lambda t: tiny_model(t, use_cache=False).logits
    t = torch.randint(2, 256, (6, 9), generator=torch.Generator().manual_seed(1))
    k = I.kl_ppl(run, t, I.Patcher(L, I.LiveCodec("int8_tok+layerK3")), batch=3)
    assert 0 < k["kl_mean"] < 1
    assert I.kl_ppl(run, t, I.Patcher(L, lambda l, x, s, st: x), batch=3)["kl_mean"] == 0.0
