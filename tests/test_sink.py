"""The Sink on a plain torch model: hooks, pre-hook tokens, both encode modes, generation-style calls, flush."""
import pytest
import torch

import rescomp as rc
from rescomp.bf16_wrapper import choose_channels
from rescomp.hooks import extract_tokens_generic
from rescomp.sink import Sink
from tests.test_generic import D, L, ToyLM, V


@pytest.fixture(scope="module")
def lm():
    return ToyLM().eval()


def layers_of(lm, idx=(0, 1, 2)):
    return rc.bind({l: lm.blocks[l] for l in idx}, root=lm, hook_names={l: {"toy": f"blocks.{l}"} for l in idx},
                   info={"d_model": D, "n_layers": L}, model_id="toy")


def reference(lm, toks, idx=(0, 1, 2)):
    """What the sink's encoder sees: the model's activations as they are."""
    return lm.activations(toks, idx)


def test_forward_calls_become_chunks(lm, tmp_path):
    stem = str(tmp_path / "fwd")
    toks = torch.randint(0, V, (7, 9), generator=torch.Generator().manual_seed(0))
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    with Sink(stem, layers_of(lm), spec) as sink:
        sink.annotate(doc=[10, 11, 12, 13], split="train")
        lm(toks[:4], attention_mask=torch.ones(4, 9))
        lm(input_ids=toks[4:])
        assert sink.stats()["calls"] == 2
    with rc.open(stem) as r:
        assert r.n_seq == 7 and r.header["n_chunks"] == 2
        assert r.header["model"]["id"] == "toy" and r.header["layers"][0]["hook"] == {"toy": "blocks.0"}
        assert torch.equal(torch.stack(r.tokens()), toks)
        assert r.attention_mask(0).tolist() == [1] * 9 and r.attention_mask(5) is None
        assert r.annotation("doc") == [10, 11, 12, 13, None, None, None] and r.annotation("split", 2) == "train"
        ref = reference(lm, toks)
        for i, l in enumerate((0, 1, 2)):
            got = torch.stack(r.read(l))
            # chunk-wise encoding of exactly these activations, outlier channels chosen on the first chunk
            cal = r.channels
            exp = torch.cat([rc.Codec(spec, channels=cal).encode(ref[:, :4])[1][i], rc.Codec(spec, channels=cal).encode(ref[:, 4:])[1][i]], 0)
            assert torch.equal(got, exp)
        assert r.header["outlier_channels"]["source"] == "first_call"
        # each layer's outlier channels came from its own activations on the first call (keyframes exactly as the
        # channels chosen offline; delta layers use the closed-loop delta, which can differ by a channel)
        off = choose_channels(ref[:, :4], spec).ids.tolist()
        assert r.channels.ids[0].tolist() == off[0] and r.channels.ids[2].tolist() == off[2]
        assert len(set(r.channels.ids[1].tolist())) == 2


def test_generation_style_calls_are_one_sequence(lm, tmp_path):
    """Prompt call [B, 3], then one-token calls: the reader returns one sequence per batch row."""
    stem = str(tmp_path / "gen")
    toks = torch.randint(0, V, (2, 6), generator=torch.Generator().manual_seed(1))
    spec = rc.Spec.parse("int8_tok+layerK2")
    with Sink(stem, layers_of(lm), spec):
        lm(toks[:, :3])                               # prompt
        for t in range(3, 6):
            lm(toks[:, t: t + 1])                     # one new token per call, no cache signal: continuation by rule
        lm(toks[:, :4])                               # T > 1: a new pair of sequences
    with rc.open(stem) as r:
        assert r.n_seq == 4 and r.sequences().column("n_tokens").to_pylist() == [6, 6, 4, 4]
        assert torch.equal(r.tokens(0), toks[0]) and torch.equal(r.tokens(3), toks[1, :4])
        assert r.header["n_chunks"] == 5                  # one chunk per forward call; stitched on read
        for l in (0, 1, 2):
            assert r.read(l, 1).shape == (6, D)
        # each call was encoded on its own; the stitched sequence is the concatenation of those encodings
        parts = [rc.Codec(spec).encode(reference(lm, toks[:, :3]))[1]] + [rc.Codec(spec).encode(reference(lm, toks[:, t: t + 1]))[1] for t in range(3, 6)]
        exp = torch.cat(parts, 2)
        assert torch.equal(torch.stack(r.read(0, [0, 1])), exp[0]) and torch.equal(torch.stack(r.read(2, [0, 1])), exp[2])


def test_generation_keeps_per_call_chunks(lm, tmp_path):
    stem = str(tmp_path / "geninl")
    toks = torch.randint(0, V, (1, 5))
    with Sink(stem, layers_of(lm), "bf16"):
        lm(toks[:, :2])
        lm(toks[:, 2:3])
        lm(toks[:, 3:4])
    with rc.open(stem) as r:
        assert r.header["n_chunks"] == 3 and r.n_seq == 1 and r.read(2, 0).shape == (4, D)
        assert torch.equal(r.tokens(0), toks[0, :4])


def test_explicit_tokens_and_tokens_off(lm, tmp_path):
    toks = torch.randint(0, V, (2, 5))
    with Sink(str(tmp_path / "a"), layers_of(lm), "bf16") as sink:
        sink.tokens(toks + 1)                          # overrides what the pre-hook would see
        lm(toks)
    with rc.open(str(tmp_path / "a")) as r:
        assert torch.equal(torch.stack(r.tokens()), toks + 1)
    with Sink(str(tmp_path / "b"), layers_of(lm), "bf16", tokens="off") as sink:
        lm(toks)
    with rc.open(str(tmp_path / "b")) as r:
        assert r.tokens() == [None, None] and r.read(1, 0).shape == (5, D)


def test_without_root_module(lm, tmp_path):
    """A plain {layer: module} dict: no pre-hook, so no tokens; the first stored layer opens the call."""
    toks = torch.randint(0, V, (3, 4))
    with Sink(str(tmp_path / "c"), {0: lm.blocks[0], 2: lm.blocks[2]}, "fp16"):
        lm(toks)
        lm(toks[:2])
    with rc.open(str(tmp_path / "c")) as r:
        assert r.layers == [0, 2] and r.n_seq == 5 and r.tokens(0) is None
        assert torch.equal(r.read(2, 4), reference(lm, toks[:2], (0, 2))[1, 1].half().float())


def test_buffer_limit_forces_flush(lm, tmp_path):
    toks = torch.randint(0, V, (4, 8))
    with Sink(str(tmp_path / "d"), layers_of(lm), "bf16", buffer_bytes=1) as sink:
        lm(toks)
        assert sink.stats()["buffered_bytes"] == 0 and sink.stats()["chunks"] == 1
        with rc.open(str(tmp_path / "d")) as r:            # readable mid-run after the forced flush
            assert r.n_seq == 4
        lm(toks[:1])
    assert rc.open(str(tmp_path / "d")).n_seq == 5


def test_nothing_written_before_flush(lm, tmp_path):
    toks = torch.randint(0, V, (4, 8))
    with Sink(str(tmp_path / "d2"), layers_of(lm), "bf16") as sink:
        lm(toks)
        assert sink.stats()["chunks"] == 0 and sink.stats()["buffered_bytes"] > 0
        sink.flush()
        assert sink.stats()["chunks"] == 1 and sink.stats()["buffered_bytes"] == 0
    assert rc.open(str(tmp_path / "d2")).n_seq == 4


def test_hooks_removed_on_exit(lm, tmp_path):
    toks = torch.randint(0, V, (1, 4))
    sink = Sink(str(tmp_path / "e"), layers_of(lm), "bf16")
    with sink:
        lm(toks)
    lm(toks)                                               # after exit: not recorded
    assert sink.stats()["calls"] == 1 and all(len(b._forward_hooks) == 0 for b in lm.blocks)
    assert len(lm._forward_pre_hooks) == 0


def test_incomplete_call_counted(lm, tmp_path):
    toks = torch.randint(0, V, (1, 4))
    with Sink(str(tmp_path / "f"), layers_of(lm), "bf16") as sink:
        lm.blocks[0](lm.emb(toks))                         # only layer 0 fires: an incomplete call
        lm(toks)
        assert sink.stats()["incomplete_calls"] == 1 and sink.stats()["calls"] == 1


def test_extract_tokens_generic():
    ids = torch.tensor([[1, 2]])
    assert extract_tokens_generic((ids,), {}) == (ids, None, None)
    assert extract_tokens_generic((), {"input_ids": ids, "attention_mask": ids, "cache_position": torch.tensor([3])})[2] is True
    assert extract_tokens_generic((), {"input_ids": ids, "cache_position": torch.tensor([0])})[2] is False

    class Cache:
        def get_seq_length(self):
            return 5
    assert extract_tokens_generic((), {"input_ids": ids, "past_key_values": Cache()})[2] is True


def test_bad_args(lm, tmp_path):
    with pytest.raises(ValueError):
        Sink(str(tmp_path / "x"), {}, "bf16")
    with pytest.raises(ValueError):
        Sink(str(tmp_path / "x"), layers_of(lm), "bf16", tokens="maybe")
    with Sink(str(tmp_path / "y"), layers_of(lm), "bf16") as sink:
        sink.annotate(doc=[1, 2, 3])
        with pytest.raises(ValueError):
            lm(torch.randint(0, V, (2, 4)))


@pytest.mark.parametrize("regime,expect", [("exact", "fp16"), ("faithful", "int8_tok+out8"),
                                           ("balanced", "int4_g4+layerK4+out8+key=bf16"), ("compact", "int4_g4+layerK4+out8")])
def test_regimes_resolve_on_first_call(lm, tmp_path, regime, expect):
    toks = torch.randint(0, V, (3, 6))
    with Sink(str(tmp_path / regime), layers_of(lm), regime) as sink:
        assert sink.spec is None
        lm(toks)
        assert sink.spec.name == expect                     # d = 32: groups of 4, at least 8 outliers
    with rc.open(str(tmp_path / regime)) as r:
        assert r.spec.name == expect and r.regime == regime and r.n_seq == 3
        assert f"(regime {regime})" in r.summary()


def test_exact_regime_checks_fp16_range(tmp_path):
    """`exact` is fp16: the Sink raises when a value of the model cannot be held exactly, unless told not to."""
    class Scale(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return x * self.k
    model = torch.nn.Sequential(Scale(1.0), Scale(1e5))
    x = torch.randn(2, 4, 32, generator=torch.Generator().manual_seed(0))
    with Sink(str(tmp_path / "ok"), {0: model[0]}, "exact") as sink:
        model(x)                                                      # within fp16's range
    assert sink.check_exact and rc.open(str(tmp_path / "ok")).n_seq == 2
    with pytest.raises(ValueError, match="exact regime .* layer\\(s\\) \\[1\\]"):
        with Sink(str(tmp_path / "big"), {0: model[0], 1: model[1]}, "exact") as sink:
            model(x)
    assert not rc.is_capture(str(tmp_path / "big")) and not list(tmp_path.glob("big.*"))   # nothing half-written
    with Sink(str(tmp_path / "off"), {0: model[0], 1: model[1]}, "exact", check_exact=False) as sink:
        model(x)
    assert rc.open(str(tmp_path / "off")).read(1, 0).abs().max() <= 65504
    with Sink(str(tmp_path / "bf16"), {0: model[0], 1: model[1]}, "bf16") as sink:
        model(x)                                                      # bf16 holds any magnitude
    assert not sink.check_exact
    # a bf16 model: every value must survive the fp16 cast, so a value below fp16's normal range is caught too
    tiny = torch.full((1, 2, 32), 3e-8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exact regime"):
        with Sink(str(tmp_path / "tiny"), {0: model[0]}, "exact") as sink:
            model[0](tiny)
    with Sink(str(tmp_path / "tiny32"), {0: model[0]}, "exact") as sink:
        model[0](tiny.float())                                        # an fp32 model: fp16 rounding is accepted


def test_overwrite_and_reserved_names_reach_the_sink(lm, tmp_path):
    toks = torch.randint(0, V, (2, 4))
    with Sink(str(tmp_path / "w"), layers_of(lm), "bf16") as sink:
        lm(toks)
        with pytest.raises(ValueError, match="reserved"):
            sink.annotate(seq_id=[1, 2])
    with pytest.raises(FileExistsError):
        Sink(str(tmp_path / "w"), layers_of(lm), "bf16")
    with Sink(str(tmp_path / "w"), layers_of(lm), "bf16", overwrite=True) as sink:
        lm(toks[:1])
    assert rc.open(str(tmp_path / "w")).n_seq == 1


def test_regime_helpers():
    from rescomp.regimes import REGIMES, needs_width, resolve
    assert set(REGIMES) == {"exact", "faithful", "balanced", "compact"}
    assert resolve("compact", 4096).name == "int4_g512+layerK4+out32"
    assert resolve("balanced", 768).name == "int4_g96+layerK4+out8+key=bf16"
    assert resolve("int4_g{G}+out{O}", 768).name == "int4_g384+out12"
    assert resolve("bf16").name == "bf16" and not needs_width("bf16") and needs_width("compact")
    with pytest.raises(ValueError):
        resolve("compact")
    with pytest.raises(ValueError):
        resolve("nonsense")


def test_raise_inside_with_discards_unflushed(lm, tmp_path):
    """A `with` body that raises leaves the stem as the last flush published it, or without a capture at all."""
    toks = torch.randint(0, V, (4, 9), generator=torch.Generator().manual_seed(1))
    stem = str(tmp_path / "none")
    with pytest.raises(RuntimeError, match="boom"):
        with Sink(stem, layers_of(lm), "bf16"):
            lm(toks)
            raise RuntimeError("boom")
    assert not rc.is_capture(stem) and not list(tmp_path.glob("none.*"))
    stem = str(tmp_path / "part")
    with pytest.raises(RuntimeError):
        with Sink(stem, layers_of(lm), "bf16") as sink:
            lm(toks[:3])
            sink.flush()
            lm(toks[3:])
            raise RuntimeError("boom")
    with rc.open(stem) as r:
        assert r.n_seq == 3 and r.header["n_chunks"] == 1
    with Sink(str(tmp_path / "after"), layers_of(lm), "bf16"):
        lm(toks)                                                      # the hooks came off: this records 4 sequences
    assert rc.open(str(tmp_path / "after")).n_seq == 4
