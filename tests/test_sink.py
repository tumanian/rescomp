"""The Sink on a plain torch model: hooks, pre-hook tokens, both encode modes, generation-style calls, flush."""
import pytest
import torch

import rescomp as rc
from rescomp.sink import Layers, Sink, extract_tokens_generic
from tests.test_generic import D, L, V, ToyLM


class ToyWithForward(ToyLM):
    """forward(input_ids, attention_mask=None): what a pre-hook sees on a real model."""

    def forward(self, input_ids, attention_mask=None, **kw):
        logits, _ = self.run(input_ids, [], {})
        return logits


@pytest.fixture(scope="module")
def lm():
    return ToyWithForward().eval()


def layers_of(lm, idx=(0, 1, 2)):
    return Layers(modules={l: lm.blocks[l] for l in idx}, root=lm, hook_names={l: {"toy": f"blocks.{l}"} for l in idx},
                  model_info={"id": "toy", "d_model": D}, n_layers=L)


def reference(lm, toks, idx=(0, 1, 2)):
    """What the sink's encoder sees: the model's activations as they are."""
    _, acts = lm.run(toks, list(idx), {})
    return torch.stack([acts[l] for l in idx]).detach()


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
            # chunk-wise encoding of exactly these activations, calibrated on the first chunk
            cal = r.calib
            exp = torch.cat([rc.encode(ref[:, :4], spec, cal)[1][i], rc.encode(ref[:, 4:], spec, cal)[1][i]], 0)
            assert torch.equal(got, exp)
        assert r.header["calib"]["source"] == "first_chunk"


def test_generation_style_calls_are_one_sequence(lm, tmp_path):
    """Prompt call [B, 3], then one-token calls: the reader returns one sequence per batch row."""
    stem = str(tmp_path / "gen")
    toks = torch.randint(0, V, (2, 6), generator=torch.Generator().manual_seed(1))
    spec = rc.Spec.parse("int8_tok+layerK2")
    with Sink(stem, layers_of(lm), spec) as sink:
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
        parts = [rc.encode(reference(lm, toks[:, :3]), spec)[1]] + [rc.encode(reference(lm, toks[:, t: t + 1]), spec)[1] for t in range(3, 6)]
        exp = torch.cat(parts, 2)
        assert torch.equal(torch.stack(r.read(0, [0, 1])), exp[0]) and torch.equal(torch.stack(r.read(2, [0, 1])), exp[2])


def test_generation_keeps_per_call_chunks(lm, tmp_path):
    stem = str(tmp_path / "geninl")
    toks = torch.randint(0, V, (1, 5))
    with Sink(stem, layers_of(lm), "bf16") as sink:
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
    with Sink(str(tmp_path / "c"), {0: lm.blocks[0], 2: lm.blocks[2]}, "fp16") as sink:
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
