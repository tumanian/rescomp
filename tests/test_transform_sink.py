"""Pooled rows for probes: the transforms, the TransformSink on the toy models, Reader.pooled on both kinds of
capture."""
import pytest
import torch

import rescomp as rc
from rescomp.transforms import Last, Mean, Stacked, transform
from tests.test_generic import D, ToyLM, V
from tests.test_sink import layers_of
from tests.test_sink_hf import hidden


@pytest.fixture(scope="module")
def lm():
    return ToyLM().eval()


def masked_reference(x, mask):
    """[B, T, d], [B, T] -> mean over real tokens and the last real token, [B, 2, d], the slow way."""
    rows = []
    for xi, mi in zip(x, mask):
        real = xi[mi.bool()]
        rows.append(torch.stack([real.mean(0), real[-1]]))
    return torch.stack(rows)


def random_mask(B, T, seed, side="right"):
    g = torch.Generator().manual_seed(seed)
    n = torch.randint(1, T + 1, (B,), generator=g)
    pos = torch.arange(T)[None]
    return (pos < n[:, None]).long() if side == "right" else (pos >= (T - n)[:, None]).long()


# -- the transforms

def test_transforms_on_tensors():
    x = torch.randn(4, 7, 8)
    for side in ("right", "left"):
        m = random_mask(4, 7, 3, side)
        ref = masked_reference(x, m)
        assert torch.allclose(transform("mean")(x, m)[:, 0], ref[:, 0], atol=1e-6)
        assert torch.equal(transform("last")(x, m)[:, 0], ref[:, 1])
        both = transform("mean+last")(x, m)
        assert both.shape == (4, 2, 8) and torch.equal(both[:, 1], ref[:, 1])
    assert torch.equal(transform("last")(x)[:, 0], x[:, -1]) and torch.allclose(transform("mean")(x)[:, 0], x.mean(1))
    assert transform("mean+last").rows == ["mean", "last"] and transform("mean").rows == ["mean"]
    assert transform("mean+last").to_dict() == {"name": "mean+last", "rows": ["mean", "last"]}
    assert isinstance(transform("last+mean"), Stacked) and isinstance(transform("mean"), Mean) and isinstance(transform(Last()), Last)
    assert transform("mean+last")(x.to(torch.bfloat16), None).dtype == torch.float32
    with pytest.raises(ValueError):
        transform("median")
    with pytest.raises(ValueError):
        transform("mean+mean")
    with pytest.raises(ValueError):
        transform("mean")(x, torch.ones(4, 6))


# -- the TransformSink on the toy LM (MLP-only blocks: each token's activation is its own, so a padded batch
# has the same real-token activations as the unpadded one)

def test_pooled_rows_are_the_masked_mean_and_last(lm, tmp_path):
    stem = str(tmp_path / "pool")
    toks = torch.randint(0, V, (5, 9), generator=torch.Generator().manual_seed(0))
    mask = random_mask(5, 9, 1)
    with rc.TransformSink(stem, layers_of(lm)) as sink:                   # the default codec, fp32
        sink.annotate(doc=[10, 11, 12, 13, 14], split="train")
        lm(toks, attention_mask=mask)
        st = sink.stats()
        assert st["calls"] == 1 and st["chunks"] == 0 and st["pending_sequences"] == 5 and st["rows"] == ["mean", "last"]
    ref = lm.activations(toks, (0, 1, 2))                      # [L, B, T, d] from the model itself
    with rc.open(stem) as r:
        assert r.n_seq == 5 and r.header["n_chunks"] == 1 and r.header["site"] == "resid_post.mean+last"
        assert r.pooled_rows == ["mean", "last"] and r.header["extra"]["pooled"] == {"name": "mean+last", "rows": ["mean", "last"]}
        assert torch.equal(torch.stack(r.tokens()), toks) and torch.equal(torch.stack(r.attention_mask()).long(), mask)
        assert r.annotation("doc") == [10, 11, 12, 13, 14] and r.annotation("split", 4) == "train"
        for i, l in enumerate((0, 1, 2)):
            exp = masked_reference(ref[i], mask)
            assert torch.equal(r.pooled(l, "mean"), exp[:, 0])            # fp32 storage: bit for bit
            assert torch.equal(r.pooled(l, "last"), exp[:, 1])
            assert torch.equal(torch.stack(r.read(l)), exp)                # read() gives the k rows per sequence
            assert torch.equal(r.pooled(l, "last", seqs=[3, 1]), exp[[3, 1], 1])
        with pytest.raises(KeyError):
            r.pooled(1, "first")
        assert "pooled rows per sequence" in r.summary()


def test_default_codec_is_fp32_and_exact_rounds_to_fp16(lm, tmp_path):
    """The default codec is fp32 (rows bit for bit); the "exact" regime (fp16) rounds the rows."""
    toks = torch.randint(0, V, (3, 6))
    with rc.TransformSink(str(tmp_path / "default"), layers_of(lm)) as sink:
        assert sink.codec.given == "fp32"
        lm(toks)
    with rc.TransformSink(str(tmp_path / "exact"), layers_of(lm), "exact") as sink:
        lm(toks)
    ref = lm.activations(toks, (0, 1, 2))
    with rc.open(str(tmp_path / "default")) as r:
        assert r.spec.name == "fp32"
        assert torch.equal(r.pooled(2, "last"), ref[2][:, -1]) and torch.equal(r.pooled(2, "mean"), ref[2].mean(1))
    with rc.open(str(tmp_path / "exact")) as r:
        assert r.spec.name == "fp16"
        assert torch.equal(r.pooled(2, "last"), ref[2][:, -1].half().float())
        assert torch.equal(r.pooled(2, "mean"), ref[2].mean(1).half().float())


def test_chunks_accumulate_across_calls(lm, tmp_path):
    """chunk_seqs=4: calls of 3 + 3 sequences make one chunk of 6 (the limit is checked after each call), the
    remaining 2 are written at close."""
    stem = str(tmp_path / "acc")
    toks = torch.randint(0, V, (8, 5), generator=torch.Generator().manual_seed(2))
    with rc.TransformSink(stem, layers_of(lm), "fp32", chunk_seqs=4, transform="last") as sink:
        lm(toks[:3])
        assert sink.stats()["pending_sequences"] == 3 and sink.stats()["chunks"] == 0
        lm(toks[3:6])
        assert sink.stats()["pending_sequences"] == 0 and sink.stats()["buffered_bytes"] > 0
        sink.flush()
        assert sink.stats()["chunks"] == 1
        with rc.open(stem) as r:                                     # readable mid-run
            assert r.n_seq == 6 and r.pooled(0, "last").shape == (6, D)
        lm(toks[6:])
    ref = torch.cat([lm.activations(t, (0, 1, 2))[1][:, -1] for t in (toks[:3], toks[3:6], toks[6:])])   # per call: a
    with rc.open(stem) as r:                                                                              # matmul's last bits depend on B
        assert r.header["n_chunks"] == 2 and r.n_seq == 8 and r.pooled_rows == ["last"]
        assert [c["seq_ids"] for c in r.chunks] == [[0, 1, 2, 3, 4, 5], [6, 7]]
        assert torch.equal(r.pooled(1, "last"), ref)
        with pytest.raises(KeyError):
            r.pooled(1, "mean")


def test_continuation_calls_are_skipped(lm, tmp_path):
    """Generation-style one-token calls after a prompt (the Sink's continuation rule) store nothing: the rows
    and the tokens describe the prompt only."""
    stem = str(tmp_path / "cont")
    toks = torch.randint(0, V, (2, 6), generator=torch.Generator().manual_seed(1))
    with rc.TransformSink(stem, layers_of(lm), "fp32") as sink:
        lm(toks[:, :3])
        for t in range(3, 6):
            lm(toks[:, t: t + 1])
        lm(toks[:, :4])                                              # T > 1: two new sequences
        assert sink.stats()["calls"] == 5 and sink.stats()["skipped_calls"] == 3
    with rc.open(stem) as r:
        assert r.n_seq == 4 and r.sequences().column("n_tokens").to_pylist() == [3, 3, 4, 4]
        assert torch.equal(r.tokens(0), toks[0, :3]) and torch.equal(r.tokens(3), toks[1, :4])
        ref3, ref4 = lm.activations(toks[:, :3], (0, 1, 2)), lm.activations(toks[:, :4], (0, 1, 2))
        assert torch.equal(r.pooled(2, "last"), torch.cat([ref3[2][:, -1], ref4[2][:, -1]]))


def test_hf_generate_skips_the_steps(tiny_model, fake_tok, tmp_path):
    """Real HF generation with the KV cache: only the prefill call is pooled; the stored tokens are the prompt."""
    prompt = torch.randint(2, 256, (2, 4), generator=torch.Generator().manual_seed(3))
    with rc.TransformSink(str(tmp_path / "gen"), rc.bind(tiny_model, [1, 3], tokenizer=fake_tok), "fp32") as sink:
        out = tiny_model.generate(prompt, max_new_tokens=5, do_sample=False, pad_token_id=1, use_cache=True)
        st = sink.stats()
    assert out.shape == (2, 9) and st["calls"] == 5 and st["skipped_calls"] == 4
    with rc.open(str(tmp_path / "gen")) as r:
        assert r.n_seq == 2 and r.header["n_chunks"] == 1
        assert torch.equal(torch.stack(r.tokens()), prompt)
        ref = hidden(tiny_model, prompt, [1, 3])                       # [2, B, T, d] fp32
        assert torch.equal(r.pooled(3, "last"), ref[1][:, -1])
        assert torch.allclose(r.pooled(1, "mean"), ref[0].mean(1), atol=1e-6)


def test_pooled_on_a_full_capture_matches(lm, tmp_path):
    """The same forward through a Sink (every token) and a TransformSink: Reader.pooled gives the same rows."""
    toks = torch.randint(0, V, (4, 7), generator=torch.Generator().manual_seed(5))
    mask = random_mask(4, 7, 6, "left")
    with rc.Sink(str(tmp_path / "full"), layers_of(lm), "fp32"):
        lm(toks, attention_mask=mask)
    with rc.TransformSink(str(tmp_path / "pool"), layers_of(lm), "fp32"):
        lm(toks, attention_mask=mask)
    with rc.open(str(tmp_path / "full")) as full, rc.open(str(tmp_path / "pool")) as pool:
        assert full.pooled_rows is None
        for l in (0, 2):
            for which in ("mean", "last"):
                assert torch.equal(full.pooled(l, which), pool.pooled(l, which))
            assert torch.equal(full.pooled(l, "last", seqs=2), pool.pooled(l, "last", seqs=[2]))
        with pytest.raises(ValueError):
            full.pooled(0, "mean+last")
        with pytest.raises(ValueError):
            full.pooled(0, "median")


def test_lossy_codec_chooses_channels_on_the_first_chunk(lm, tmp_path):
    stem = str(tmp_path / "lossy")
    toks = torch.randint(0, V, (6, 5))
    with rc.TransformSink(stem, layers_of(lm), "int8_tok+out4", chunk_seqs=4, fidelity=True):
        lm(toks[:4])
        lm(toks[4:])
    with rc.open(stem) as r:
        assert r.header["n_chunks"] == 2 and r.header["outlier_channels"]["source"] == "first_chunk"
        assert r.channels.ids.shape == (3, 4) and 0 < r.header["fidelity"]["rel_l2"] < 1e-2
        ref = lm.activations(toks, (0, 1, 2))
        rel = (r.pooled(2, "last") - ref[2][:, -1]).norm() / ref[2][:, -1].norm()
        assert rel < 1e-2
