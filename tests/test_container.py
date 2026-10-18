import json
import struct

import numpy as np
import pytest
import torch

import rescomp as rc
from rescomp.container import FORMAT, HEAD
from tests.conftest import ALL_SPECS, make_acts

MODEL = {"id": "test/tiny", "revision": "abc", "n_layers": 12, "d_model": 64, "dtype": "bfloat16"}
LAYERS = [8, 9, 10, 11]


def write_file(path, spec, chunks=((0, 3), (3, 5), (8, 2)), annotations=True, **kw):
    """Three uneven chunks of sequences; returns (acts [L, N, T, d], tokens [N, T], ann [N])."""
    N = sum(n for _, n in chunks)
    acts = make_acts(L=4, B=N, T=8, d=64, seed=3)
    tokens = torch.randint(0, 50000, (N, 8))
    doc = torch.arange(N) * 10
    spec = rc.Spec.parse(spec)
    cal = rc.calibrate(acts[:, :4], spec)
    with rc.Writer(path, model=MODEL, layers=LAYERS, spec=spec, calib=cal, seq_len=8, **kw) as w:
        for start, n in chunks:
            ann = {"doc_id": doc[start: start + n], "flags": np.zeros((n, 8), dtype=np.uint8)} if annotations else None
            w.write(acts[:, start: start + n], tokens[start: start + n], annotations=ann)
    return acts, tokens, doc


@pytest.mark.parametrize("name", ALL_SPECS)
def test_roundtrip_all_specs(tmp_path, name):
    p = tmp_path / "a.rsc"
    acts, tokens, _ = write_file(p, name)
    spec = rc.Spec.parse(name)
    cal = rc.calibrate(acts[:, :4], spec)
    with rc.Reader(p) as r:
        assert r.layers == LAYERS and r.n_seq == 10 and r.seq_len == 8 and r.d == 64 and r.n_chunks == 3
        assert r.spec == spec
        assert torch.equal(r.tokens(), tokens)
        # what the reader returns is exactly the encoder's reconstruction, chunk by chunk
        expect = torch.cat([rc.encode(acts[:, a: a + n], spec, cal)[1] for a, n in ((0, 3), (3, 5), (8, 2))], 1)
        assert torch.equal(r.read_stack(), expect)
        h = r.header
        assert h["format"] == "RSC1" and h["spec"] == name and h["capture"]["n_seq"] == 10
        assert h["bytes"]["ratio_vs_bf16"] > (0.4 if "fp32" in name else 1.0)
        rel = ((expect - acts).norm() / acts.norm()).item()
        assert abs(h["fidelity"]["rel_l2"] - rel) < 1e-6
        if name in ("fp32",):
            assert h["fidelity"]["rel_l2"] == 0.0 and h["fidelity"]["max_abs"] == 0.0


def test_partial_reads_preserve_order(tmp_path):
    p = tmp_path / "a.rsc"
    acts, tokens, doc = write_file(p, "int4_g16+layerK2+out2")
    with rc.Reader(p) as r:
        full = r.read(10)
        ids = [9, 0, 4, 3, 9]
        assert torch.equal(r.read(10, ids), full[ids])
        assert torch.equal(r.read(10, slice(2, 7)), full[2:7])
        assert torch.equal(r.read(10, 5), full[[5]])
        assert torch.equal(r.tokens(ids), tokens[ids])
        assert torch.equal(r.annotation("doc_id", ids), doc[ids])
        assert r.annotation("flags").shape == (10, 8) and r.annotations == ["doc_id", "flags"]
        assert r.read(10, []).shape == (0, 8, 64) and r.tokens([]).shape == (0, 8)
        assert r.annotation("flags", []).shape == (0, 8) and r.annotation("doc_id", slice(0, 0)).shape == (0,)
        with pytest.raises(IndexError):
            r.read(10, [10])
        with pytest.raises(KeyError):
            r.read(7)
        with pytest.raises(KeyError):
            r.annotation("nope")


def test_read_one_layer_decodes_only_its_group(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "int4_g16+layerK2+out2")
    with rc.Reader(p) as r:
        calls = []
        orig = r.streams
        r.streams = lambda ci, l: (calls.append(l), orig(ci, l))[1]
        r.read(11, [0])                     # group {10, 11}: must not touch 8 or 9
        assert set(calls) == {10, 11}
        calls.clear()
        r.read(10, [0])                     # cached
        assert calls == []
        r.read(8, [0])
        assert calls == [8]
        r.read(9, [0])
        assert calls == [8, 9]              # resumed from the cached keyframe, 10/11 still cached
        r.read(11, [0])
        assert calls == [8, 9]
        r.read(11, [5])                     # another chunk: decode its group only
        assert calls == [8, 9, 10, 11]
        calls.clear()
        r.read_stack()                      # 3 chunks x 4 layers: each stream read exactly once
        assert sorted(calls) == sorted(LAYERS * 3) and len(calls) == 12


def test_read_dtype_and_stack_subset(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "bf16")
    with rc.Reader(p) as r:
        assert r.read(8, dtype=torch.bfloat16).dtype == torch.bfloat16
        s = r.read_stack([9, 11], slice(0, 2))
        assert s.shape == (2, 2, 8, 64)
        assert torch.equal(s[1], r.read(11, slice(0, 2)))


def test_header_layers_and_summary(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "int4_g16+layerK2+out2+key=bf16", hook_names={8: {"hf": "transformer.h.8"}}, extra={"note": "x"})
    with rc.Reader(p) as r:
        ls = r.header["layers"]
        assert [l["layer"] for l in ls] == LAYERS
        assert [l["keyframe"] for l in ls] == [True, False, True, False]
        assert [l["quant"] for l in ls] == ["bf16", "int4_g16", "bf16", "int4_g16"]
        assert ls[0]["hook"] == {"hf": "transformer.h.8"} and ls[1]["hook"] is None
        assert ls[0]["outlier_channels"] == [5, 17]
        assert 0 < ls[0]["fidelity"]["rel_l2"] < 2 ** -8      # bf16 keyframe: rounding only
        assert r.header["capture"]["extra"] == {"note": "x"} and r.header["model"] == MODEL
        assert r.layer_info(9)["index"] == 1
        s = r.summary()
        assert "int4_g16+layerK2+out2+key=bf16" in s and "doc_id" in s and "layer   9" in s


def test_attention_mask_and_no_annotations(tmp_path):
    p = tmp_path / "a.rsc"
    acts = make_acts(L=4, B=2, T=8)
    with rc.Writer(p, model=MODEL, layers=LAYERS, spec="bf16") as w:
        w.write(acts, torch.zeros(2, 8, dtype=torch.long), attention_mask=torch.tensor([[1] * 8, [1] * 5 + [0] * 3]))
    with rc.Reader(p) as r:
        assert r.attention_mask().tolist() == [[1] * 8, [1] * 5 + [0] * 3]
        assert r.annotations == []
    with rc.Writer(tmp_path / "b.rsc", model=MODEL, layers=LAYERS, spec="bf16") as w:
        w.write(acts, torch.zeros(2, 8, dtype=torch.long))
    assert rc.Reader(tmp_path / "b.rsc").attention_mask() is None


def test_auto_calibration_from_first_chunk(tmp_path):
    p = tmp_path / "a.rsc"
    acts = make_acts(L=4, B=4, T=8)
    with rc.Writer(p, model=MODEL, layers=LAYERS, spec="int4_g16+out2") as w:
        w.write(acts[:, :2], torch.zeros(2, 8, dtype=torch.long))
        w.write(acts[:, 2:], torch.zeros(2, 8, dtype=torch.long))
    with rc.Reader(p) as r:
        assert r.header["calib"] == {"source": "first_chunk", "outliers": [[5, 17]] * 4}


def test_writer_validation(tmp_path):
    acts = make_acts(L=4, B=2, T=8)
    tok = torch.zeros(2, 8, dtype=torch.long)
    with pytest.raises(ValueError):
        rc.Writer(tmp_path / "x.rsc", model=MODEL, layers=[9, 8], spec="bf16")
    with pytest.raises(ValueError):
        rc.Writer(tmp_path / "x.rsc", model=MODEL, layers=[8, 8], spec="bf16")
    w = rc.Writer(tmp_path / "a.rsc", model=MODEL, layers=LAYERS, spec="bf16")
    with pytest.raises(ValueError):
        w.write(acts[:3], tok)
    with pytest.raises(ValueError):
        w.write(acts, tok[:1])
    w.write(acts, tok)
    with pytest.raises(ValueError):
        w.write(acts[:, :, :4], tok[:, :4])                  # seq_len changed
    with pytest.raises(ValueError):
        w.write(acts, tok, annotations={"a": torch.zeros(3)})  # wrong first dim
    with pytest.raises(ValueError):
        w.write(acts[:, :0], tok[:0])                            # empty batch
    with pytest.raises(ValueError):
        w.write(acts, tok, annotations={"a": torch.zeros(2)})    # annotation set differs from chunk 0
    with pytest.raises(ValueError):
        w.write(acts, tok, attention_mask=torch.ones(2, 8))      # mask only on some chunks
    with pytest.raises(ValueError, match="uint16"):
        w.write(acts, tok, annotations={"a": np.array([1, 2], dtype=np.uint16)})   # not a storable dtype
    with pytest.raises(ValueError, match="batch size"):
        w.write(acts, tok, annotations={"a": torch.tensor(1)})   # 0-d annotation
    with pytest.raises(ValueError, match="integer"):
        w.write(acts, tok.float())                               # float tokens
    with pytest.raises(ValueError, match="non-finite"):
        w.write(acts * float("nan"), tok)
    assert w.n_seq == 2 and len(w.chunks) == 1 and w.pos == w.chunks[0]["layers"]["11"]["codes"]["offset"] + w.chunks[0]["layers"]["11"]["codes"]["nbytes"]
    w.close()
    with pytest.raises(ValueError):
        w.write(acts, tok)
    w.close()  # idempotent


def test_metadata_must_be_json(tmp_path):
    with pytest.raises(TypeError):
        rc.Writer(tmp_path / "x.rsc", model=MODEL, layers=LAYERS, spec="bf16", extra={"seed": np.int64(0)})
    assert not (tmp_path / "x.rsc").exists()


def test_empty_file_refused(tmp_path):
    p = tmp_path / "a.rsc"
    with pytest.raises(ValueError, match="at least one chunk"):
        with rc.Writer(p, model=MODEL, layers=LAYERS, spec="int4_g16+out2"):
            pass
    with pytest.raises(ValueError, match="not closed"):
        rc.Reader(p)


def test_bad_spec_for_data_writes_nothing(tmp_path):
    acts = make_acts(L=4, B=2, T=8)
    w = rc.Writer(tmp_path / "a.rsc", model=MODEL, layers=LAYERS, spec="int4_g48+sink9")
    with pytest.raises(ValueError):
        w.write(acts, torch.zeros(2, 8, dtype=torch.long))
    assert w.pos == HEAD and w.tok_stored == 0
    w.abort()


def test_write_returns_copy(tmp_path):
    p = tmp_path / "a.rsc"
    with rc.Writer(p, model=MODEL, layers=LAYERS, spec="bf16") as w:
        c = w.write(make_acts(L=4, B=2, T=8), torch.zeros(2, 8, dtype=torch.long))
        c["n_seq"] = 99
    assert rc.Reader(p).chunks[0]["n_seq"] == 2


def test_scalar_index_types(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "bf16")
    with rc.Reader(p) as r:
        full = r.read(8)
        assert torch.equal(r.read(8, np.int64(3)), full[[3]])
        assert torch.equal(r.read(8, torch.tensor(3)), full[[3]])
        assert torch.equal(r.read(8, torch.tensor([3, 1])), full[[3, 1]])
        assert torch.equal(r.read(8, np.array([2, 9])), full[[2, 9]])


def test_noncontiguous_and_gpu_free_write(tmp_path):
    p = tmp_path / "a.rsc"
    acts = make_acts(L=4, B=2, T=8)
    with rc.Writer(p, model=MODEL, layers=LAYERS, spec="int4_g16+layerK2") as w:
        w.write(acts.transpose(2, 3).transpose(2, 3), torch.zeros(2, 8, dtype=torch.long))  # non-contiguous view
    assert torch.equal(rc.Reader(p).read_stack(), rc.encode(acts, "int4_g16+layerK2")[1])


def test_transcode_records_source_fidelity(tmp_path):
    write_file(tmp_path / "lossy.rsc", "int2_g16")
    rc.transcode(tmp_path / "lossy.rsc", tmp_path / "out.rsc", "bf16")
    ex = rc.Reader(tmp_path / "out.rsc").header["capture"]["extra"]
    assert ex["source_spec"] == "int2_g16" and ex["source_rel_l2"] > 0.01


def test_bool_mask_index(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "bf16")
    with rc.Reader(p) as r:
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[2, 7]] = True
        assert torch.equal(r.tokens(mask), r.tokens([2, 7]))
        assert torch.equal(r.read(8, mask.numpy()), r.read(8, [2, 7]))
        with pytest.raises(ValueError):
            r.tokens(torch.ones(3, dtype=torch.bool))


def test_calib_checked_before_file_exists(tmp_path):
    bad = rc.Calib(torch.zeros(3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="calibration is"):
        rc.Writer(tmp_path / "x.rsc", model=MODEL, layers=LAYERS, spec="int4_g16+out2", calib=bad)
    assert not (tmp_path / "x.rsc").exists()
    # a calibration given with a spec that has no outliers is dropped, not stored
    with rc.Writer(tmp_path / "y.rsc", model=MODEL, layers=LAYERS, spec="bf16", calib=rc.calibrate(make_acts(), "bf16+out2")) as w:
        w.write(make_acts(L=4, B=2, T=8), torch.zeros(2, 8, dtype=torch.long))
    assert rc.Reader(tmp_path / "y.rsc").header["calib"]["outliers"] is None


def test_reader_errors_leave_no_open_handle(tmp_path):
    import warnings
    p = tmp_path / "a.rsc"
    p.write_bytes(b"NOPE" + b"\0" * 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        with pytest.raises(ValueError):
            rc.Reader(p)
        import gc
        gc.collect()


def test_transcode_keeps_attention_mask(tmp_path):
    acts = make_acts(L=4, B=3, T=8)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3, [1] * 2 + [0] * 6])
    with rc.Writer(tmp_path / "a.rsc", model=MODEL, layers=LAYERS, spec="bf16") as w:
        w.write(acts, torch.zeros(3, 8, dtype=torch.long), attention_mask=mask)
    rc.transcode(tmp_path / "a.rsc", tmp_path / "b.rsc", "int8_tok")
    assert torch.equal(rc.Reader(tmp_path / "b.rsc").attention_mask(), mask.to(torch.uint8))


def test_format_fixture():
    """A file written by rescomp 0.1.0 (tests/data/fixture_v1.rsc) must keep reading back the same values: a
    self-consistent change to the byte layout, header names or shuffle would otherwise go unnoticed."""
    import hashlib
    from pathlib import Path
    p = Path(__file__).parent / "data" / "fixture_v1.rsc"
    with rc.Reader(p) as r:
        assert r.header["version"] == 1 and r.spec.name == "int4_g16+layerK2+out2" and r.layers == [8, 9, 10, 11]
        assert r.n_seq == 5 and r.seq_len == 8 and r.d == 64 and r.n_chunks == 2
        assert r.header["model"]["id"] == "test/tiny" and r.annotations == ["doc_id"]
        assert r.header["layers"][0]["outlier_channels"] == [5, 17]
        h = hashlib.sha256()
        for t in (r.read_stack(), r.tokens(), r.annotation("doc_id")):
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        assert h.hexdigest()[:16] == FIXTURE_DIGEST


FIXTURE_DIGEST = "57f8eb051c13049b"


def test_unclosed_file_refused(tmp_path):
    p = tmp_path / "a.rsc"
    w = rc.Writer(p, model=MODEL, layers=LAYERS, spec="bf16")
    w.write(make_acts(L=4, B=2, T=8), torch.zeros(2, 8, dtype=torch.long))
    w.f.close()
    with pytest.raises(ValueError, match="not closed"):
        rc.Reader(p)


def test_exception_in_with_block_leaves_no_header(tmp_path):
    p = tmp_path / "a.rsc"
    with pytest.raises(RuntimeError):
        with rc.Writer(p, model=MODEL, layers=LAYERS, spec="bf16") as w:
            w.write(make_acts(L=4, B=2, T=8), torch.zeros(2, 8, dtype=torch.long))
            raise RuntimeError("boom")
    with pytest.raises(ValueError, match="not closed"):
        rc.Reader(p)


def test_corruption_detected(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "int4_g16")
    r = rc.Reader(p)
    blob = r.chunks[1]["layers"]["9"]["codes"]
    r.close()
    b = bytearray(p.read_bytes())
    b[blob["offset"] + 5] ^= 0xFF
    p.write_bytes(bytes(b))
    with rc.Reader(p) as r:
        r.read(9, [0])                       # chunk 0 is fine
        with pytest.raises(ValueError, match="crc"):
            r.read(9, [4])


def test_not_an_rsc_file(tmp_path):
    p = tmp_path / "a.rsc"
    p.write_bytes(b"NOPE" + b"\0" * 16)
    with pytest.raises(ValueError, match="not an RSC1"):
        rc.Reader(p)
    p.write_bytes(FORMAT + struct.pack("<QQ", 20, 10) + b"\0" * 4)
    with pytest.raises(ValueError, match="truncated"):
        rc.Reader(p)


def test_newer_version_refused(tmp_path):
    p = tmp_path / "a.rsc"
    write_file(p, "bf16")
    b = bytearray(p.read_bytes())
    off, n = struct.unpack("<QQ", b[len(FORMAT): HEAD])
    h = json.loads(bytes(b[off: off + n]))
    h["version"] = 99
    nh = json.dumps(h).encode()
    p.write_bytes(bytes(b[:len(FORMAT)]) + struct.pack("<QQ", off, len(nh)) + bytes(b[HEAD:off]) + nh)
    with pytest.raises(ValueError, match="newer"):
        rc.Reader(p)


def test_zstd_level_and_no_shuffle(tmp_path):
    p3 = tmp_path / "a.rsc"
    p19 = tmp_path / "b.rsc"
    praw = tmp_path / "c.rsc"
    write_file(p3, "bf16", level=3)
    write_file(p19, "bf16", level=19)
    write_file(praw, "bf16", level=3, shuffle=False)
    s3, s19, sraw = (rc.Reader(p).header["bytes"]["activations_stored"] for p in (p3, p19, praw))
    assert s19 <= s3 and s3 < sraw
    assert torch.equal(rc.Reader(p3).read(8), rc.Reader(praw).read(8))


def test_ratio_ordering(tmp_path):
    """With the massive channels split out (otherwise absmax codes collapse and 8-bit looks better than 4-bit)."""
    ratios = {}
    for name in ("fp32", "bf16", "int8_tok+out2", "int4_g16+out2", "int2_g16+out2"):
        write_file(tmp_path / f"{name}.rsc", name, annotations=False)
        ratios[name] = rc.Reader(tmp_path / f"{name}.rsc").header["bytes"]["ratio_vs_bf16"]
    assert ratios["fp32"] < ratios["bf16"] < ratios["int8_tok+out2"] < ratios["int4_g16+out2"] < ratios["int2_g16+out2"]
    assert ratios["int4_g16+out2"] > 2.5


def test_numpy_inputs(tmp_path):
    p = tmp_path / "a.rsc"
    acts = make_acts(L=4, B=2, T=8).numpy()
    with rc.Writer(p, model=MODEL, layers=LAYERS, spec="fp16") as w:
        w.write([a for a in acts], np.zeros((2, 8), dtype=np.int64), annotations={"s": np.array(["a", "b"]) == "a"})
    with rc.Reader(p) as r:
        assert r.annotation("s").tolist() == [True, False]
        assert r.read(8).shape == (2, 8, 64)


def test_transcode(tmp_path):
    src, dst = tmp_path / "src.rsc", tmp_path / "dst.rsc"
    acts, tokens, doc = write_file(src, "fp32", hook_names={8: {"hf": "transformer.h.8"}})
    h = rc.transcode(src, dst, "int4_g16+layerK2+out2", calib_seqs=4)
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    cal = rc.calibrate(acts[:, :4], spec)
    with rc.Reader(dst) as r:
        assert r.spec == spec and r.n_chunks == 3 and r.layers == LAYERS
        assert torch.equal(r.tokens(), tokens) and torch.equal(r.annotation("doc_id"), doc)
        expect = torch.cat([rc.encode(acts[:, a: a + n], spec, cal)[1] for a, n in ((0, 3), (3, 5), (8, 2))], 1)
        assert torch.equal(r.read_stack(), expect)
        ex = r.header["capture"]["extra"]
        assert ex["transcoded_from"] == "src.rsc" and ex["source_spec"] == "fp32"
        assert r.header["layers"][0]["hook"] == {"hf": "transformer.h.8"}
        assert r.header["calib"]["source"].startswith("first 4 sequences")
        assert h["bytes"]["ratio_vs_bf16"] > rc.Reader(src).header["bytes"]["ratio_vs_bf16"]
