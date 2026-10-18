"""Format 2: rotated blob files + Parquet tables, written by Store and read back by Reader2."""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import rescomp as rc
from rescomp import store as S
from tests.conftest import make_acts

MODEL = {"id": "test/tiny", "revision": "abc", "n_layers": 12, "d_model": 64, "dtype": "bfloat16"}
LAYERS = [8, 9, 10, 11]


def err_rows(X, xh):
    e = (X.double() - xh.double())
    return [(float(e[i].pow(2).sum()), float(X[i].double().pow(2).sum()), float(e[i].abs().max())) for i in range(X.shape[0])]


def write_store(stem, spec="int4_g16+layerK2+out2", rotate_bytes=4 << 30, chunks=((0, 3), (3, 5), (8, 2))):
    """Every sequence lives in one chunk here (the one-shot case)."""
    N = sum(n for _, n in chunks)
    acts = make_acts(L=4, B=N, T=8, d=64, seed=3)
    tokens = torch.randint(0, 50000, (N, 8))
    spec = rc.Spec.parse(spec)
    cal = rc.calibrate(acts[:, :4], spec)
    expect = []
    with S.Store(stem, model=MODEL, layers=LAYERS, spec=spec, calib=cal, rotate_bytes=rotate_bytes,
                 hook_names={8: {"hf": "transformer.h.8"}}, extra={"note": "x"}) as st:
        for a, n in chunks:
            X = acts[:, a: a + n]
            streams, xh = rc.encode(X, spec, cal)
            ids = [st.new_sequence(tokens[a + j], mask=[1] * 8, doc_id=10 * (a + j), tag=f"s{a + j}",
                                   vec=np.arange(3) + a + j) for j in range(n)]
            st.add_chunk(ids, [0] * n, streams, err_rows(X, xh))
            st.note_raw(X.numel() * 2, 64)
            expect.append(xh)
    return acts, tokens, torch.cat(expect, 1)


def test_roundtrip(tmp_path):
    stem = str(tmp_path / "run")
    acts, tokens, expect = write_store(stem)
    assert S.is_v2(stem) and not S.is_v2(str(tmp_path / "nope"))
    assert sorted(os.path.basename(p) for p in os.listdir(tmp_path)) == [
        "run.00000.rsc", "run.blobs.parquet", "run.chunks.parquet", "run.meta.parquet", "run.tokens.parquet"]
    with S.Reader(stem) as r:
        assert r.layers == LAYERS and r.n_seq == 10 and r.d == 64 and r.spec.name == "int4_g16+layerK2+out2"
        assert r.header["model"] == MODEL and r.header["extra"] == {"note": "x"}
        assert r.header["layers"][0]["hook"] == {"hf": "transformer.h.8"} and r.header["layers"][0]["outlier_channels"] == [5, 17]
        assert r.header["n_chunks"] == 3 and r.header["n_tokens"] == 80 and r.header["bytes"]["ratio_vs_bf16"] > 1
        for l in LAYERS:
            got = r.read(l)
            assert len(got) == 10 and all(torch.equal(g, expect[LAYERS.index(l), i]) for i, g in enumerate(got))
        assert torch.equal(r.read(9, 4), expect[1, 4]) and torch.equal(r.read(9, [9, 0])[1], expect[1, 0])
        assert torch.equal(torch.stack(r.tokens()), tokens) and torch.equal(r.tokens(3), tokens[3])
        assert r.attention_mask(0).tolist() == [1] * 8
        assert r.annotations == ["doc_id", "tag", "vec"]
        assert r.annotation("doc_id") == [10 * i for i in range(10)] and r.annotation("tag", 2) == "s2"
        assert torch.equal(r.annotation("vec", 1), torch.tensor([1, 2, 3]))
        padded, mask = r.read_padded(8, [0, 1])
        assert padded.shape == (2, 8, 64) and mask.all()
        assert "run: RSC2 v2" in r.summary() and "annotations: doc_id, tag, vec" in r.summary()
        assert r.sequences().num_rows == 10 and abs(r.header["fidelity"]["rel_l2"] - float((expect - acts).norm() / acts.norm())) < 1e-6
        with pytest.raises(KeyError):
            r.read(7)
        with pytest.raises(IndexError):
            r.read(8, [42])


def test_rotation(tmp_path):
    stem = str(tmp_path / "rot")
    acts, tokens, expect = write_store(stem, rotate_bytes=4000)
    files = sorted(p for p in os.listdir(tmp_path) if p.endswith(".rsc"))
    assert len(files) > 1 and all(os.path.getsize(tmp_path / f) >= S.HEAD2 for f in files)
    with S.Reader(stem) as r:
        assert r.header["n_files"] == len(files)
        assert set(b["file"] for v in r.blobs.values() for b in v.values()) == set(range(len(files)))
        assert all(torch.equal(g, expect[3, i]) for i, g in enumerate(r.read(11)))


def test_sequence_spanning_chunks(tmp_path):
    """A prompt chunk and then one-token chunks (generation): read() returns the whole sequence."""
    stem = str(tmp_path / "gen")
    spec = rc.Spec.parse("int8_tok+layerK2")
    full = make_acts(L=4, B=2, T=6, d=64, seed=5)
    toks = torch.arange(12).reshape(2, 6)
    with S.Store(stem, model=MODEL, layers=LAYERS, spec=spec) as st:
        ids = [st.new_sequence(toks[j, :3]) for j in range(2)]          # prompt: 3 tokens
        parts = []
        for a, b in [(0, 3), (3, 4), (4, 5), (5, 6)]:
            X = full[:, :, a:b]
            streams, xh = rc.encode(X, spec)
            st.add_chunk(ids, [a, a], streams, err_rows(X, xh))
            st.note_raw(X.numel() * 2, 64)
            if a > 0:
                for j in range(2):
                    st.extend_sequence(ids[j], toks[j, a:b])
            parts.append(xh)
        st.flush()                                                      # readable before close
        with S.Reader(stem) as r:
            assert r.header["n_chunks"] == 4 and r.read(8, 0).shape == (6, 64)
    expect = torch.cat(parts, 2)
    with S.Reader(stem) as r:
        for l in LAYERS:
            for j in range(2):
                assert torch.equal(r.read(l, j), expect[LAYERS.index(l), j])
        assert torch.equal(r.tokens(1), toks[1]) and r.sequences().column("n_tokens").to_pylist() == [6, 6]
        assert r.annotations == [] and r.attention_mask(0) is None
        padded, mask = r.read_padded(9)
        assert padded.shape == (2, 6, 64)


def test_tokens_optional_and_ragged(tmp_path):
    stem = str(tmp_path / "rag")
    spec = rc.Spec.parse("bf16")
    with S.Store(stem, model=MODEL, layers=LAYERS, spec=spec) as st:
        a = st.new_sequence()                                           # no tokens recorded
        b = st.new_sequence([1, 2, 3])
        X = make_acts(L=4, B=1, T=5, d=64)
        streams, xh = rc.encode(X, spec)
        st.add_chunk([a], [0], streams, err_rows(X, xh))
        X2 = make_acts(L=4, B=1, T=3, d=64, seed=9)
        streams, xh2 = rc.encode(X2, spec)
        st.add_chunk([b], [0], streams, err_rows(X2, xh2))
        st.note_raw(0, 64)
    with S.Reader(stem) as r:
        assert r.tokens(a) is None and r.tokens(b).tolist() == [1, 2, 3]
        assert r.sequences().column("n_tokens").to_pylist() == [5, 3]  # from chunks when no tokens
        padded, mask = r.read_padded(8)
        assert padded.shape == (2, 5, 64) and mask.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
        assert torch.equal(r.read(8, [a])[0], xh[0, 0]) and r.read(8, b).shape == (3, 64)


def test_store_validation(tmp_path):
    with pytest.raises(ValueError):
        S.Store(str(tmp_path / "x"), model=MODEL, layers=[], spec="bf16")
    with pytest.raises(TypeError):
        S.Store(str(tmp_path / "x"), model=MODEL, layers=LAYERS, spec="bf16", extra={"a": np.int64(1)})
    st = S.Store(str(tmp_path / "y"), model=MODEL, layers=LAYERS, spec="bf16")
    with pytest.raises(ValueError):
        st.add_chunk([0], [0], [{}], [])
    st.close()
    st.close()
    with pytest.raises(ValueError):
        st.add_chunk([0], [0], [{}] * 4, [])


def test_corruption_detected(tmp_path):
    stem = str(tmp_path / "c")
    write_store(stem)
    with S.Reader(stem) as r:
        rec = r.blobs[(1, 9)]["codes"]
    p = tmp_path / "c.00000.rsc"
    b = bytearray(p.read_bytes())
    b[rec["offset"] + 3] ^= 0xFF
    p.write_bytes(bytes(b))
    with S.Reader(stem) as r:
        r.read(9, 0)
        with pytest.raises(ValueError, match="crc"):
            r.read(9, 3)



def test_parquet_schema(tmp_path):
    from rescomp.storage import location
    stem = location(str(tmp_path / "s"))
    write_store(stem.path)
    t = pq.read_table(S.table_path(stem, "tokens"))
    assert t.column_names == ["seq_id", "n_tokens", "tokens", "mask", "doc_id", "tag", "vec"]
    assert t.schema.field("tokens").type == pa.list_(pa.int32()) and t.schema.field("vec").type == pa.list_(pa.int64())
    b = pq.read_table(S.table_path(stem, "blobs"))
    assert b.num_rows == 3 * 4 * 3 and set(b.column("stream").to_pylist()) == {"codes", "scale0", "outliers"}
    c = pq.read_table(S.table_path(stem, "chunks"))
    assert c.column("n_seq").to_pylist() == [3, 5, 2] and c.column("seq_ids").to_pylist()[1] == [3, 4, 5, 6, 7]


def test_open(tmp_path):
    stem = str(tmp_path / "v2")
    write_store(stem)
    assert isinstance(rc.open(stem), rc.Reader)
    with pytest.raises(ValueError):
        rc.open(tmp_path / "missing")


def test_transcode(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    acts, tokens, _ = write_store(src, spec="fp32")
    st = rc.transcode(src, dst, "int4_g16+layerK2+out2", calib_seqs=4)
    assert st["chunks"] == 3 and st["ratio"] > 2
    spec = rc.Spec.parse("int4_g16+layerK2+out2")
    with rc.open(dst) as r:
        assert r.spec == spec and r.header["extra"]["source_spec"] == "fp32" and r.n_seq == 10
        assert torch.equal(torch.stack(r.tokens()), tokens) and r.annotation("doc_id") == [10 * i for i in range(10)]
        assert r.header["calib"]["source"].startswith("first ")
        for a, n in ((0, 3), (3, 5), (8, 2)):              # chunk-wise re-encoding with the file's own calibration
            exp = rc.encode(acts[:, a: a + n], spec, r.calib)[1]
            assert torch.equal(torch.stack(r.read(9, list(range(a, a + n)))), exp[1])
    rc.transcode(src, str(tmp_path / "x"), "compact")
    assert rc.open(str(tmp_path / "x")).spec.name == "int4_g8+layerK4+out8"
    with pytest.raises(ValueError):
        rc.Store(str(tmp_path / "z"), model=MODEL, layers=LAYERS).add_chunk([0], [0], [{}] * 4, None)
