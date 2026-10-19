"""Captures in S3, against a local moto server: write with bucket/prefix, read back, flush mid-run, rotation,
transcode between buckets, rc.open with a URI."""
import pytest
import torch

import rescomp as rc
from rescomp.sink import Sink
from tests.test_generic import D, V, ToyLM, toy_layers

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")
PORT = 5987


@pytest.fixture(scope="module")
def s3():
    from moto.server import ThreadedMotoServer
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=PORT, verbose=False)
    server.start()
    opts = {"access_key": "testing", "secret_key": "testing", "endpoint_override": f"http://127.0.0.1:{PORT}",
            "scheme": "http", "region": "us-east-1"}
    client = boto3.client("s3", endpoint_url=opts["endpoint_override"], aws_access_key_id="testing",
                          aws_secret_access_key="testing", region_name="us-east-1")
    for b in ("acts", "acts2"):
        client.create_bucket(Bucket=b)
    yield opts, client
    server.stop()


@pytest.fixture(scope="module")
def lm():
    return ToyLM().eval()


def keys(client, bucket, prefix):
    return sorted(o["Key"] for o in client.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []))


def test_sink_to_s3_and_read_back(s3, lm):
    opts, client = s3
    toks = torch.randint(0, V, (6, 7), generator=torch.Generator().manual_seed(0))
    with Sink("run1", toy_layers(lm), "int4_g16+layerK2+out2", bucket="acts", prefix="team/exp", s3=opts) as sink:
        sink.annotate(doc=[1, 2, 3])
        lm(toks[:3])
        lm(toks[3:])
    assert keys(client, "acts", "team/exp/run1") == ["team/exp/run1.00000.rsc", "team/exp/run1.blobs.parquet",
                                                     "team/exp/run1.chunks.parquet", "team/exp/run1.meta.parquet",
                                                     "team/exp/run1.tokens.parquet"]
    assert sink.stats()["files"] == ["acts/team/exp/run1.00000.rsc"]
    for opener in (lambda: rc.open("run1", bucket="acts", prefix="team/exp", s3=opts),
                   lambda: rc.open("s3://acts/team/exp/run1", s3=opts)):
        with opener() as r:
            assert r.stem == "s3://acts/team/exp/run1" and r.n_seq == 6 and r.header["location"] == r.stem
            assert torch.equal(torch.stack(r.tokens()), toks) and r.annotation("doc")[:3] == [1, 2, 3]
            e = rc.encode(lm.activations(toks[:3], (0, 1, 2)), r.spec, r.channels)[1]
            assert torch.equal(torch.stack(r.read(2, [0, 1, 2])), e[2])
            assert rc.is_capture("run1", bucket="acts", prefix="team/exp", s3=opts)
    assert not rc.is_capture("nope", bucket="acts", prefix="team/exp", s3=opts)
    with pytest.raises(ValueError):
        rc.open("missing", bucket="acts", prefix="team/exp", s3=opts)


def test_flush_makes_objects_and_rotation(s3, lm):
    opts, client = s3
    toks = torch.randint(0, V, (4, 5))
    with Sink("run2", toy_layers(lm), "exact", bucket="acts", prefix="p", s3=opts, rotate_bytes=3000) as sink:
        lm(toks[:2])
        sink.flush()                                        # on S3 a flush closes the current object
        with rc.open("s3://acts/p/run2", s3=opts) as r:
            assert r.n_seq == 2 and r.read(1, 0).shape == (5, D)
        lm(toks[2:])
        lm(toks)
    with rc.open("s3://acts/p/run2", s3=opts) as r:
        assert r.n_seq == 8 and r.header["n_files"] >= 3            # flush boundary + rotation
        assert len(keys(client, "acts", "p/run2.0")) == r.header["n_files"]
        assert torch.equal(torch.stack(r.tokens([4, 5, 6, 7])), toks)


def test_transcode_between_buckets_and_local(s3, lm, tmp_path):
    opts, client = s3
    toks = torch.randint(0, V, (5, 6))
    with Sink("src", toy_layers(lm), "exact", bucket="acts", prefix="t", s3=opts) as sink:
        lm(toks)
    loc = {"bucket": "acts", "prefix": "t", "s3": opts}
    rc.transcode("src", "dst", "compact", src_loc=loc, dst_loc={"bucket": "acts2", "prefix": "out", "s3": opts})
    with rc.open("s3://acts2/out/dst", s3=opts) as r:
        assert r.spec.name == "int4_g4+layerK4+out8" and r.n_seq == 5 and r.header["extra"]["transcoded_from"] == "s3://acts/t/src"
    rc.transcode("src", str(tmp_path / "local"), "faithful", src_loc=loc)
    assert rc.open(str(tmp_path / "local")).n_seq == 5


def test_bad_stems():
    from rescomp.store import location
    with pytest.raises(ValueError):
        location("a/b", bucket="x")
    loc = location("s3://bkt/some/prefix/name", s3={"region": "us-east-1", "endpoint_override": "http://127.0.0.1:1"})
    assert loc.remote and loc.path == "bkt/some/prefix/name" and loc.uri == "s3://bkt/some/prefix/name"
    assert loc.rotate_default == 1 << 30 and location("x").rotate_default == 4 << 30
