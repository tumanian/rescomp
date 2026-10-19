import numpy as np
import pytest
import torch

from rescomp import lossless as LL
from rescomp.lossless import ZstdShuffle, lossless


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8,
                                   torch.uint8, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("shuffled", [True, False])
def test_roundtrip(dtype, shuffled):
    g = torch.Generator().manual_seed(0)
    t = torch.randn(3, 5, 16, generator=g)
    t = (t > 0) if dtype == torch.bool else (t * 50).to(dtype)
    codec = ZstdShuffle(level=3, shuffle=shuffled)
    blob, raw = codec.compress(t)
    assert raw == t.numel() * t.element_size()
    back = codec.decompress(blob, LL.dtype_name(dtype), t.shape, raw)
    assert back.dtype == dtype and back.shape == t.shape
    assert torch.equal(back.view(torch.uint8), t.contiguous().view(torch.uint8))


def test_factory_and_record():
    c = lossless({"codec": "zstd", "level": 7, "shuffle": False})
    assert isinstance(c, ZstdShuffle) and c.level == 7 and not c.shuffle
    assert c.to_dict() == {"codec": "zstd", "level": 7, "shuffle": False}
    assert lossless().to_dict() == {"codec": "zstd", "level": 3, "shuffle": True}
    assert lossless(level=19).level == 19
    with pytest.raises(ValueError):
        lossless({"codec": "nope"})


def test_empty():
    t = torch.empty(0, 4, dtype=torch.float16)
    blob, raw = lossless().compress(t)
    assert raw == 0
    assert lossless().decompress(blob, "float16", (0, 4), 0).shape == (0, 4)


def test_shuffle_helps_on_floats():
    t = torch.linspace(0, 1, 4096).float()
    s, _ = ZstdShuffle(shuffle=True).compress(t)
    r, _ = ZstdShuffle(shuffle=False).compress(t)
    assert len(s) < len(r)


def test_shuffle_inverse():
    b = np.arange(24, dtype=np.uint8)
    for elem in (1, 2, 4, 8):
        assert np.array_equal(LL.unshuffle(LL.shuffle(b, elem), elem), b)


def test_wrong_raw_size_rejected():
    t = torch.arange(10, dtype=torch.int32)
    blob, raw = lossless().compress(t)
    with pytest.raises((ValueError, RuntimeError)):
        lossless().decompress(blob, "int32", (10,), raw - 4)


def test_unknown_dtype_rejected():
    blob, raw = lossless().compress(torch.zeros(4))
    with pytest.raises(ValueError, match="unknown stored dtype"):
        lossless().decompress(blob, "complex64", (4,), raw)


def test_noncontiguous_input():
    t = torch.arange(24, dtype=torch.int16).reshape(4, 6).T
    blob, raw = lossless().compress(t)
    assert torch.equal(lossless().decompress(blob, "int16", t.shape, raw), t)
