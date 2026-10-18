import numpy as np
import pytest
import torch

from rescomp import lossless as LL


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8,
                                   torch.uint8, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("shuffled", [True, False])
def test_roundtrip(dtype, shuffled):
    g = torch.Generator().manual_seed(0)
    t = torch.randn(3, 5, 16, generator=g)
    t = (t > 0) if dtype == torch.bool else (t * 50).to(dtype)
    blob, raw = LL.compress(t, level=3, shuffled=shuffled)
    assert raw == t.numel() * t.element_size()
    back = LL.decompress(blob, LL.dtype_name(dtype), t.shape, raw, shuffled)
    assert back.dtype == dtype and back.shape == t.shape
    assert torch.equal(back.view(torch.uint8), t.contiguous().view(torch.uint8))


def test_empty():
    t = torch.empty(0, 4, dtype=torch.float16)
    blob, raw = LL.compress(t)
    assert raw == 0
    assert LL.decompress(blob, "float16", (0, 4), 0).shape == (0, 4)


def test_shuffle_helps_on_floats():
    t = torch.linspace(0, 1, 4096).float()
    s, _ = LL.compress(t, shuffled=True)
    r, _ = LL.compress(t, shuffled=False)
    assert len(s) < len(r)


def test_shuffle_inverse():
    b = np.arange(24, dtype=np.uint8)
    for elem in (1, 2, 4, 8):
        assert np.array_equal(LL.unshuffle(LL.shuffle(b, elem), elem), b)


def test_wrong_raw_size_rejected():
    t = torch.arange(10, dtype=torch.int32)
    blob, raw = LL.compress(t)
    with pytest.raises((ValueError, RuntimeError)):
        LL.decompress(blob, "int32", (10,), raw - 4)


def test_noncontiguous_input():
    t = torch.arange(24, dtype=torch.int16).reshape(4, 6).T
    blob, raw = LL.compress(t)
    assert torch.equal(LL.decompress(blob, "int16", t.shape, raw), t)
