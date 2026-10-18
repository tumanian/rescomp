"""Lossless stage: byte-shuffle (group the k-th byte of every element together) + zstd.

Byte shuffle won 122 of 124 times against bit-shuffle and raw in the earlier sweep, and zstd-3 is within a few
percent of zstd-19 at a fraction of the CPU time; both are options here.
"""
import numpy as np
import torch
import zstandard

DTYPES = {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float8_e4m3fn": torch.float8_e4m3fn, "int8": torch.int8, "uint8": torch.uint8, "int16": torch.int16,
          "int32": torch.int32, "int64": torch.int64, "bool": torch.bool}


def check_dtype(t, what):
    if dtype_name(t.dtype) not in DTYPES:
        raise ValueError(f"{what}: dtype {dtype_name(t.dtype)} not storable; use one of {sorted(DTYPES)}")


def dtype_name(dt):
    return str(dt).removeprefix("torch.")


def shuffle(b, elem):
    return b if elem == 1 or len(b) == 0 else b.reshape(-1, elem).T.copy().ravel()


def unshuffle(b, elem):
    return b if elem == 1 or len(b) == 0 else b.reshape(elem, -1).T.copy().ravel()


def to_bytes(t):
    """Any tensor -> (uint8 numpy of its raw bytes, elem size)."""
    t = t.detach().contiguous().cpu()
    elem = t.element_size()
    return t.view(torch.uint8).numpy().ravel(), elem


def compress(t, level=3, shuffled=True):
    """tensor -> (compressed bytes, raw byte count)."""
    b, elem = to_bytes(t)
    if shuffled:
        b = shuffle(b, elem)
    return zstandard.ZstdCompressor(level=level).compress(b.tobytes()), len(b)


def decompress(blob, dtype, shape, raw_nbytes, shuffled=True):
    """compressed bytes -> tensor of dtype/shape (CPU)."""
    if isinstance(dtype, str):
        if dtype not in DTYPES:
            raise ValueError(f"unknown stored dtype {dtype!r}")
        dtype = DTYPES[dtype]
    dt = dtype
    elem = torch.empty(0, dtype=dt).element_size()
    b = np.frombuffer(zstandard.ZstdDecompressor().decompress(blob, max_output_size=raw_nbytes), np.uint8)
    if len(b) != raw_nbytes:
        raise ValueError(f"decompressed {len(b)} bytes, expected {raw_nbytes}")
    if shuffled:
        b = unshuffle(b, elem)
    if raw_nbytes == 0:
        return torch.empty(list(shape), dtype=dt)
    if not b.flags.writeable:                                    # zstd's buffer is read-only; unshuffle already copied
        b = b.copy()
    return torch.from_numpy(b).view(dt).reshape(list(shape))
