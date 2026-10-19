"""Lossless stages (registered Stages): the bytes of a code or scale tensor -> compressed bytes, built by `lossless(record)`.

One implementation today, ZstdShuffle: byte-shuffle (group the k-th byte of every element together, which won
122 of 124 times against bit-shuffle and raw in the earlier sweep) then zstd. The header records the backend
as {"codec": name, ...options}, so another backend is a subclass plus a header value.
"""
from abc import abstractmethod

import numpy as np
import torch
import zstandard

from .stage import Stage, register, stage

DTYPES = {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float8_e4m3fn": torch.float8_e4m3fn, "float8_e8m0fnu": torch.float8_e8m0fnu, "int8": torch.int8,
          "uint8": torch.uint8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64, "bool": torch.bool}


def check_dtype(t, what):
    if dtype_name(t.dtype) not in DTYPES:
        raise ValueError(f"{what}: dtype {dtype_name(t.dtype)} not storable; use one of {sorted(DTYPES)}")


def dtype_name(dt):
    return str(dt).removeprefix("torch.")


def to_bytes(t):
    """Any tensor -> (uint8 numpy of its raw bytes, element size)."""
    t = t.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy().ravel(), t.element_size()


def shuffle(b, elem):
    return b if elem == 1 or len(b) == 0 else b.reshape(-1, elem).T.copy().ravel()


def unshuffle(b, elem):
    return b if elem == 1 or len(b) == 0 else b.reshape(elem, -1).T.copy().ravel()


class Lossless(Stage):
    kind = "lossless"
    name: str                      # the one name a backend answers to

    @classmethod
    def matches(cls, name):
        return name == cls.name

    @abstractmethod
    def compress(self, t):
        """tensor -> (compressed bytes, raw byte count)."""

    @abstractmethod
    def decompress(self, blob, dtype, shape, raw_nbytes):
        """compressed bytes -> tensor of dtype/shape (CPU)."""



@register
class ZstdShuffle(Lossless):
    name = "zstd"

    def __init__(self, name="zstd", level=3, shuffle=True):
        super().__init__(name)
        self.level, self.shuffle = int(level), bool(shuffle)

    def to_dict(self):
        return {"codec": self.name, "level": self.level, "shuffle": self.shuffle}

    def compress(self, t):
        b, elem = to_bytes(t)
        if self.shuffle:
            b = shuffle(b, elem)
        return zstandard.ZstdCompressor(level=self.level).compress(b.tobytes()), len(b)

    def decompress(self, blob, dtype, shape, raw_nbytes):
        if isinstance(dtype, str):
            if dtype not in DTYPES:
                raise ValueError(f"unknown stored dtype {dtype!r}")
            dtype = DTYPES[dtype]
        elem = torch.empty(0, dtype=dtype).element_size()
        b = np.frombuffer(zstandard.ZstdDecompressor().decompress(blob, max_output_size=raw_nbytes), np.uint8)
        if len(b) != raw_nbytes:
            raise ValueError(f"decompressed {len(b)} bytes, expected {raw_nbytes}")
        if self.shuffle:
            b = unshuffle(b, elem)
        if raw_nbytes == 0:
            return torch.empty(list(shape), dtype=dtype)
        if not b.flags.writeable:                                # zstd's buffer is read-only; unshuffle already copied
            b = b.copy()
        return torch.from_numpy(b).view(dtype).reshape(list(shape))


def lossless(record=None, **options):
    """Factory: from a header record {"codec": "zstd", "level": 3, "shuffle": True}, a backend name, or keyword
    options; a Lossless instance passes through."""
    if isinstance(record, Lossless):
        return record
    rec = {"codec": record} if isinstance(record, str) else dict(record or {})
    rec.update(options)
    return stage("lossless", rec.pop("codec", "zstd"), **rec)
