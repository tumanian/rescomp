"""Codec spec strings.

    <quant>[+layerK<K>][+out<n>][+sink<s>][+key=<quant>]

quant     fp32 | bf16 | fp16 | fp8_e4m3 | int8_tok | int{4,8}_g<G> | mxfp4 | nvfp4   (see quant.py)
layerK    closed-loop layer delta: stored layer i is coded as x_i - xhat_{i-1} unless i % K == 0 (a keyframe)
out       the n highest-energy channels of each stored layer are kept in bf16 (chosen on the first call, or from a sample; see bf16_wrapper)
sink      the first s tokens of every sequence are kept in bf16 (attention-sink / BOS rows)
key       hybrid: a different quantizer for keyframe layers (e.g. key=bf16 keeps keyframes exact)

Examples: "fp16", "bf16", "int4_g384+layerK4+out12", "int4_g512+layerK4+out32+key=bf16".
"""
from dataclasses import dataclass, asdict
from typing import Optional



def check_quant(q):
    from .quant import quantizer
    quantizer(q)                      # raises for unknown names
    return q


@dataclass(frozen=True)
class Spec:
    quant: str = "bf16"
    K: int = 0                       # keyframe interval for layer deltas; 0 = no deltas
    outliers: int = 0
    sink: int = 0
    key_quant: Optional[str] = None  # quantizer for keyframes; None = same as quant

    def __post_init__(self):
        check_quant(self.quant)
        if self.key_quant is not None:
            check_quant(self.key_quant)
        if self.K < 0 or self.outliers < 0 or self.sink < 0:
            raise ValueError("K, outliers and sink must be >= 0")
        if self.K == 1:
            raise ValueError("layerK1 would make every layer a keyframe; use K=0 for no deltas")
        if self.key_quant == self.quant:
            object.__setattr__(self, "key_quant", None)
        if self.K == 0 and self.key_quant is not None:
            raise ValueError("key= needs layer deltas (layerK)")

    @property
    def name(self):
        p = [self.quant]
        if self.K:
            p.append(f"layerK{self.K}")
        if self.outliers:
            p.append(f"out{self.outliers}")
        if self.sink:
            p.append(f"sink{self.sink}")
        if self.key_quant is not None:
            p.append(f"key={self.key_quant}")
        return "+".join(p)

    def __str__(self):
        return self.name

    def dict(self):
        return asdict(self)

    @staticmethod
    def parse(name):
        if isinstance(name, Spec):
            return name
        parts = name.split("+")
        kw = {"quant": parts[0]}
        seen = set()
        for p in parts[1:]:
            kind = p.split("=")[0].rstrip("0123456789.")
            if kind in seen:
                raise ValueError(f"repeated spec part {p!r} in {name!r}")
            seen.add(kind)
            if p.startswith("layerK"):
                kw["K"] = int(p[6:])
            elif p.startswith("out"):
                kw["outliers"] = int(p[3:])
            elif p.startswith("sink"):
                kw["sink"] = int(p[4:])
            elif p.startswith("key="):
                kw["key_quant"] = p[4:]
            else:
                raise ValueError(f"unknown spec part {p!r} in {name!r}")
        return Spec(**kw)

    def is_key(self, i):
        """Is stored layer i (0-based position in the stored list) a keyframe?"""
        return self.K == 0 or i % self.K == 0

    def quant_for(self, i):
        if self.is_key(i) and self.key_quant is not None:
            return self.key_quant
        return self.quant
