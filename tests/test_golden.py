"""Encoded bytes and reconstructions are pinned: a refactor that changes any codec's output fails here.
Regenerate GOLDEN only for a deliberate format change (and bump the container VERSION if files are affected)."""
import hashlib

import pytest
import torch

import rescomp as rc
from rescomp import lossless as LL
from tests.conftest import make_acts

GOLDEN = {
    "fp32": "3f523ccd8ff63a87",
    "bf16": "ac1547841c12011c",
    "fp16": "5afb98eea76b5372",
    "fp8_e4m3": "f7c49abac3b36280",
    "int8_tok": "5318059e5631b454",
    "int8_g64": "5318059e5631b454",          # d = 64, so one group per token: identical to int8_tok by construction
    "int4_g16": "6feddde8084cdad6",
    "int2_g16": "51b7ceb67bfa504c",
    "uabs0.05": "bc01dd26eda27d43",
    "mxfp4": "6fcc54219b5f905d",
    "nvfp4": "2654e3c889b87d3d",
    "int4_g16+layerK2": "c140e9302815d3dc",
    "int4_g16+layerK4+out4": "d4b57f5247e4d766",
    "int4_g16+layerK2+out4+sink1": "bf6e73fd975635d2",   # changed 2026-10-17: calibration ignores sink rows
    "int4_g16+layerK4+out2+key=bf16": "e479a7a6ebe3a847",
    "uabs0.1+out8": "5b3ca28fff654d60",
    "bf16+sink2": "17586e36166629b2",
    "fp16+layerK3+key=fp32": "ae6088d73489f3bb",
    "fp32+out2+sink1": "2b5f01d871dd81c8",            # fp32 codes must stay zero where outliers/sink were split out
}


def golden_acts():
    """bf16-exact values: torch.randn differs in its last bits between CPUs, bf16 rounding hides that."""
    return make_acts(L=4, B=3, T=8, d=64, seed=11).to(torch.bfloat16).float()


def digest(name, acts):
    spec = rc.Spec.parse(name)
    cal = rc.calibrate(acts, spec)
    streams, xh = rc.encode(acts, spec, cal)
    h = hashlib.sha256()
    for s in streams:
        for k in sorted(s):
            h.update(k.encode())
            h.update(LL.to_bytes(s[k])[0].tobytes())
    h.update(LL.to_bytes(xh)[0].tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("name", list(GOLDEN))
def test_golden(name):
    assert digest(name, golden_acts()) == GOLDEN[name]


def test_golden_is_sensitive():
    acts = golden_acts()
    acts[0, 0, 0, 0] += 1.0
    assert digest("fp32", acts) != GOLDEN["fp32"]
