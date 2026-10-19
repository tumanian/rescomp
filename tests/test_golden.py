"""Encoded bytes and reconstructions are pinned: a refactor that changes any codec's output fails here.
Regenerate GOLDEN only for a deliberate format change (and bump quant.CODEC_VERSION).
Codec 2 (2026-10-18): int groups, mxfp4 and nvfp4 moved to torchao; those hashes changed, the rest did not."""
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
    "int8_g64": "c32d8916e4a6b450",          # d = 64, so one group per token: identical to int8_tok by construction
    "int4_g16": "75c70f3dcf8a568e",
    "uabs0.05": "bc01dd26eda27d43",
    "mxfp4": "4eaf361f495d21ec",
    "nvfp4": "1b1fcd5b7c7c2e68",
    "int4_g16+layerK2": "bef1b603af6728a1",
    "int4_g16+layerK4+out4": "67d3f6916793ce14",
    "int4_g16+layerK2+out4+sink1": "d35af5d6ebb89afa",   # changed 2026-10-17: calibration ignores sink rows
    "int4_g16+layerK4+out2+key=bf16": "ad5c7d7be787e07d",
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
    cal = rc.choose_channels(acts, spec)
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
