"""The whole pipeline on a hand-written torch model through FunctionAdapter, with Hugging Face blocked from
importing: nothing outside rescomp.hf may depend on transformers."""
import builtins
import math
import sys

import pytest
import torch

import rescomp as rc
from rescomp import capture as C
from rescomp import evaluate as E
from rescomp.model import FunctionAdapter, ModelAdapter, apply_patch

D, L, V = 32, 3, 50


class ResBlock(torch.nn.Module):
    """x -> x + mlp(x): the block's output is the residual stream after it, like a decoder block."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 2 * D), torch.nn.GELU(), torch.nn.Linear(2 * D, D))

    def forward(self, x):
        return x + self.mlp(x)


class ToyLM(torch.nn.Module):
    """Embedding, L residual MLP blocks, unembedding. Not a Hugging Face model."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, D)
        self.blocks = torch.nn.ModuleList([ResBlock() for _ in range(L)])
        self.out = torch.nn.Linear(D, V)

    def run(self, tokens, capture, patch):
        x = self.emb(tokens)
        acts = {}
        for l, blk in enumerate(self.blocks):
            x = blk(x)
            x = apply_patch(l, x, patch)
            if l in capture:
                acts[l] = x
        return self.out(x), acts


@pytest.fixture(scope="module")
def toy():
    lm = ToyLM().eval()
    return FunctionAdapter(lm.run, n_layers=L, d_model=D, info={"id": "toy-lm", "framework": "plain torch"},
                           hook_names=lambda l: {"toy": f"blocks.{l}"})


@pytest.fixture
def no_transformers(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("transformers is blocked in this test")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    for m in [m for m in sys.modules if m == "transformers" or m.startswith("transformers.")]:
        monkeypatch.delitem(sys.modules, m)


def test_protocol(toy):
    assert isinstance(toy, ModelAdapter)
    assert toy.n_layers == L and toy.d_model == D and toy.moe_layers() == [] and toy.device.type == "cpu"
    assert toy.info()["id"] == "toy-lm" and toy.hook_names(1) == {"toy": "blocks.1"}
    with pytest.raises(ValueError):
        toy.forward(torch.zeros(1, 4, dtype=torch.long), routers=[0])


def test_pipeline_without_transformers(toy, tmp_path, no_transformers):
    toks = torch.randint(0, V, (10, 12), generator=torch.Generator().manual_seed(2))
    logits, acts, routing = toy.forward(toks[:3], capture=[0, 2])
    assert logits.shape == (3, 12, V) and set(acts) == {0, 2} and routing == {}
    assert E.identity_check(toy, toks, [0, 1, 2], batch=4)["kl_mean"] == 0.0

    h = C.capture(toy, toks, [0, 1, 2], "int4_g16+layerK2+out2", tmp_path / "toy.rsc", batch=4, calib_seqs=4,
                  annotations={"doc": torch.arange(10)}, log=None)
    assert h["model"] == {"n_layers": L, "d_model": D, "id": "toy-lm", "framework": "plain torch"}
    assert h["layers"][1]["hook"] == {"toy": "blocks.1"} and h["capture"]["n_seq"] == 10
    with rc.Reader(tmp_path / "toy.rsc") as r:
        X = torch.stack([toy.forward(toks[:4], capture=[0, 1, 2])[1][l] for l in (0, 1, 2)])
        assert torch.equal(r.read_stack(seqs=slice(0, 4)), rc.encode(X, r.spec, r.calib)[1])
        spec, calib = r.spec, r.calib

    f = E.evaluate_file(tmp_path / "toy.rsc", toy, batch=4)
    assert f["layers"] == [2] and 0 < f["kl_mean"] < 1
    per = E.evaluate_file_per_layer(tmp_path / "toy.rsc", toy, batch=4)
    assert list(per) == [0, 1, 2]
    lv = E.evaluate_live(toy, toks, [0, 1, 2], spec, calib, batch=4)
    assert 0 < lv["kl_mean"] < 1
    co = E.coherence(toy, [[1, 2, 3], [4, 5]], [0, 1, 2], spec, calib, n_new=5, seeds=2)["summary"]
    assert co["n_pairs"] == 4 and abs(co["ppl_ratio"] - math.exp(co["log_ppl_diff"])) < 1e-9
    assert "transformers" not in sys.modules


def test_capture_rejects_bad_layers(toy, tmp_path):
    with pytest.raises(ValueError):
        C.capture(toy, torch.zeros(2, 4, dtype=torch.long), [0, 7], "bf16", tmp_path / "x.rsc", log=None)


def test_tl_adapter_if_installed():
    tl = pytest.importorskip("transformer_lens")
    from rescomp.model import TLAdapter
    cfg = tl.HookedTransformerConfig(n_layers=2, d_model=16, d_head=8, n_heads=2, d_mlp=32, d_vocab=40, n_ctx=16,
                                     act_fn="gelu")
    m = TLAdapter(tl.HookedTransformer(cfg).eval(), model_id="tl-toy")
    toks = torch.randint(0, 40, (2, 8))
    logits, acts, _ = m.forward(toks, capture=[0, 1])
    assert logits.shape == (2, 8, 40) and acts[1].shape == (2, 8, 16)
    assert E.identity_check(m, toks, [0, 1], batch=2)["kl_mean"] == 0.0
    assert m.info()["n_layers"] == 2 and m.hook_names(0) == {"transformer_lens": "blocks.0.hook_resid_post"}
