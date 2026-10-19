"""The whole pipeline on a hand-written torch model, with Hugging Face blocked from importing: nothing outside
rescomp.hf may depend on transformers."""
import builtins
import math
import sys

import pytest
import torch

import rescomp as rc
from rescomp import instrument as I
from rescomp.hooks import Layers
from rescomp.sink import Sink

D, L, V = 32, 3, 50


class ResBlock(torch.nn.Module):
    """x -> x + mlp(x): the block's output is the residual stream after it, like a decoder block."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 2 * D), torch.nn.GELU(), torch.nn.Linear(2 * D, D))

    def forward(self, x):
        return x + self.mlp(x)


class ToyLM(torch.nn.Module):
    """Embedding, L residual MLP blocks, unembedding. forward(input_ids, attention_mask=None) like a real LM."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, D)
        self.blocks = torch.nn.ModuleList([ResBlock() for _ in range(L)])
        self.out = torch.nn.Linear(D, V)

    def forward(self, input_ids, attention_mask=None, **kw):
        x = self.emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)

    def activations(self, input_ids, idx):
        """Reference: the residual stream after each block in idx, [len(idx), B, T, d]."""
        x, acts = self.emb(input_ids), {}
        for l, blk in enumerate(self.blocks):
            x = blk(x)
            acts[l] = x
        return torch.stack([acts[l] for l in idx]).detach()


@pytest.fixture(scope="module")
def toy():
    return ToyLM().eval()


def toy_layers(lm, idx=(0, 1, 2)):
    return Layers(modules={l: lm.blocks[l] for l in idx}, root=lm, hook_names={l: {"toy": f"blocks.{l}"} for l in idx},
                  model_info={"id": "toy-lm", "framework": "plain torch", "d_model": D}, n_layers=L)


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


def test_pipeline_without_transformers(toy, tmp_path, no_transformers):
    toks = torch.randint(0, V, (10, 12), generator=torch.Generator().manual_seed(2))
    stem = str(tmp_path / "toy")
    with Sink(stem, toy_layers(toy), "int4_g16+layerK2+out2", fidelity=True) as sink:
        sink.annotate(doc=list(range(4)))
        toy(toks[:4])
        toy(toks[4:])
    with rc.open(stem) as r:
        assert r.header["model"]["id"] == "toy-lm" and r.header["layers"][1]["hook"] == {"toy": "blocks.1"}
        assert r.n_seq == 10 and r.annotation("doc")[:4] == [0, 1, 2, 3] and r.header["fidelity"]["rel_l2"] > 0
        X = toy.activations(toks[:4], (0, 1, 2))
        assert torch.equal(torch.stack(r.read(2, [0, 1, 2, 3])), rc.encode(X, r.spec, r.channels)[1][2])
        spec, channels = r.spec, r.channels
    run = lambda t: toy(t)
    assert I.kl_ppl(run, toks, I.Patcher(toy_layers(toy), lambda l, x, s, st: x), batch=4)["kl_mean"] == 0.0
    with rc.open(stem) as r:
        f = I.kl_ppl(run, toks, I.Patcher(toy_layers(toy), r, only=[2]), seq_ids=list(range(10)), batch=4)
    assert 0 < f["kl_mean"] < 1
    lv = I.kl_ppl(run, toks, I.Patcher(toy_layers(toy), I.LiveCodec(spec, channels)), batch=4)
    assert 0 < lv["kl_mean"] < 1
    per = I.per_layer(run, toks, toy_layers(toy), I.LiveCodec(spec, channels), batch=4)
    assert list(per) == [0, 1, 2]
    co = I.coherence(run, [[1, 2, 3], [4, 5]], I.Patcher(toy_layers(toy), I.LiveCodec(spec, channels)), n_new=5, seeds=2)["summary"]
    assert co["n_pairs"] == 4 and abs(co["ppl_ratio"] - math.exp(co["log_ppl_diff"])) < 1e-9
    assert "transformers" not in sys.modules
