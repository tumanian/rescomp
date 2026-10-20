"""scripts/: the CLI (info / eval on a capture) and the token-packing helper. conftest puts scripts/ on sys.path."""
import json

import pytest
import torch
from cli import main
from corpus import pack_sequences

import rescomp as rc


def test_cli_info_and_eval(tiny_model, fake_tok, tmp_path, monkeypatch, capsys):
    toks = torch.randint(2, 256, (8, 10), generator=torch.Generator().manual_seed(0))
    stem = str(tmp_path / "c")
    with rc.Sink(stem, rc.bind(tiny_model, [1, 3], tokenizer=fake_tok, model_id="tiny"), "int4_g16+layerK2"):
        tiny_model(toks[:4])
        tiny_model(toks[4:])
    main(["info", stem])
    out = capsys.readouterr().out
    assert "RSC4" in out and "layers [1, 3]" in out
    monkeypatch.setattr("rescomp.hf.load_model", lambda *a, **k: (tiny_model, fake_tok))
    main(["eval", stem, "--batch", "4", "--live", "--coherence", "--n-prompts", "1", "--n-new", "4", "--seeds", "2",
          "--out", str(tmp_path / "e.json")])
    res = json.loads((tmp_path / "e.json").read_text())
    assert res["model"] == "tiny" and 0 <= res["last_layer"]["kl_mean"] < 1 and res["live"]["n_seq"] == 8
    assert res["coherence"]["n_pairs"] == 2



def test_pack_sequences(fake_tok):
    t = pack_sequences(fake_tok, ["abcdef", "ghijkl", "mnop"], n_seq=2, seq_len=4)
    assert t.shape == (2, 4) and t[0].tolist() == [ord(c) for c in "abcd"]
    b = pack_sequences(fake_tok, ["abcdef", "ghijkl"], n_seq=2, seq_len=4, bos=True)
    assert b[:, 0].tolist() == [0, 0] and b[0, 1:].tolist() == [ord(c) for c in "abc"]
    with pytest.raises(ValueError):
        pack_sequences(fake_tok, ["ab"], n_seq=2, seq_len=4)


def test_probes_helpers(tmp_path):
    """scripts/probes.py: the logistic-regression probe separates what is separable; features() reads a capture
    by its split annotation."""
    from probes import features, predict, train_probe
    from tests.test_generic import V, ToyLM
    from tests.test_sink import layers_of
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 8, generator=g)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).long()
    probe = train_probe(X[:150], y[:150], 2)
    assert (predict(probe, X[150:]) == y[150:]).float().mean() > 0.95
    lm = ToyLM().eval()
    toks = torch.randint(0, V, (6, 5), generator=g)
    with rc.TransformSink(str(tmp_path / "p"), layers_of(lm)) as sink:
        sink.annotate(label=[0, 1, 0, 1, 0, 1], split=["train"] * 4 + ["test"] * 2)
        lm(toks)
    feats, info = features(str(tmp_path / "p"), [1, 2], ["mean", "last"])
    Xtr, ytr, Xte, yte = feats[(2, "last")]
    assert Xtr.shape == (4, 32) and ytr.tolist() == [0, 1, 0, 1] and Xte.shape == (2, 32) and yte.tolist() == [0, 1]
    assert info["spec"] == "fp32" and set(feats) == {(1, "mean"), (1, "last"), (2, "mean"), (2, "last")}
