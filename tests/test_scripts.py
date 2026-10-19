"""scripts/: the CLI (info / eval on a capture) and the token-packing helper. conftest puts scripts/ on sys.path."""
import json

import pytest
import torch

import rescomp as rc
from cli import main
from corpus import pack_sequences


def test_cli_info_and_eval(tiny_model, fake_tok, tmp_path, monkeypatch, capsys):
    toks = torch.randint(2, 256, (8, 10), generator=torch.Generator().manual_seed(0))
    stem = str(tmp_path / "c")
    with rc.Sink(stem, rc.bind(tiny_model, [1, 3], tokenizer=fake_tok, model_id="tiny"), "int4_g16+layerK2") as sink:
        tiny_model(toks[:4])
        tiny_model(toks[4:])
    main(["info", stem])
    out = capsys.readouterr().out
    assert "RSC3" in out and "layers [1, 3]" in out
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
