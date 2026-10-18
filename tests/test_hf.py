"""The Hugging Face path on a tiny random GPT-2: hooks, capture, evaluation, generation, CLI."""
import json

import pytest
import torch

import rescomp as rc
from rescomp import capture as C
from rescomp import evaluate as E
from rescomp.cli import main


@pytest.fixture(scope="session")
def toks():
    return torch.randint(2, 256, (12, 16), generator=torch.Generator().manual_seed(0))


def test_decoder_blocks_and_hook_names(tiny_model):
    blocks, path = C.decoder_blocks(tiny_model)
    assert len(blocks) == 4 and path == "transformer.h"
    assert C.hook_names(tiny_model, 2) == {"hf": "transformer.h.2", "transformer_lens": "blocks.2.hook_resid_post"}
    info = C.model_info(tiny_model, model_id="x/y")
    assert info["n_layers"] == 4 and info["d_model"] == 64 and info["id"] == "x/y" and info["dtype"] == "float32"
    assert info["config"]["n_head"] == 4
    with pytest.raises(ValueError):
        C.decoder_blocks(torch.nn.Linear(2, 2))


def test_capture_hook_matches_hidden_states(tiny_model, toks):
    logits, acts = C.forward(tiny_model, toks[:2], layers=[0, 2])
    out = tiny_model(toks[:2], output_hidden_states=True)
    # HF hidden_states[l + 1] is block l's output (the last one has the final norm applied, so we avoid it)
    assert torch.equal(acts[0], out.hidden_states[1]) and torch.equal(acts[2], out.hidden_states[3])
    assert torch.equal(logits, out.logits)


def test_patch_hook_replaces_output(tiny_model, toks):
    _, acts = C.forward(tiny_model, toks[:2], layers=[1])
    l1, _ = C.forward(tiny_model, toks[:2], patch={1: torch.zeros_like(acts[1])})
    l2, _ = C.forward(tiny_model, toks[:2], patch={1: lambda t: t * 0})
    l0, _ = C.forward(tiny_model, toks[:2])
    assert torch.equal(l1, l2) and not torch.equal(l0, l1)


def test_identity_patch_gives_zero_kl(tiny_model, toks):
    k = E.identity_check(tiny_model, toks, [1, 2, 3], batch=5)
    assert k["kl_mean"] == 0.0 and k["top1_agree"] == 1.0 and k["ppl_ratio"] == 1.0 and k["n_positions"] == 12 * 16


def test_capture_file_and_evaluate(tiny_model, toks, tmp_path, fake_tok):
    p = tmp_path / "t.rsc"
    h = C.capture(tiny_model, fake_tok, toks, [1, 2, 3], "int4_g16+layerK3+out2", p, batch=5, calib_seqs=4,
                  model_id="tiny", annotations={"doc": torch.arange(12)}, log=None)
    assert h["capture"]["n_seq"] == 12 and h["capture"]["n_chunks"] == 3 and h["calib"]["source"] == "first 4 sequences"
    with rc.Reader(p) as r:
        assert r.layers == [1, 2, 3] and r.header["model"]["id"] == "tiny"
        assert r.header["layers"][0]["hook"]["hf"] == "transformer.h.1"
        assert torch.equal(r.tokens(), toks) and r.annotation("doc").tolist() == list(range(12))
        _, acts = C.forward(tiny_model, toks[:5], layers=[1, 2, 3])
        X = torch.stack([acts[l] for l in (1, 2, 3)])
        assert torch.equal(r.read_stack(seqs=slice(0, 5)), rc.encode(X, r.spec, r.calib)[1])
        spec, calib = r.spec, r.calib
    f = E.evaluate_file(p, tiny_model, batch=4)
    assert f["mode"] == "file" and f["n_positions"] == 12 * 16 and 0 < f["kl_mean"] < 1 and 0 < f["top1_agree"] <= 1
    assert f["ratio_vs_bf16"] > 1
    lv = E.evaluate_live(tiny_model, toks, [1, 2, 3], spec, calib, batch=4)
    assert lv["mode"] == "live" and 0 < lv["kl_mean"] < 1
    per = E.evaluate_file_per_layer(p, tiny_model, batch=4, layers=[1, 3])
    assert list(per) == [1, 3] and all(0 < v["kl_mean"] < 1 for v in per.values())
    # lossless storage of an fp32 model gives KL 0 both ways
    C.capture(tiny_model, fake_tok, toks, [1, 2, 3], "fp32", tmp_path / "f.rsc", batch=4, log=None)
    assert E.evaluate_file(tmp_path / "f.rsc", tiny_model, batch=4)["kl_mean"] == 0.0
    assert E.evaluate_live(tiny_model, toks, [1, 2, 3], "fp32", batch=4)["kl_mean"] == 0.0


def test_capture_calibrates_on_exactly_calib_seqs(tiny_model, toks, tmp_path, fake_tok, monkeypatch):
    seen = []
    real = C.forward
    monkeypatch.setattr(C, "forward", lambda model, tb, *a, **k: (seen.append(len(tb)), real(model, tb, *a, **k))[1])
    C.capture(tiny_model, fake_tok, toks, [1, 2], "int4_g16+out2", tmp_path / "c.rsc", batch=5, calib_seqs=4, log=None)
    assert seen[0] == 4 and sum(seen) == 4 + 12


def test_more_damage_more_kl(tiny_model, toks):
    kl = {s: E.evaluate_live(tiny_model, toks, [2, 3], s, batch=6)["kl_mean"] for s in ("fp16", "int8_tok", "int2_g16")}
    assert kl["fp16"] < kl["int8_tok"] < kl["int2_g16"]


def test_coherence(tiny_model, fake_tok):
    out = E.coherence(tiny_model, ["hello world", [5, 6, 7]], [2, 3], "int4_g16+layerK2", tokenizer=fake_tok,
                      n_new=6, seeds=2)
    s = out["summary"]
    assert s["n_pairs"] == 4 and len(out["records"]) == 4
    r = out["records"][0]
    assert len(r["clean"]["tokens"]) == 6 and len(r["codec"]["tokens"]) == 6 and isinstance(r["codec"]["text"], str)
    assert r["prompt"] == "hello world" and out["records"][2]["prompt"] is None
    assert abs(s["ppl_ratio"] - __import__("math").exp(s["log_ppl_diff"])) < 1e-9
    # with no compression the paired difference is exactly zero
    z = E.coherence(tiny_model, [[5, 6, 7]], [2, 3], "fp32", n_new=4, seeds=2)["summary"]
    assert z["log_ppl_diff"] == 0.0 and z["t"] is None and z["n_prompts"] == 1


def test_generate_is_seeded(tiny_model):
    ids = torch.tensor([[3, 4, 5]])
    a = E.generate(tiny_model, ids, 5, seed=1)
    b = E.generate(tiny_model, ids, 5, seed=1)
    c = E.generate(tiny_model, ids, 5, seed=2)
    assert torch.equal(a, b) and a.shape == (1, 8) and not torch.equal(a, c)


def test_ngram_stats():
    s = E.ngram_stats([1, 2, 3, 4, 1, 2, 3, 4])
    assert s["rep4"] == 1 / 5 and s["distinct2"] == 4 / 7
    assert E.ngram_stats([]) == {"distinct2": 0.0, "rep4": 0.0}


def test_pack_sequences(fake_tok):
    t = C.pack_sequences(fake_tok, ["abcdef", "ghijkl", "mnop"], n_seq=2, seq_len=4)
    assert t.shape == (2, 4) and t[0].tolist() == [ord(c) for c in "abcd"]
    b = C.pack_sequences(fake_tok, ["abcdef", "ghijkl"], n_seq=2, seq_len=4, bos=True)
    assert b[:, 0].tolist() == [0, 0] and b[0, 1:].tolist() == [ord(c) for c in "abc"]
    with pytest.raises(ValueError):
        C.pack_sequences(fake_tok, ["ab"], n_seq=2, seq_len=4)


def test_cli_info(tiny_model, toks, tmp_path, fake_tok, capsys):
    p = tmp_path / "t.rsc"
    C.capture(tiny_model, fake_tok, toks, [1, 3], "bf16", p, batch=6, log=None)
    main(["info", str(p)])
    out = capsys.readouterr().out
    assert "RSC1" in out and "layers [1, 3]" in out
    main(["info", str(p), "--json"])
    out = capsys.readouterr().out
    assert '"spec": "bf16"' in out and "chunks" not in json.loads(out[out.index("{"):])


def test_moe_routing_capture(tiny_moe, tiny_model, fake_tok, tmp_path):
    assert C.moe_layers(tiny_moe) == [0, 1] and C.moe_layers(tiny_model) == []
    toks = torch.randint(2, 64, (6, 8), generator=torch.Generator().manual_seed(1))
    logits, acts, routing = C.forward(tiny_moe, toks[:2], layers=[1], routers=[0, 1])
    assert set(routing) == {0, 1} and routing[1]["experts"].shape == (2, 8, 2) and routing[1]["experts"].dtype == torch.int16
    assert routing[1]["experts"].min() >= 0 and routing[1]["experts"].max() < 4
    assert (routing[1]["experts"][..., 0] != routing[1]["experts"][..., 1]).all()      # top-2 are distinct experts
    assert (routing[1]["weights"].float() > 0).all() and (routing[1]["weights"].float().sum(-1) <= 1.001).all()
    h = C.capture(tiny_moe, fake_tok, toks, [1], "bf16", tmp_path / "moe.rsc", batch=4, routers="stored", log=None)
    assert h["capture"]["extra"]["routers"] == {"layers": [1], "top_k": 2, "num_experts": 4}
    assert h["model"]["config"]["moe_layers"] == [0, 1] and h["model"]["config"]["num_experts"] == 4
    with rc.Reader(tmp_path / "moe.rsc") as r:
        assert r.annotations == ["experts.1", "expert_weights.1"]
        assert torch.equal(r.annotation("experts.1")[:2], routing[1]["experts"])
        assert r.annotation("expert_weights.1").shape == (6, 8, 2)
    with pytest.raises(ValueError):
        C.capture(tiny_moe, fake_tok, toks, [1], "bf16", tmp_path / "x.rsc", routers=[5], log=None)
    assert E.identity_check(tiny_moe, toks, [0, 1], batch=3)["kl_mean"] == 0.0


def test_capture_calib_seqs_zero_uses_first_chunk(tiny_model, toks, tmp_path, fake_tok):
    h = C.capture(tiny_model, fake_tok, toks, [1, 2], "int4_g16+out2", tmp_path / "c.rsc", batch=5, calib_seqs=0, log=None)
    assert h["calib"]["source"] == "first_chunk" and h["capture"]["extra"]["calib_seqs"] == 0


def test_cli_eval(tiny_model, toks, tmp_path, fake_tok, monkeypatch, capsys):
    p = tmp_path / "t.rsc"
    C.capture(tiny_model, fake_tok, toks, [1, 3], "int4_g16+layerK2", p, batch=6, model_id="tiny", log=None)
    monkeypatch.setattr("rescomp.capture.load_model", lambda *a, **k: (tiny_model, fake_tok))
    main(["eval", str(p), "--batch", "4", "--live", "--coherence", "--n-prompts", "1", "--n-new", "4", "--seeds", "2",
          "--out", str(tmp_path / "e.json")])
    out = json.loads((tmp_path / "e.json").read_text())
    assert out["model"] == "tiny" and out["file_eval"]["layers"] == [3] and out["live_eval"]["layers"] == [1, 3]
    assert out["coherence"]["n_pairs"] == 2


def test_load_model_from_disk(tiny_model, tmp_path):
    from transformers import GPT2TokenizerFast
    tiny_model.save_pretrained(tmp_path / "m")
    # a minimal tokenizer so AutoTokenizer can load from the same folder
    vocab = {chr(i): i for i in range(256)}
    (tmp_path / "m" / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "m" / "merges.txt").write_text("#version: 0.2\n")
    GPT2TokenizerFast(str(tmp_path / "m" / "vocab.json"), str(tmp_path / "m" / "merges.txt")).save_pretrained(tmp_path / "m")
    model, tok = C.load_model(str(tmp_path / "m"), dtype="float32", device="cpu")
    assert isinstance(model, torch.nn.Module) and hasattr(tok, "encode") and not model.training
    assert C.model_info(model, tok)["n_layers"] == 4
