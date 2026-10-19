"""Hugging Face specifics: block discovery, header info, the hooked site equals hidden_states, MoE info."""
import json

import pytest
import torch

import rescomp as rc
from rescomp import hf as H
from rescomp import instrument as I


def test_decoder_blocks_and_info(tiny_model, fake_tok):
    blocks, path = H.decoder_blocks(tiny_model)
    assert len(blocks) == 4 and path == "transformer.h"
    info = H.HuggingFaceBinding(tiny_model).info(fake_tok, model_id="x/y")
    assert info["n_layers"] == 4 and info["d_model"] == 64 and info["id"] == "x/y" and info["dtype"] == "float32"
    assert info["config"]["n_head"] == 4 and info["loader"] == "transformers" and info["tokenizer"] == "fake"
    with pytest.raises(ValueError):
        H.decoder_blocks(torch.nn.Linear(2, 2))
    L = rc.bind(tiny_model, [1, 3], tokenizer=fake_tok)
    assert L.hook_names[3] == {"hf": "transformer.h.3", "transformer_lens": "blocks.3.hook_resid_post"}
    assert L.root is tiny_model and L.n_layers == 4 and L.model_info["loader"] == "transformers"
    with pytest.raises(ValueError):
        rc.bind(tiny_model, [9])
    with pytest.raises(ValueError):
        rc.bind(tiny_model)
    with pytest.raises(ValueError):
        rc.bind(torch.nn.Linear(2, 2), [0])


def test_hooked_site_is_hidden_states(tiny_model, fake_tok, tmp_path):
    toks = torch.randint(2, 256, (3, 10))
    with rc.Sink(str(tmp_path / "s"), rc.bind(tiny_model, [0, 2], tokenizer=fake_tok), "fp32"):
        tiny_model(toks)
    out = tiny_model(toks, output_hidden_states=True)
    with rc.open(str(tmp_path / "s")) as r:
        # HF hidden_states[l + 1] is block l's output (the last one has the final norm applied, so we avoid it)
        assert torch.equal(torch.stack(r.read(0)), out.hidden_states[1])
        assert torch.equal(torch.stack(r.read(2)), out.hidden_states[3])


def test_moe_info(tiny_moe, fake_tok, tmp_path):
    assert H.HuggingFaceBinding(tiny_moe).moe_layers() == [0, 1]
    toks = torch.randint(2, 64, (3, 6))
    with rc.Sink(str(tmp_path / "moe"), rc.bind(tiny_moe, [1], tokenizer=fake_tok), "fp16"):
        tiny_moe(toks)
    with rc.open(str(tmp_path / "moe")) as r:
        cfg = r.header["model"]["config"]
        assert cfg["num_experts"] == 4 and cfg["num_experts_per_tok"] == 2 and cfg["moe_layers"] == [0, 1]
        assert r.read(1, 0).shape == (6, 32)


def test_hf_patcher_measurements(tiny_model, fake_tok):
    L = rc.bind(tiny_model, [1, 2, 3], tokenizer=fake_tok)
    run = lambda t: tiny_model(t, use_cache=False).logits
    t = torch.randint(2, 256, (6, 9), generator=torch.Generator().manual_seed(1))
    assert I.kl_ppl(run, t, I.Patcher(L, lambda l, x, s, st: x), batch=3)["kl_mean"] == 0.0
    kl = {s: I.kl_ppl(run, t, I.Patcher(L, I.LiveCodec(s)), batch=3)["kl_mean"] for s in ("fp16", "int8_tok", "int4_g16+layerK3")}
    assert kl["fp16"] < kl["int8_tok"] < kl["int4_g16+layerK3"]


def test_load_model_from_disk(tiny_model, tmp_path):
    from transformers import GPT2TokenizerFast
    tiny_model.save_pretrained(tmp_path / "m")
    vocab = {chr(i): i for i in range(256)}
    (tmp_path / "m" / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "m" / "merges.txt").write_text("#version: 0.2\n")
    GPT2TokenizerFast(str(tmp_path / "m" / "vocab.json"), str(tmp_path / "m" / "merges.txt")).save_pretrained(tmp_path / "m")
    model, tok = H.load_model(str(tmp_path / "m"), dtype="float32", device="cpu")
    assert isinstance(model, torch.nn.Module) and hasattr(tok, "encode") and not model.training
