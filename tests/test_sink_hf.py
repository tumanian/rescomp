"""Sink on Hugging Face models: hf_layers, real generate() with the KV cache, MoE models, TransformerLens."""
import pytest
import torch

import rescomp as rc
from rescomp.hf import HFAdapter


def test_hf_layers_and_forward(tiny_model, fake_tok, tmp_path):
    L = rc.hf_layers(tiny_model, [1, 3], fake_tok, model_id="tiny")
    assert sorted(L.modules) == [1, 3] and L.root is tiny_model and L.n_layers == 4
    assert L.hook_names[3] == {"hf": "transformer.h.3", "transformer_lens": "blocks.3.hook_resid_post"}
    assert L.model_info["id"] == "tiny" and L.model_info["loader"] == "transformers"
    toks = torch.randint(2, 256, (5, 10), generator=torch.Generator().manual_seed(0))
    mask = torch.ones(5, 10, dtype=torch.long)
    mask[4, 7:] = 0
    with rc.Sink(str(tmp_path / "hf"), L, "int4_g16+layerK2+out2") as sink:
        tiny_model(input_ids=toks[:3], attention_mask=mask[:3])
        tiny_model(toks[3:], attention_mask=mask[3:])
    with rc.open(str(tmp_path / "hf")) as r:
        assert r.n_seq == 5 and r.header["n_chunks"] == 2
        assert torch.equal(torch.stack(r.tokens()), toks) and r.attention_mask(4).tolist() == [1] * 7 + [0] * 3
        _, acts, _ = HFAdapter(tiny_model).forward(toks[:3], capture=[1, 3])   # the encoder saw the fp32 activations
        exp = rc.encode(torch.stack([acts[1], acts[3]]), r.spec, r.calib)[1]
        assert torch.equal(torch.stack(r.read(3, [0, 1, 2])), exp[1])
    with pytest.raises(ValueError):
        rc.hf_layers(tiny_model, [9])


def test_generate_with_kv_cache(tiny_model, fake_tok, tmp_path):
    """Real HF generation: prefill + per-token steps with past_key_values; one sequence per prompt in the file."""
    prompt = torch.randint(2, 256, (2, 4), generator=torch.Generator().manual_seed(3))
    with rc.Sink(str(tmp_path / "gen"), rc.hf_layers(tiny_model, [0, 2, 3], fake_tok), "int8_tok+layerK3") as sink:
        out = tiny_model.generate(prompt, max_new_tokens=5, do_sample=False, pad_token_id=1, use_cache=True)
        stats = sink.stats()
    # 5 new tokens take 5 forward calls (prefill + 4 steps); the last sampled token is never run through the model
    assert out.shape == (2, 9) and stats["calls"] == 5 and stats["incomplete_calls"] == 0
    with rc.open(str(tmp_path / "gen")) as r:
        assert r.n_seq == 2 and r.header["n_chunks"] == 5              # one chunk per forward call
        assert r.sequences().column("n_tokens").to_pylist() == [8, 8]
        assert torch.equal(torch.stack(r.tokens()), out[:, :8])        # prompt + generated tokens, in order
        x = r.read(3, 0)
        assert x.shape == (8, 64)
        # the stored prefill part is exactly the encoding of a one-shot run of the prompt
        _, acts, _ = HFAdapter(tiny_model).forward(prompt, capture=[0, 2, 3])
        exp_prompt = rc.encode(torch.stack([acts[l] for l in (0, 2, 3)]), r.spec)[1]
        assert torch.equal(r.read(0, 0)[:4], exp_prompt[0, 0]) and torch.equal(r.read(3, 1)[:4], exp_prompt[2, 1])


def test_generate_single_layer(tiny_model, fake_tok, tmp_path):
    prompt = torch.randint(2, 256, (1, 3))
    with rc.Sink(str(tmp_path / "geni"), rc.hf_layers(tiny_model, [3], fake_tok), "bf16") as sink:
        out = tiny_model.generate(prompt, max_new_tokens=4, do_sample=False, pad_token_id=1)
    with rc.open(str(tmp_path / "geni")) as r:
        assert r.header["n_chunks"] == 4 and r.n_seq == 1 and r.read(3, 0).shape == (6, 64)
        assert torch.equal(r.tokens(0), out[0, :6])


def test_moe_routing_via_sink(tiny_moe, fake_tok, tmp_path):
    """Routing is not captured by the Sink (that is the HFAdapter's job); the MoE model still records fine."""
    toks = torch.randint(2, 64, (3, 6))
    with rc.Sink(str(tmp_path / "moe"), rc.hf_layers(tiny_moe, [1], fake_tok), "fp16") as sink:
        tiny_moe(toks)
    with rc.open(str(tmp_path / "moe")) as r:
        assert r.n_seq == 3 and r.header["model"]["config"]["num_experts"] == 4 and r.read(1, 0).shape == (6, 32)


def test_tl_layers_if_installed(tmp_path):
    tl = pytest.importorskip("transformer_lens")
    cfg = tl.HookedTransformerConfig(n_layers=2, d_model=16, d_head=8, n_heads=2, d_mlp=32, d_vocab=40, n_ctx=16, act_fn="gelu")
    m = tl.HookedTransformer(cfg).eval()
    toks = torch.randint(0, 40, (2, 8))
    with rc.Sink(str(tmp_path / "tl"), rc.tl_layers(m, [0, 1]), "bf16") as sink:
        m(toks)
    with rc.open(str(tmp_path / "tl")) as r:
        assert r.n_seq == 2 and torch.equal(torch.stack(r.tokens()), toks) and r.read(1, 0).shape == (8, 16)
