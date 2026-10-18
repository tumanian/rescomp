import pytest
import torch

import rescomp as rc

ALL_SPECS = ["fp32", "bf16", "fp16", "fp8_e4m3", "int8_tok", "int8_g64", "int4_g16", "int2_g16", "uabs0.05", "mxfp4",
             "nvfp4", "int4_g16+layerK2", "int4_g16+layerK4+out4", "int4_g16+layerK2+out4+sink1",
             "int4_g16+layerK4+out2+key=bf16", "uabs0.1+out8", "bf16+sink2", "fp16+layerK3+key=fp32"]


def make_acts(L=4, B=3, T=8, d=64, seed=0, outlier_channels=(5, 17)):
    """Residual-like activations: gaussian, a few massive channels, layers that drift from each other."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, T, d, generator=g) * 2
    xs = []
    for i in range(L):
        x = base + 0.3 * i * torch.randn(B, T, d, generator=g)
        for c in outlier_channels:
            x[..., c] = x[..., c] * 200 + 50 * (i + 1)
        x[:, 0, :] *= 20  # a sink-like first token
        xs.append(x)
    return torch.stack(xs)


@pytest.fixture
def acts():
    return make_acts()


@pytest.fixture
def tokens():
    return torch.randint(0, 1000, (3, 8))


@pytest.fixture(scope="session")
def tiny_model():
    """A random 4-layer GPT-2 (no download). fp32 so bf16 is a real (small) loss."""
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=256, n_positions=64, bos_token_id=0, eos_token_id=1)
    return GPT2LMHeadModel(cfg).eval()


class FakeTokenizer:
    name_or_path = "fake"
    bos_token_id = 0

    def encode(self, s, add_special_tokens=True):
        return [ord(c) % 256 for c in s]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture(scope="session")
def fake_tok():
    return FakeTokenizer()


@pytest.fixture(scope="session")
def tiny_moe():
    """A random 2-layer Qwen3-MoE with 4 experts, 2 active per token (no download)."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=48, moe_intermediate_size=48,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64,
                         num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
    return Qwen3MoeForCausalLM(cfg).eval()
