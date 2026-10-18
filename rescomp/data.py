"""Token packing helpers (any tokenizer with encode(text, add_special_tokens=) and bos_token_id)."""
import numpy as np
import torch


def pack_sequences(tokenizer, texts, n_seq, seq_len, bos=False):
    """Concatenate texts (one newline between them) and cut n_seq windows of seq_len tokens. Texts are encoded
    without special tokens, so no BOS lands mid-window; bos=True starts every window with one BOS token (what
    Gemma/Llama-style models and their SAEs expect)."""
    body = seq_len - 1 if bos else seq_len
    need, ids = n_seq * body, []
    for t in texts:
        ids.extend(tokenizer.encode(t + "\n", add_special_tokens=False))
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"texts give {len(ids)} tokens, need {need} for {n_seq} x {seq_len}")
    tokens = torch.tensor(ids[:need], dtype=torch.long).view(n_seq, body)
    if bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("tokenizer has no BOS token")
        tokens = torch.cat([torch.full((n_seq, 1), tokenizer.bos_token_id, dtype=torch.long), tokens], 1)
    return tokens


def wikitext_sequences(tokenizer, n_seq, seq_len, seed=0, bos=False, split="train"):
    """wikitext-103 paragraphs in a seeded random order, packed into windows (headings dropped). Needs `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    lines = [t.strip() for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    order = np.random.default_rng(seed).permutation(len(lines))
    return pack_sequences(tokenizer, (lines[i] for i in order), n_seq, seq_len, bos=bos)
