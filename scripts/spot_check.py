"""Spot checks on a real model: what the Sink wrote is what the model saw.

    python scripts/spot_check.py [--model gpt2] [--layers 8 9 10 11]

For each regime: run GPT-2 over a few paragraphs with a Sink attached and plain forward hooks capturing the same
residual stream; then open the file and check (1) the stored token ids are the tokenizer's ids and decode back
to the input text, (2) the attention masks match, (3) the stored activations are the hooked ones (exactly for
`exact`, within the regime's error otherwise), (4) a generate() run with the KV cache stores the prompt plus the
generated tokens in order, (5) re-encoding the hooked activations on the CPU gives the same bytes as the GPU
encode for the exact regime and the keyframes (delta layers and nvfp4 may differ in the last bits).
Then the TransformSink (--pooled, on by default): (6) the stored "last" rows are the last real token of the
hooked activations bit for bit and the "mean" rows the masked mean within fp32 rounding (fp32, the default
codec), the "exact" regime rounds both to fp16, (7) Reader.pooled on a full capture gives the same rows, (8) a
generate() run pools the prompt once and skips the cached steps.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc  # noqa: E402
from rescomp.hf import load_model  # noqa: E402

TEXTS = [
    "The history of the Roman Empire begins with the founding of the city in the eighth century BC.",
    "Photosynthesis is the process by which green plants turn sunlight, water and carbon dioxide into sugar.",
    "In a small village at the edge of the forest, an old woman kept bees and sold honey at the market.",
    "The city council met on Tuesday to discuss the budget for the new library and the repair of the bridge.",
    "To make a simple tomato sauce, first heat olive oil in a pan and add garlic, then the tomatoes and salt.",
    "Before the invention of the printing press, every book had to be copied by hand, slowly and at great cost.",
]


def hooked(model, blocks, ids, mask):
    acts = {}
    hs = [b.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
          for l, b in blocks.items()]
    try:
        with torch.no_grad():
            model(ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return acts


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument("--regimes", nargs="*", default=["exact", "faithful", "balanced", "compact"])
    ap.add_argument("--pooled", action=argparse.BooleanOptionalAction, default=True, help="also check the TransformSink")
    a = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load_model(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = model.get_input_embeddings().weight.device
    enc = tok(TEXTS, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    from rescomp.hf import decoder_blocks
    blocks = {l: decoder_blocks(model)[0][l] for l in a.layers}
    ref = hooked(model, blocks, ids, mask)
    ok = True
    bound = {"exact": 2e-3, "faithful": 2e-2, "balanced": 1e-1, "compact": 1e-1}   # GPT-2 layer 11 sits near 6e-2 at int4
    with tempfile.TemporaryDirectory() as d:
        for regime in a.regimes:
            print(f"regime {regime}")
            stem = f"{d}/{regime}"
            with rc.Sink(stem, rc.bind(model, a.layers, tokenizer=tok, model_id=a.model), regime, fidelity=True) as sink:
                with torch.no_grad():
                    model(ids, attention_mask=mask, use_cache=False)
                sink.stats()
            with rc.open(stem) as r:
                ok &= check(r.n_seq == len(TEXTS) and r.spec.name == rc.resolve_spec(regime, r.d).name, f"{r.n_seq} sequences, spec {r.spec.name}, ratio {r.header['bytes']['ratio_vs_bf16']:.2f}x")
                toks = torch.stack(r.tokens())
                ok &= check(torch.equal(toks, ids.cpu()), "stored token ids == tokenizer ids")
                ok &= check(torch.equal(torch.stack(r.attention_mask()).long(), mask.cpu().long()), "stored attention mask == input mask")
                texts = [tok.decode(t[m.bool()]) for t, m in zip(r.tokens(), r.attention_mask())]
                ok &= check(texts == TEXTS, "stored tokens decode back to the input text")
                print(f"       seq 2 decodes to: {texts[2][:60]!r}...")
                for l in a.layers:
                    x = torch.stack(r.read(l))
                    rel = ((x - ref[l]).norm() / ref[l].norm()).item()
                    ok &= check(rel <= bound[regime], f"layer {l}: rel L2 vs hooked activations {rel:.2e} (bound {bound[regime]:.0e}); file says {r.header['layers'][r.layers.index(l)]['fidelity']['rel_l2']:.2e}")
                # bytes: the CPU encode of the hooked activations vs what the GPU Sink wrote
                e = r.codec.encoder(len(a.layers))
                for i, l in enumerate(a.layers):
                    strm, _ = e.encode_layer(i, ref[l])
                    stored = r.streams(0, l)
                    same = all(torch.equal(strm[k], stored[k]) for k in strm)
                    key = r.spec.is_key(i)
                    print(f"       layer {l} {'key  ' if key else 'delta'} CPU re-encode bytes {'==' if same else '!='} GPU-written bytes")
                    if regime == "exact" or key:
                        ok &= check(same, f"layer {l}: CPU and GPU encodes agree byte for byte")
            # generation with the KV cache
            prompt = tok(TEXTS[0].split(",")[0], return_tensors="pt").to(dev)
            gstem = f"{d}/{regime}-gen"
            with rc.Sink(gstem, rc.bind(model, a.layers, tokenizer=tok), regime) as sink:
                out = model.generate(**prompt, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
                calls = sink.n_calls
            with rc.open(gstem) as r:
                n = out.shape[1] - 1                          # the last generated token is never run through the model
                ok &= check(calls == 12 and r.header["n_chunks"] == 12, f"generate: {calls} forward calls, {r.header['n_chunks']} chunks")
                ok &= check(torch.equal(r.tokens(0), out[0, :n].cpu()), "generate: stored tokens == prompt + generated tokens")
                ok &= check(r.read(a.layers[-1], 0).shape[0] == n, f"generate: {n} stored positions")
                print(f"       generated: {tok.decode(out[0])!r}")
        if a.pooled:
            ok &= check_pooled(model, tok, a, ids, mask, ref, d)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def masked_rows(x, mask):
    """[B, T, d], [B, T] -> (mean over real tokens, last real token), each [B, d], the slow way."""
    means, lasts = [], []
    for xi, mi in zip(x, mask.cpu().bool()):
        real = xi[mi]
        means.append(real.mean(0))
        lasts.append(real[-1])
    return torch.stack(means), torch.stack(lasts)


def check_pooled(model, tok, a, ids, mask, ref, d):
    """The TransformSink on the same texts: pooled rows vs the hooked activations, a full capture, generate()."""
    ok = True
    print("pooled rows (TransformSink)")
    for codec in ("fp32", "exact"):
        stem = f"{d}/pooled-{codec}"
        with rc.TransformSink(stem, rc.bind(model, a.layers, tokenizer=tok, model_id=a.model), codec) as sink:
            with torch.no_grad():
                model(ids, attention_mask=mask, use_cache=False)
            st = sink.stats()
        with rc.open(stem) as r:
            ok &= check(r.n_seq == len(TEXTS) and r.header["n_chunks"] == 1 and r.pooled_rows == ["mean", "last"]
                        and r.header["site"] == "resid_post.mean+last", f"codec {codec}: {r.n_seq} sequences in {r.header['n_chunks']} chunk, rows {r.pooled_rows}, site {r.header['site']}")
            ok &= check(torch.equal(torch.stack(r.tokens()), ids.cpu()), "stored token ids == tokenizer ids")
            for l in a.layers:
                mean, last = masked_rows(ref[l], mask)
                got_mean, got_last = r.pooled(l, "mean"), r.pooled(l, "last")
                if codec == "fp32":
                    rel = ((got_mean - mean).norm() / mean.norm()).item()
                    ok &= check(torch.equal(got_last, last), f"layer {l}: stored last rows == last real token of the hooked activations, bit for bit")
                    ok &= check(rel < 1e-5, f"layer {l}: stored mean rows vs masked mean of the hooked activations: rel L2 {rel:.1e} (fp32 summation order)")
                else:
                    ok &= check(torch.equal(got_last, last.half().float()), f"layer {l}: exact (fp16) codec stores the last rows rounded to fp16")
                    ok &= check(torch.equal(got_mean, got_mean.half().float()) and ((got_mean - mean).norm() / mean.norm()).item() < 2e-3,
                                f"layer {l}: exact (fp16) codec stores the mean rows rounded to fp16")
            if codec == "fp32":
                full = f"{d}/full-fp32"
                with rc.Sink(full, rc.bind(model, a.layers, tokenizer=tok), "fp32") as sink:
                    with torch.no_grad():
                        model(ids, attention_mask=mask, use_cache=False)
                with rc.open(full) as f:
                    same = all(torch.allclose(f.pooled(l, w), r.pooled(l, w), rtol=1e-5, atol=1e-4) for l in a.layers for w in ("mean", "last"))
                    ok &= check(same, "Reader.pooled on the full capture == the TransformSink's rows (mean within fp32 rounding, last exact)")
                    ok &= check(all(torch.equal(f.pooled(l, "last"), r.pooled(l, "last")) for l in a.layers), "  last rows bit for bit")
    prompt = tok(TEXTS[0].split(",")[0], return_tensors="pt").to(model.get_input_embeddings().weight.device)
    gstem = f"{d}/pooled-gen"
    with rc.TransformSink(gstem, rc.bind(model, a.layers, tokenizer=tok), "fp32") as sink:
        out = model.generate(**prompt, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        st = sink.stats()
    with rc.open(gstem) as r:
        ok &= check(st["calls"] == 12 and st["skipped_calls"] == 11 and r.header["n_chunks"] == 1, f"generate: {st['calls']} forward calls, {st['skipped_calls']} skipped as cached steps, {r.header['n_chunks']} chunk")
        ok &= check(torch.equal(r.tokens(0), prompt["input_ids"][0].cpu()), "generate: stored tokens == the prompt only")
        pref = hooked(model, {l: decoder_blocks_of(model)[l] for l in a.layers}, prompt["input_ids"], None)
        ok &= check(all(torch.equal(r.pooled(l, "last")[0], pref[l][0, -1]) for l in a.layers), "generate: last rows == last prompt token of a plain prompt forward")
    return ok


def decoder_blocks_of(model):
    from rescomp.hf import decoder_blocks
    return decoder_blocks(model)[0]


if __name__ == "__main__":
    sys.exit(main())
