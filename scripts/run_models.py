"""Capture one model once (exact), transcode to a ladder of specs, and measure each: ratio, rel L2, last-layer KL
(stored reconstruction), live chain KL (codec inside the forward pass), per-layer KL, and generation coherence.

    python scripts/run_models.py --model gpt2 --layers 8 9 10 11 --n-seq 128 --seq-len 256 --batch 16 \
        --specs exact faithful balanced compact --coherence-specs compact --out results/gpt2

Specs may be regime names, templates with {G}/{O}/{G8}/{O2} or plain spec strings. Keeps one JSON line per spec in <out>/results.jsonl
(specs already there are skipped, or extended when a new measurement is requested)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc                                   # noqa: E402
from rescomp import instrument as I                    # noqa: E402
from rescomp.hf import load_model                      # noqa: E402
from rescomp.bindings import bind                      # noqa: E402
from cli import PROMPTS                                # noqa: E402  (scripts/cli.py, beside this file)
from corpus import wikitext_sequences                  # noqa: E402

DEFAULT_SPECS = ["exact", "faithful", "balanced", "compact", "fp8_e4m3+out8", "nvfp4+layerK4+out8",
                 "mxfp4+layerK4+out8", "int4_g64+layerK4+out8", "int4_g{G}+layerK4+out{O}", "int4_g{G8}+out{O2}"]


def show(x, fmt=".4f"):
    return "-" if x is None else format(x, fmt)


def resolve_name(name, d):
    return rc.resolve_spec(name, d).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kl-seqs", type=int, default=None, help="sequences for KL/ppl (default: all)")
    ap.add_argument("--specs", nargs="*", default=DEFAULT_SPECS)
    ap.add_argument("--coherence-specs", nargs="*", default=[])
    ap.add_argument("--per-layer-specs", nargs="*", default=[])
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--n-new", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default=None, help="attn_implementation (eager for Gemma-2)")
    ap.add_argument("--device-map", default=None, help="auto: spread the model over all GPUs")
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-live", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for l in open(res_path):
            rec = json.loads(l)
            done[rec["spec"]] = rec
    torch.manual_seed(0)
    t0 = time.time()
    model, tok = load_model(a.model, dtype=a.dtype, revision=a.revision, attn_implementation=a.attn, device_map=a.device_map)
    dev = model.get_input_embeddings().weight.device
    print(f"loaded {a.model} in {time.time() - t0:.0f}s on {set(str(p.device) for p in model.parameters())}", flush=True)
    tokens = wikitext_sequences(tok, a.n_seq, a.seq_len, seed=0, bos=a.bos, split=a.dataset_split)
    L = bind(model, a.layers, tokenizer=tok, model_id=a.model, revision=a.revision)
    run = lambda t: model(t.to(dev), use_cache=False).logits
    base = str(out / "base")
    if not rc.is_capture(base):
        with rc.Sink(base, L, "exact", level=a.level, extra={"corpus": f"wikitext-103-raw-v1 {a.dataset_split}", "seed": 0}), torch.no_grad():
            for i in range(0, a.n_seq, a.batch):
                model(tokens[i: i + a.batch].to(dev), use_cache=False)
        print(f"captured {base} in {time.time() - t0:.0f}s", flush=True)
    with rc.open(base) as r:
        d = r.d
        ident = I.kl_ppl(run, tokens[:8], I.Patcher(L, lambda l, x, s, st: x), batch=a.batch)
    assert ident["kl_mean"] == 0.0, ident
    print(f"base: identity KL {ident['kl_mean']}", flush=True)
    n_kl = min(a.kl_seqs or a.n_seq, a.n_seq)
    for raw in a.specs:
        spec = resolve_name(raw, d)
        want_per = raw in a.per_layer_specs or spec in a.per_layer_specs
        want_coh = raw in a.coherence_specs or spec in a.coherence_specs
        stem = str(out / spec.replace("+", "_").replace("=", "-"))
        t1 = time.time()
        if spec in done:
            rec = done[spec]
            if (not want_per or "per_layer_eval" in rec) and (not want_coh or "coherence" in rec):
                print(f"skip {spec} (done)")
                continue
            if not rc.is_capture(stem):
                rc.transcode(base, stem, spec, sample_seqs=32, level=a.level)
            print(f"extend {spec}", flush=True)
        else:
            st = rc.transcode(base, stem, spec, sample_seqs=32, level=a.level)
            with rc.open(stem) as r:
                h = r.header
            rec = {"model": a.model, "layers": a.layers, "spec": spec, "spec_template": raw, "n_seq": a.n_seq,
                   "seq_len": a.seq_len, "stem": os.path.basename(stem), "bytes": h["bytes"], "ratio": st["ratio"],
                   "fidelity": h["fidelity"], "per_layer_rel_l2": [l["fidelity"]["rel_l2"] for l in h["layers"]],
                   "encode_seconds": round(time.time() - t1, 1)}
            t2 = time.time()
            with rc.open(stem) as r:
                rec["file_eval"] = I.kl_ppl(run, tokens[:n_kl], I.Patcher(L, r, only=[a.layers[-1]]), seq_ids=list(range(n_kl)), batch=a.batch)
                channels = r.channels
            rec["file_eval_seconds"] = round(time.time() - t2, 1)
            if not a.no_live:
                t3 = time.time()
                rec["live_eval"] = I.kl_ppl(run, tokens[:n_kl], I.Patcher(L, I.LiveCodec(spec, channels)), batch=a.batch)
                rec["live_eval_seconds"] = round(time.time() - t3, 1)
        if want_per and "per_layer_eval" not in rec:
            t5 = time.time()
            with rc.open(stem) as r:
                per = {l: I.kl_ppl(run, tokens[:n_kl], I.Patcher(L, r, only=[l]), seq_ids=list(range(n_kl)), batch=a.batch) for l in a.layers}
            rec["per_layer_eval"] = {str(l): {"kl_mean": v["kl_mean"], "ppl_ratio": v["ppl_ratio"], "top1_agree": v["top1_agree"]}
                                     for l, v in per.items()}
            rec["per_layer_eval_seconds"] = round(time.time() - t5, 1)
        if want_coh and "coherence" not in rec:
            t4 = time.time()
            with rc.open(stem) as r:
                co = I.coherence(run, PROMPTS[: a.n_prompts], I.Patcher(L, I.LiveCodec(spec, r.channels)), tokenizer=tok,
                                 n_new=a.n_new, seeds=a.seeds, device=dev)
            rec["coherence"] = co["summary"]
            rec["coherence_seconds"] = round(time.time() - t4, 1)
            with open(out / f"generations_{os.path.basename(stem)}.jsonl", "w") as f:
                for g in co["records"]:
                    f.write(json.dumps(g) + "\n")
        rec["seconds"] = rec.get("seconds", 0) + round(time.time() - t1, 1)
        done[spec] = rec
        with open(res_path, "w") as f:
            for r_ in done.values():
                f.write(json.dumps(r_) + "\n")
        fe, le, co = rec["file_eval"], rec.get("live_eval", {}), rec.get("coherence", {})
        print(f"{spec:40s} {rec['ratio']:5.2f}x relL2 {show(rec['fidelity']['rel_l2'], '.2e')} | last-layer KL {show(fe['kl_mean'])} "
              f"ppl x{show(fe['ppl_ratio'])} | live KL {show(le.get('kl_mean'))} ppl x{show(le.get('ppl_ratio'))}"
              + (f" | coh ppl x{show(co['ppl_ratio'], '.3f')} t={show(co['t'], '.1f')}" if co else "")
              + f"  ({rec['seconds']:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
