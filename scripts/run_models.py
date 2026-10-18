"""Capture one model once (lossless bf16), transcode to a ladder of specs, and measure each:
ratio, rel L2, KL + perplexity with the stored reconstructions ("file"), KL + perplexity with the live codec
("live"), and, for a few specs, generation coherence (paired perplexity under the clean model).

    python scripts/run_models.py --model gpt2 --layers 8 9 10 11 --n-seq 128 --seq-len 256 --batch 16 \
        --specs bf16 fp16 int4_g384+layerK4+out12 --coherence-specs int4_g384+layerK4+out12 --out results/gpt2

Keeps one JSON line per spec in <out>/results.jsonl (specs already there are skipped, or extended when a
per-layer or coherence measurement is newly requested) and keeps the .rsc files.
"uabs" specs may be written as "uabs@0.1" = step 0.1 x the rms of the calibration sample (outlier channels excluded)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc                                   # noqa: E402
from rescomp import capture as C, evaluate as E       # noqa: E402
from rescomp.evaluate import PROMPTS                   # noqa: E402

DEFAULT_SPECS = ["bf16", "fp16", "fp8_e4m3+out8", "int8_tok+out8", "uabs@0.1+out8",
                 "int4_g64+layerK4+out8", "int4_g{G}+layerK4+out{O}", "int4_g{G}+out{O}",
                 "int4_g{G}+layerK4+out{O}+key=bf16", "int4_g{G8}+layerK4+out{O2}", "int4_g{G8}+layerK4+out{O2}+key=bf16",
                 "nvfp4+layerK4+out8", "mxfp4+layerK4+out8", "int2_g16+layerK4+out16"]


def show(x, fmt=".4f"):
    return "-" if x is None else format(x, fmt)


def resolve_spec(name, d, rms):
    """Fill the size placeholders from d and uabs@ steps from the rms:
    {G} = d/2 and {O} = d/64 (the GPT-2 winner's proportions: g384 + out12 at d = 768),
    {G8} = d/8 and {O2} = d/128 (the Qwen3-8B winner's: g512 + out32 at d = 4096)."""
    name = (name.replace("{G8}", str(d // 8)).replace("{G}", str(d // 2))
                .replace("{O2}", str(max(8, d // 128))).replace("{O}", str(max(8, d // 64))))
    if name.startswith("uabs@"):
        head, *rest = name.split("+")
        step = np.format_float_positional(float(head[5:]) * rms, precision=4, trim="-", fractional=False)
        name = "+".join([f"uabs{step}"] + rest)
    return name


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
    ap.add_argument("--per-layer-specs", nargs="*", default=[], help="also KL with one layer replaced at a time")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--n-new", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default=None, help="attn_implementation (eager for Gemma-2)")
    ap.add_argument("--device-map", default=None, help="auto: spread the model over all GPUs")
    ap.add_argument("--routers", default=None, help="MoE: store routing as annotations: 'stored' or 'all'")
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
    model, tok = C.load_model(a.model, dtype=a.dtype, revision=a.revision, attn_implementation=a.attn, device_map=a.device_map)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s on {set(str(p.device) for p in model.parameters())}", flush=True)
    tokens = C.wikitext_sequences(tok, a.n_seq, a.seq_len, seed=0, bos=a.bos, split=a.dataset_split)
    base = out / "base_bf16.rsc"
    if not base.exists():
        C.capture(model, tok, tokens, a.layers, "bf16", base, batch=a.batch, model_id=a.model, revision=a.revision,
                  level=a.level, extra={"corpus": f"wikitext-103-raw-v1 {a.dataset_split}", "seed": 0}, routers=a.routers)
    with rc.Reader(base) as r:
        d = r.d
        ident = E.identity_check(model, r.tokens(slice(0, min(8, r.n_seq))), r.layers, batch=a.batch)
        assert ident["kl_mean"] == 0.0, ident
        sample = r.read_stack(seqs=slice(0, min(32, r.n_seq)))
        cal = rc.calibrate(sample, f"bf16+out{max(8, d // 64)}")
        keep = torch.ones(d, dtype=torch.bool)
        keep[cal.outliers.flatten().unique()] = False
        rms = float(sample[..., keep].pow(2).mean().sqrt())
        base_h = r.header
    print(f"base: {base_h['bytes']['ratio_vs_bf16']:.2f}x, identity KL {ident['kl_mean']}, rms (no outliers) {rms:.3g}", flush=True)
    n_kl = a.kl_seqs or a.n_seq
    for raw in a.specs:
        spec = resolve_spec(raw, d, rms)
        want_per = raw in a.per_layer_specs or spec in a.per_layer_specs
        want_coh = raw in a.coherence_specs or spec in a.coherence_specs
        path = out / f"{spec.replace('+', '_').replace('=', '-')}.rsc"
        t1 = time.time()
        if spec in done:
            rec = done[spec]
            if (not want_per or "per_layer_eval" in rec) and (not want_coh or "coherence" in rec):
                print(f"skip {spec} (done)")
                continue
            if not path.exists():
                rc.transcode(base, path, spec, calib_seqs=32, level=a.level)
            print(f"extend {spec}", flush=True)
        else:
            h = rc.transcode(base, path, spec, calib_seqs=32, level=a.level)
            rec = {"model": a.model, "layers": a.layers, "spec": spec, "spec_template": raw, "n_seq": a.n_seq,
                   "seq_len": a.seq_len, "file": path.name, "file_bytes": os.path.getsize(path),
                   "bytes": h["bytes"], "ratio": h["bytes"]["ratio_vs_bf16"], "fidelity": h["fidelity"],
                   "per_layer_rel_l2": [l["fidelity"]["rel_l2"] for l in h["layers"]],
                   "encode_seconds": round(time.time() - t1, 1)}
            t2 = time.time()
            rec["file_eval"] = E.evaluate_file(path, model, batch=a.batch, n_seq=n_kl)
            rec["file_eval_seconds"] = round(time.time() - t2, 1)
            if not a.no_live:
                t3 = time.time()
                with rc.Reader(path) as r:
                    rec["live_eval"] = E.evaluate_live(model, r.tokens(slice(0, n_kl)), r.layers, r.spec, r.calib, batch=a.batch)
                rec["live_eval_seconds"] = round(time.time() - t3, 1)
        if want_per and "per_layer_eval" not in rec:
            t5 = time.time()
            per = E.evaluate_file_per_layer(path, model, batch=a.batch, n_seq=n_kl)
            rec["per_layer_eval"] = {str(l): {"kl_mean": v["kl_mean"], "ppl_ratio": v["ppl_ratio"], "top1_agree": v["top1_agree"]}
                                     for l, v in per.items()}
            rec["per_layer_eval_seconds"] = round(time.time() - t5, 1)
        if want_coh and "coherence" not in rec:
            t4 = time.time()
            with rc.Reader(path) as r:
                co = E.coherence(model, PROMPTS[: a.n_prompts], r.layers, r.spec, r.calib, tokenizer=tok,
                                 n_new=a.n_new, seeds=a.seeds)
            rec["coherence"] = co["summary"]
            rec["coherence_seconds"] = round(time.time() - t4, 1)
            with open(out / f"generations_{path.stem}.jsonl", "w") as f:
                for g in co["records"]:
                    f.write(json.dumps(g) + "\n")
        rec["seconds"] = rec.get("seconds", 0) + round(time.time() - t1, 1)
        done[spec] = rec
        with open(res_path, "w") as f:                      # rewrite: extended rows replace their old line
            for r_ in done.values():
                f.write(json.dumps(r_) + "\n")
        fe, le, co = rec["file_eval"], rec.get("live_eval", {}), rec.get("coherence", {})
        print(f"{spec:40s} {rec['ratio']:5.2f}x relL2 {rec['fidelity']['rel_l2']:.2e} | file KL {show(fe['kl_mean'])} "
              f"ppl x{show(fe['ppl_ratio'])} | live KL {show(le.get('kl_mean'))} ppl x{show(le.get('ppl_ratio'))}"
              + (f" | coh ppl x{show(co['ppl_ratio'], '.3f')} t={show(co['t'], '.1f')}" if co else "")
              + f"  ({rec['seconds']:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
