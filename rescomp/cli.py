"""rescomp command line.

    rescomp info FILE.rsc
    rescomp capture --model gpt2 --layers 8 9 10 11 --spec int4_g384+layerK4+out12 --n-seq 200 --seq-len 256 --out gpt2.rsc
    rescomp eval FILE.rsc [--model gpt2] [--n-seq 64] [--live] [--coherence]
"""
import argparse
import json
import sys

import torch


def cmd_info(a):
    from .container import Reader
    with Reader(a.file, verify=not a.no_verify) as r:
        print(r.summary())
        if a.json:
            h = dict(r.header)
            h.pop("chunks")
            print(json.dumps(h, indent=1))


def cmd_capture(a):
    from .capture import capture, wikitext_sequences
    from .hf import load_model
    model, tok = load_model(a.model, dtype=a.dtype, revision=a.revision, attn_implementation=a.attn)
    tokens = wikitext_sequences(tok, a.n_seq, a.seq_len, seed=a.seed, bos=a.bos)
    capture(model, tokens, a.layers, a.spec, a.out, batch=a.batch, calib_seqs=a.calib_seqs, level=a.level)


def cmd_eval(a):
    from .container import Reader
    from .hf import load_model
    from .evaluate import PROMPTS, coherence, evaluate_file, evaluate_live
    with Reader(a.file) as r:
        header, layers, spec, calib = r.header, r.layers, r.spec, r.calib
        n = min(a.n_seq or r.n_seq, r.n_seq)
        tokens = r.tokens(slice(0, n))
    model_id = a.model or header["model"]["id"]
    model, tok = load_model(model_id, dtype=a.dtype, revision=header["model"].get("revision"))
    out = {"file": a.file, "model": model_id, "header_bytes": header["bytes"], "fidelity": header["fidelity"]}
    out["file_eval"] = evaluate_file(a.file, model, batch=a.batch, n_seq=n)
    if a.live:
        out["live_eval"] = evaluate_live(model, tokens, layers, spec, calib, batch=a.batch)
    if a.coherence:
        out["coherence"] = coherence(model, PROMPTS[: a.n_prompts], layers, spec, calib, tokenizer=tok,
                                     n_new=a.n_new, seeds=a.seeds)["summary"]
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)




def main(argv=None):
    ap = argparse.ArgumentParser(prog="rescomp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="print a file's header")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("capture", help="capture an HF model's resid_post into a file")
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--spec", default="bf16")
    p.add_argument("--n-seq", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--calib-seqs", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", help="attn_implementation, e.g. eager for Gemma-2")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bos", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("eval", help="KL / perplexity / coherence of a file against its model")
    p.add_argument("file")
    p.add_argument("--model")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-seq", type=int)
    p.add_argument("--live", action="store_true")
    p.add_argument("--coherence", action="store_true")
    p.add_argument("--n-prompts", type=int, default=10)
    p.add_argument("--n-new", type=int, default=80)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
