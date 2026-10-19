"""rescomp command line.

    rescomp info STEM
    rescomp capture --model gpt2 --layers 8 9 10 11 --spec balanced --n-seq 200 --seq-len 256 --out runs/gpt2
    rescomp eval STEM [--model gpt2] [--n-seq 64] [--live] [--coherence]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPTS = [
    "The history of the Roman Empire", "In a small village at the edge of the forest, an old woman",
    "Photosynthesis is the process by which", "The best way to learn a new programming language is",
    "When the spacecraft finally reached the orbit of Jupiter,", "The city council met on Tuesday to discuss",
    "My grandfather always told me that", "The main causes of the French Revolution were",
    "To make a simple tomato sauce, first", "Scientists have recently discovered that",
    "The museum's newest exhibit focuses on", "Before the invention of the printing press,",
    "The detective looked at the muddy footprints and", "Climate change affects coastal cities because",
    "The first step in training a neural network is", "She opened the letter and read the first line:",
    "The economy of ancient Egypt depended on", "A healthy breakfast should include",
    "The rules of chess are simple:", "After years of drought, the farmers decided to",
]


def cmd_info(a):
    from rescomp.store import Reader
    with Reader(a.stem, verify=not a.no_verify) as r:
        print(r.summary())
        if a.json:
            print(json.dumps({k: v for k, v in r.header.items() if k != "layers"}, indent=1))


def cmd_capture(a):
    from corpus import wikitext_sequences          # beside this file
    from rescomp.hf import load_model
    from rescomp.bindings import bind
    from rescomp.sink import Sink
    model, tok = load_model(a.model, dtype=a.dtype, revision=a.revision, attn_implementation=a.attn)
    tokens = wikitext_sequences(tok, a.n_seq, a.seq_len, seed=a.seed, bos=a.bos)
    with Sink(a.out, bind(model, a.layers, tokenizer=tok, model_id=a.model, revision=a.revision), a.spec, level=a.level,
              fidelity=a.fidelity) as sink, torch.no_grad():
        for i in range(0, a.n_seq, a.batch):
            model(tokens[i: i + a.batch].to(sink.layers.root.get_input_embeddings().weight.device), use_cache=False)
        print(json.dumps(sink.stats(), indent=1, default=str))


def cmd_eval(a):
    from rescomp import instrument as I
    from rescomp.hf import load_model
    from rescomp.bindings import bind
    from rescomp.store import Reader
    with Reader(a.stem) as r:
        header, layers, spec, channels = r.header, r.layers, r.spec, r.channels
        n = min(a.n_seq or r.n_seq, r.n_seq)
        toks = r.tokens(slice(0, n))
        if any(t is None for t in toks) or len({len(t) for t in toks}) != 1:
            raise SystemExit("eval needs stored tokens of one length")
        tokens = torch.stack(toks)
    model_id = a.model or header["model"]["id"]
    model, tok = load_model(model_id, dtype=a.dtype, revision=header["model"].get("revision"))
    L = bind(model, layers, tokenizer=tok)
    dev = model.get_input_embeddings().weight.device
    run = lambda t: model(t.to(dev), use_cache=False).logits
    out = {"stem": a.stem, "model": model_id, "spec": spec.name, "bytes": header["bytes"], "fidelity": header["fidelity"]}
    with Reader(a.stem) as r:
        out["last_layer"] = I.kl_ppl(run, tokens, I.Patcher(L, r, only=[layers[-1]]), seq_ids=list(range(n)), batch=a.batch)
    if a.live:
        out["live"] = I.kl_ppl(run, tokens, I.Patcher(L, I.LiveCodec(spec, channels)), batch=a.batch)
    if a.coherence:
        out["coherence"] = I.coherence(run, PROMPTS[: a.n_prompts], I.Patcher(L, I.LiveCodec(spec, channels)),
                                       tokenizer=tok, n_new=a.n_new, seeds=a.seeds, device=dev)["summary"]
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rescomp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="print a capture's header")
    p.add_argument("stem")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("capture", help="run an HF model over wikitext with a Sink attached")
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--spec", default="balanced")
    p.add_argument("--n-seq", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", help="attn_implementation, e.g. eager for Gemma-2")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bos", action="store_true")
    p.add_argument("--fidelity", action="store_true", help="measure reconstruction error in the hook")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("eval", help="KL / perplexity / coherence of a capture against its model")
    p.add_argument("stem")
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
