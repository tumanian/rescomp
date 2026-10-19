"""Simple probes on stored activations: does a probe trained on one kind of capture work on another?

    python scripts/probes.py --model gpt2 --layers 6 9 11 --train 2000 --test 872
    python scripts/probes.py --model Qwen/Qwen3-1.7B --layers 14 21 27 --train 2000 --test 872

SST-2 sentences (positive / negative) go through the model once with a lossless full Sink ("exact", an fp16 cast,
for a bf16 model; --lossless fp32 for an fp32 model) and a TransformSink (fp32, "mean+last") attached together, so
both see the same forward; then once more per lossy
regime (--regimes). From every capture `Reader.pooled(layer, row)` gives one vector per sentence (the full captures
compute it from the stored tokens and the mask, the TransformSink stored it). A logistic-regression probe per
(layer, row) is trained on the lossless full capture's features and on the TransformSink's, and each probe is
tested on every capture's features. Reported: feature error of each source against the lossless full capture,
test accuracy per (train source, test source), and how often the predictions agree with the lossless ones.
Writes results/probes/<model>.md.
"""
import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc                                   # noqa: E402
from rescomp.hf import load_model                      # noqa: E402

POOLED = "TransformSink"


def sst2(n_train, n_test, seed):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    g = torch.Generator().manual_seed(seed)
    tr = ds["train"].select(torch.randperm(len(ds["train"]), generator=g)[:n_train].tolist())
    te = ds["validation"].select(range(min(n_test, len(ds["validation"]))))
    return [(t["sentence"].strip(), int(t["label"]), "train") for t in tr] + [(t["sentence"].strip(), int(t["label"]), "test") for t in te]


def capture(model, tok, items, sinks, batch, dev):
    """Run the items through the model with the given sinks attached together; each batch is annotated with its
    labels and split on every sink."""
    for s in sinks:
        s.__enter__()
    try:
        with torch.no_grad():
            for i in range(0, len(items), batch):
                part = items[i: i + batch]
                enc = tok([t for t, _, _ in part], return_tensors="pt", padding=True, truncation=True, max_length=128)
                for s in sinks:
                    s.annotate(label=[l for _, l, _ in part], split=[sp for _, _, sp in part])
                model(enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), use_cache=False)
    finally:
        for s in sinks:
            s.__exit__(None, None, None)


def features(stem, layers, rows):
    """{(layer, row): (X_train, y_train, X_test, y_test)} from one capture, by its split annotation."""
    out = {}
    with rc.open(stem) as r:
        y = torch.tensor(r.annotation("label"))
        split = r.annotation("split")
        tr = torch.tensor([s == "train" for s in split])
        for l in layers:
            for row in rows:
                X = r.pooled(l, row)
                out[(l, row)] = (X[tr], y[tr], X[~tr], y[~tr])
        info = {"bytes": r.header["bytes"]["activations_stored"], "ratio": r.header["bytes"]["ratio_vs_bf16"], "spec": r.spec.name}
    return out, info


def train_probe(X, y, n_classes, l2=1e-3, iters=300):
    """Multinomial logistic regression on standardized features, L-BFGS, fp32 on the CPU."""
    X = X.double()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    W = torch.zeros(X.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xn @ W + b, y) + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, W.detach(), b.detach()


def predict(probe, X):
    mu, sd, W, b = probe
    return ((X.double() - mu) / sd @ W + b).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 9, 11])
    ap.add_argument("--regimes", nargs="*", default=["balanced", "compact"], help="lossy full captures to test on")
    ap.add_argument("--rows", nargs="*", default=["mean", "last"])
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=872)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lossless", default="exact", help="codec of the lossless full capture: exact (fp16) for a bf16 model, fp32 for an fp32 one")
    ap.add_argument("--out", default=None, help="directory for the captures (default: a temporary one)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    FULL = f"full {a.lossless}"
    model, tok = load_model(a.model, dtype=a.dtype)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = model.get_input_embeddings().weight.device
    items = sst2(a.train, a.test, a.seed)
    n_classes = 1 + max(l for _, l, _ in items)
    print(f"{a.model} ({a.dtype}) layers {a.layers} rows {a.rows}: SST-2, {a.train} train / {a.test} test sentences, batch {a.batch} on {dev}")
    tmp = tempfile.TemporaryDirectory() if a.out is None else None
    out = Path(a.out or tmp.name)
    out.mkdir(parents=True, exist_ok=True)
    bind = lambda: rc.bind(model, a.layers, tokenizer=tok, model_id=a.model)
    t0 = time.time()
    stems = {FULL: str(out / "full-lossless"), POOLED: str(out / "pooled")}
    capture(model, tok, items, [rc.Sink(stems[FULL], bind(), a.lossless), rc.TransformSink(stems[POOLED], bind())], a.batch, dev)
    for regime in a.regimes:
        stems[f"full {regime}"] = str(out / f"full-{regime}")
        capture(model, tok, items, [rc.Sink(stems[f"full {regime}"], bind(), regime)], a.batch, dev)
    print(f"captured in {time.time() - t0:.0f} s")
    feats, info = {}, {}
    for src, stem in stems.items():
        feats[src], info[src] = features(stem, a.layers, a.rows)
    lines = [f"# Probes on {a.model} ({a.dtype}): SST-2, layers {a.layers}, {a.train} train / {a.test} test sentences", "",
             "| Capture | spec | stored bytes | ratio vs bf16 |", "|---|---|---|---|"]
    for src in stems:
        lines.append(f"| {src} | `{info[src]['spec']}` | {info[src]['bytes'] / 1e6:.1f} MB | {info[src]['ratio']:.2f}x |")
    lines += ["", "Feature error against the lossless full capture (rel L2 over the test set):", "",
              "| Layer | row | " + " | ".join(s for s in stems if s != FULL) + " |", "|---|---|" + "---|" * (len(stems) - 1)]
    for l in a.layers:
        for row in a.rows:
            ref = feats[FULL][(l, row)][2]
            cells = []
            for src in stems:
                if src == FULL:
                    continue
                X = feats[src][(l, row)][2]
                cells.append(f"{((X - ref).norm() / ref.norm()).item():.1e}")
            lines.append(f"| {l} | {row} | " + " | ".join(cells) + " |")
    lines += ["", "Probe accuracy on the test set (train source -> test source) and, in brackets, agreement with the "
              f"predictions of the same probe on the {FULL} features:", ""]
    heads = [f"{FULL} -> {s}" for s in stems] + [f"{POOLED} -> {s}" for s in (FULL, POOLED)]
    lines += ["| Layer | row | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
    t0 = time.time()
    for l in a.layers:
        for row in a.rows:
            cells = []
            for train_src, test_srcs in ((FULL, list(stems)), (POOLED, [FULL, POOLED])):
                Xtr, ytr, _, _ = feats[train_src][(l, row)]
                probe = train_probe(Xtr, ytr, n_classes)
                ref_pred = predict(probe, feats[FULL][(l, row)][2])
                for test_src in test_srcs:
                    _, _, Xte, yte = feats[test_src][(l, row)]
                    pred = predict(probe, Xte)
                    acc = (pred == yte).double().mean().item()
                    agree = (pred == ref_pred).double().mean().item()
                    cells.append(f"{100 * acc:.1f}% ({100 * agree:.1f}%)" if test_src != FULL else f"{100 * acc:.1f}%")
            lines.append(f"| {l} | {row} | " + " | ".join(cells) + " |")
    print(f"probes trained and tested in {time.time() - t0:.0f} s")
    text = "\n".join(lines)
    print(text)
    dst = Path(__file__).resolve().parents[1] / "results" / "probes" / (re.sub(r"[^A-Za-z0-9.-]+", "-", a.model) + ".md")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text + "\n")
    print(f"written to {dst}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
