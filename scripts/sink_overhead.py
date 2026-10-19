"""How much forward time does a Sink add? Runs a model on fixed batches with and without the Sink attached.

    python scripts/sink_overhead.py --model gpt2 --layers 8 9 10 11 --batch 16 --seq-len 256 --spec int4_g384+layerK4+out12
    python scripts/sink_overhead.py --model Qwen/Qwen3-8B --layers 24 25 26 27 28 29 30 31 32 33 34 35 --batch 4 --seq-len 512

Reports the median per-forward time clean, with hooks (encode on device, copies started, no flush), and the
flush cost separately. The target is under 1% of the clean forward for the hooked pass."""
import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc  # noqa: E402
from rescomp.hf import load_model  # noqa: E402


def timed(fn, n, sync):
    out = []
    for _ in range(n):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--spec", default="int4_g384+layerK4+out12")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    model, tok = load_model(a.model, dtype=a.dtype)
    dev = model.get_input_embeddings().weight.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    ids = torch.randint(2, model.config.vocab_size, (a.batch, a.seq_len), device=dev)
    run = lambda: model(ids, use_cache=False)
    cuda = dev.type == "cuda"
    with torch.no_grad():
        timed(run, 3, sync)                                           # warm-up
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        clean = timed(run, a.iters, sync)
        peak_clean = torch.cuda.max_memory_allocated() if cuda else 0
        res = {}
        for fid in (False, True):
            with tempfile.TemporaryDirectory() as d:
                sink = rc.Sink(f"{d}/ov", rc.bind(model, a.layers, tokenizer=tok), a.spec, buffer_bytes=1 << 40, fidelity=fid,
                               compile=a.compile)
                with sink:
                    timed(run, 3, sync)
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    hooked = timed(run, a.iters, sync)
                    peak = torch.cuda.max_memory_allocated() if cuda else 0
                    t = time.perf_counter()
                    sink.flush()
                    res[fid] = (statistics.median(hooked), time.perf_counter() - t, sink.stats(), peak)
    c = statistics.median(clean)
    print(f"{a.model} layers {a.layers} spec {a.spec} batch {a.batch}x{a.seq_len} on {dev}{' (compiled quantizer)' if a.compile else ''}")
    print(f"clean forward            {c * 1000:8.1f} ms (median of {a.iters}); peak GPU memory {peak_clean / 2**30:.2f} GiB")
    for fid, (h, flush, st, peak) in res.items():
        print(f"with sink, fidelity={str(fid):5s} {h * 1000:8.1f} ms  -> overhead {100 * (h - c) / c:+.2f}%; "
              f"peak GPU memory {peak / 2**30:.2f} GiB ({(peak - peak_clean) / 2**20:+.0f} MiB); "
              f"flush {flush:.2f} s for {st['chunks']} chunks, {st['bytes_stored'] / 1e6:.1f} MB, ratio {st['ratio']:.2f}x")


if __name__ == "__main__":
    main()
