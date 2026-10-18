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
import rescomp as rc                                   # noqa: E402
from rescomp.hf import load_model                      # noqa: E402


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
    a = ap.parse_args()
    model, tok = load_model(a.model, dtype=a.dtype)
    dev = model.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    ids = torch.randint(2, model.model.config.vocab_size, (a.batch, a.seq_len), device=dev)
    run = lambda: model.model(ids, use_cache=False)
    with torch.no_grad():
        timed(run, 3, sync)                                           # warm-up
        clean = timed(run, a.iters, sync)
        with tempfile.TemporaryDirectory() as d:
            sink = rc.Sink(f"{d}/ov", rc.hf_layers(model.model, a.layers, tok), a.spec, buffer_bytes=1 << 40)
            with sink:
                timed(run, 3, sync)
                hooked = timed(run, a.iters, sync)
                t = time.perf_counter()
                sink.flush()
                flush = time.perf_counter() - t
                st = sink.stats()
    c, h = statistics.median(clean), statistics.median(hooked)
    print(f"{a.model} layers {a.layers} spec {a.spec} batch {a.batch}x{a.seq_len} on {dev}")
    print(f"clean forward   {c * 1000:8.1f} ms (median of {a.iters})")
    print(f"with sink hooks {h * 1000:8.1f} ms  -> overhead {100 * (h - c) / c:+.2f}%")
    print(f"flush (zstd + write, off the forward path): {flush:.2f} s for {st['chunks']} chunks, "
          f"{st['bytes_stored'] / 1e6:.1f} MB stored, ratio {st['ratio']:.2f}x")


if __name__ == "__main__":
    main()
