"""Where does a hooked layer spend its time? Profiles a few forwards with the Sink attached (CPU + CUDA), prints
the top operators by CUDA and CPU time, the host-side sync points, and a breakdown of one hooked layer.

    python scripts/sink_profile.py --model Qwen/Qwen3-8B --layers 24 25 26 27 28 29 30 31 32 33 34 35 --batch 4 --seq-len 512 --spec compact [--compile]
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc  # noqa: E402
from rescomp.hf import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--spec", default="compact")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    model, tok = load_model(a.model)
    dev = model.get_input_embeddings().weight.device
    ids = torch.randint(2, model.config.vocab_size, (a.batch, a.seq_len), device=dev)
    run = lambda: model(ids, use_cache=False)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with torch.no_grad(), tempfile.TemporaryDirectory() as d:
        sink = rc.Sink(f"{d}/p", rc.bind(model, a.layers, tokenizer=tok), rc.Codec(a.spec, compile=a.compile), buffer_bytes=1 << 40)
        with sink:
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
                for _ in range(a.iters):
                    run()
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total" if dev.type == "cuda" else "cpu_time_total", row_limit=25))
        # sync points: host-side events that wait on the device
        syncs = [e for e in prof.key_averages() if any(k in e.key for k in ("Synchronize", "cudaMemcpy", "cudaStreamSynchronize", "item", "_local_scalar_dense"))]
        print("--- sync-like events ---")
        for e in syncs:
            print(f"{e.key:50s} count {e.count:6d}  cpu total {e.cpu_time_total / 1000:8.2f} ms")
        # one hooked layer, pieces timed with events
        x = torch.randn(a.batch, a.seq_len, model.config.hidden_size, device=dev, dtype=torch.bfloat16)
        enc = sink.enc
        enc.reset()
        enc.encode_layer(0, x)                         # warm
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            enc.reset()
            enc.encode_layer(0, x)
        torch.cuda.synchronize()
        print(f"--- one keyframe layer, encode_layer end to end: {(time.perf_counter() - t) / 20 * 1000:.2f} ms "
              f"({a.batch}x{a.seq_len}x{model.config.hidden_size}, {sink.spec.quant_for(0)})")
        t = time.perf_counter()
        for _ in range(20):
            enc.reset()
            enc.encode_layer(0, x)
        print(f"    CPU side only (no sync): {(time.perf_counter() - t) / 20 * 1000:.2f} ms per call")
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
