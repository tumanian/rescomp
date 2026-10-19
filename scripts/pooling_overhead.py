"""What does a TransformSink cost, and how small is what it stores? Runs a model on fixed batches clean and with
the TransformSink attached (rows accumulate on the device, nothing is written during the timed passes).

    python scripts/pooling_overhead.py --model gpt2 --layers 8 9 10 11 --batch 16 --seq-len 256
    python scripts/pooling_overhead.py --model Qwen/Qwen3-8B --layers 24 25 26 27 28 29 30 31 32 33 34 35 --batch 4 --seq-len 512

Reports, per codec: the median forward time and its overhead over the clean forward, the peak GPU memory added
(the accumulated rows included) and the flush time; then the stored bytes against the raw bf16 size of the same
sequences' full activations and against a full Sink capture of a few of the batches (exact and compact)."""
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
    ap.add_argument("--codecs", nargs="*", default=["exact", "fp32"])
    ap.add_argument("--transform", default="mean+last")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--full-iters", type=int, default=2, help="batches to store whole with a Sink, for the byte comparison")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    model, tok = load_model(a.model, dtype=a.dtype)
    dev = model.get_input_embeddings().weight.device
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    g = torch.Generator().manual_seed(0)
    n_real = torch.randint(a.seq_len // 2, a.seq_len + 1, (a.batch,), generator=g)       # right padding, as a loader would
    mask = (torch.arange(a.seq_len)[None] < n_real[:, None]).long().to(dev)
    batches = [torch.randint(2, model.config.vocab_size, (a.batch, a.seq_len), generator=g).to(dev) for _ in range(a.iters + 3)]
    n_run = [0]                                                                               # every call sees different tokens: no repeated rows for zstd to find

    def run():
        ids = batches[n_run[0] % len(batches)]
        n_run[0] += 1
        return model(ids, attention_mask=mask, use_cache=False)
    L, d = len(a.layers), model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    raw_full = a.batch * a.seq_len * d * 2 * L                                                 # bf16 bytes of one batch's stored layers
    print(f"{a.model} layers {a.layers} transform {a.transform} batch {a.batch}x{a.seq_len} (real tokens {n_real.min()}..{n_real.max()}) on {dev}")
    with torch.no_grad():
        timed(run, 3, sync)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        clean = timed(run, a.iters, sync)
        peak_clean = torch.cuda.max_memory_allocated() if cuda else 0
        c = statistics.median(clean)
        print(f"clean forward            {c * 1000:8.1f} ms (median of {a.iters}); peak GPU memory {peak_clean / 2**30:.2f} GiB")
        with tempfile.TemporaryDirectory() as tmp:
            per_seq = {}
            for codec in a.codecs:
                sink = rc.TransformSink(f"{tmp}/pool-{codec}", rc.bind(model, a.layers, tokenizer=tok), codec,
                                        transform=a.transform, chunk_seqs=1 << 30, buffer_bytes=1 << 40)
                with sink:
                    timed(run, 3, sync)
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    hooked = timed(run, a.iters, sync)
                    peak = torch.cuda.max_memory_allocated() if cuda else 0
                    pending = sink.stats()["pending_sequences"]
                    acc = pending * sink.transform.k * d * 4 * L
                    t = time.perf_counter()
                    sink.flush()
                    flush = time.perf_counter() - t
                    st = sink.stats()
                h = statistics.median(hooked)
                n_seq = st["sequences"]
                print(f"TransformSink, codec {codec:5s} {h * 1000:8.1f} ms  -> overhead {100 * (h - c) / c:+.2f}% ({(h - c) * 1000 / L:.2f} ms per hooked layer); "
                      f"peak GPU memory {peak / 2**30:.2f} GiB ({(peak - peak_clean) / 2**20:+.0f} MiB, of which {acc / 2**20:.0f} MiB are the "
                      f"{pending} sequences' rows); flush {flush:.2f} s: {n_seq} sequences in {st['chunks']} chunk(s), {st['bytes_stored'] / 1e6:.2f} MB")
                per_seq[codec] = st["bytes_stored"] / n_seq
                print(f"    stored {per_seq[codec] / L:.0f} bytes per sequence per layer ({per_seq[codec]:.0f} per sequence); vs the raw bf16 rows {st['ratio']:.2f}x; "
                      f"vs the raw bf16 full activations of the same sequences {raw_full * (n_seq / a.batch) / st['bytes_stored']:.0f}x")
            # a full Sink capture of a few batches, exact and compact, for the on-disk comparison
            for regime in ("exact", "compact"):
                with rc.Sink(f"{tmp}/full-{regime}", rc.bind(model, a.layers, tokenizer=tok), regime, buffer_bytes=1 << 40) as sink:
                    timed(run, a.full_iters, sync)
                st = sink.stats()
                print(f"full Sink capture, {regime:7s}: {st['bytes_stored'] / st['sequences']:.0f} bytes per sequence ({st['ratio']:.2f}x vs raw bf16)"
                      + "; the pooled rows are " + ", ".join(f"{st['bytes_stored'] / st['sequences'] / v:.0f}x smaller ({k})" for k, v in per_seq.items()))


if __name__ == "__main__":
    main()
