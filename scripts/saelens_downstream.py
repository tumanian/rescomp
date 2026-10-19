"""Downstream check of rescomp against SAELens: does a TopK SAE trained on rescomp-stored residual activations
match one trained on exact float32 activations, when both are evaluated on true activations?

    python scripts/saelens_downstream.py --out results/saelens_gpt2 --arms faithful balanced compact --seeds 0 1 2
    python scripts/saelens_downstream.py --out /tmp/smoke --smoke          # plumbing only, CPU, a minute

Arms: `baseline` (float32 activations streamed live from the model, what SAELens' own float32 cache holds) and one
arm per rescomp regime or spec (captured once with rc.Sink, read back with rc.open(stem).read(L, seqs=...)).
Within a seed every arm gets the same SAE init and the same data order; only the activation values differ.
Evaluation always uses true float32 activations of held-out sequences, BOS excluded. Outputs in --out:
config.json, tokens.pt, captures/<arm>/, saes/<arm>_s<seed>.pt, results.jsonl (one line per arm and seed;
finished pairs are skipped on a re-run), summary.md.

The pass rule is fixed before running (needs >= 2 seeds). A codec passes if
    |mean dCE(arm) - mean dCE(baseline)|  <= max(baseline seed range, 0.005 nats)
    |mean FVE(arm) - mean FVE(baseline)|  <= max(baseline seed range, 0.002)
    cross-seed feature match(arm)         >= cross-seed match(baseline) - 0.01
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rescomp as rc  # noqa: E402

DCE_FLOOR, FVE_FLOOR, MATCH_SLACK = 0.005, 0.002, 0.01     # the pass rule's fixed tolerances
PROVIDER_SEED_OFFSET = 10_007                               # derived seed for the data order: seed + this


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- tokens

def dataset_token_stream(name, column, bos_id):
    """Token ids of a pretokenized Hub dataset, row after row, leading BOS rows stripped (we add our own). Reads
    the parquet shards directly through the Hub filesystem: fast to start and no full download."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{name}/data/train-*.parquet")) or \
        sorted(p for p in fs.glob(f"datasets/{name}/**/*.parquet") if "train" in os.path.basename(p))
    if not shards:
        raise ValueError(f"no train parquet shards found in datasets/{name}")
    for shard in shards:
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                for row in pf.read_row_group(g, columns=[column]).column(column).to_pylist():
                    if row and row[0] == bos_id:
                        row = row[1:]
                    yield row


def draw_tokens(a, bos_id, d_vocab, out_path):
    """[N, context] sequences, BOS at position 0 and context-1 real tokens after it; train first, then eval."""
    body = a.context - 1
    n_train = math.ceil(a.train_tokens / body)
    n_eval = math.ceil(a.eval_tokens / body)
    n = n_train + n_eval
    if a.smoke:
        g = torch.Generator().manual_seed(0)
        flat = torch.randint(1, d_vocab, (n * body,), generator=g)
    else:
        log(f"drawing {n * body:,} tokens from {a.dataset} ({n_train:,} train + {n_eval:,} eval sequences of {a.context})")
        parts, have = [], 0
        for row in dataset_token_stream(a.dataset, a.dataset_column, bos_id):
            parts.append(torch.tensor(row, dtype=torch.long))
            have += len(row)
            if have >= n * body:
                break
        flat = torch.cat(parts)[: n * body]
    seqs = torch.cat([torch.full((n, 1), bos_id, dtype=torch.long), flat.view(n, body)], 1)
    rec = {"train": seqs[:n_train].contiguous(), "eval": seqs[n_train:].contiguous(), "bos_id": bos_id,
           "context": a.context, "dataset": None if a.smoke else a.dataset}
    torch.save(rec, out_path)
    return rec


# ---------------------------------------------------------------- model

def load_model(a, device):
    if a.smoke:
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        torch.manual_seed(1234)
        cfg = HookedTransformerConfig(n_layers=4, d_model=64, d_head=16, n_heads=4, d_mlp=256, d_vocab=512, n_ctx=a.context,
                                      act_fn="gelu", normalization_type="LN", device=device)
        return HookedTransformer(cfg).eval()
    from sae_lens.load_model import load_model as sae_lens_load_model  # what SAELens' own runner loads with
    return sae_lens_load_model("HookedTransformer", a.model, device=device).eval()


def live_acts(model, toks, hook, layer):
    """True float32 activations at `hook` for one batch of token ids: [B, T, d] on the model's device."""
    _, cache = model.run_with_cache(toks, names_filter=hook, stop_at_layer=layer + 1, return_type=None)
    return cache[hook].float()


def batches(seqs, size):
    for i in range(0, seqs.shape[0], size):
        yield i, seqs[i: i + size]


# ---------------------------------------------------------------- captures

def capture_layers_for(spec, layer, override):
    """Worst case for layer deltas: capture [L-K+1 .. L] so the target is the last delta before a keyframe."""
    if override:
        if layer not in override:
            raise ValueError(f"--capture-layers {override} must contain the target layer {layer}")
        return sorted(override)
    if spec.K:
        return list(range(max(0, layer - spec.K + 1), layer + 1))
    return [layer]


def capture(a, arm, model, train, layer, hook, device, out):
    """rc.Sink over the train sequences; returns the capture record (alignment checks included)."""
    cdir = out / "captures" / arm
    stem = str(cdir / arm)
    marker = cdir / "capture.json"
    if marker.exists():
        return json.loads(marker.read_text())
    d = model.cfg.d_model
    spec = rc.resolve_spec(arm, d)
    layers = capture_layers_for(spec, layer, a.capture_layers)
    codec = rc.Codec(spec.name, level=a.level)
    log(f"[{arm}] capturing {train.shape[0]:,} sequences at layers {layers} with {spec.name}")
    t0 = time.time()
    sink = rc.Sink(stem, rc.bind(model, layers, model_id=a.model), codec, fidelity=True, overwrite=True,
                   extra={"experiment": "saelens_downstream", "arm": arm, "target_layer": layer})
    with sink, torch.no_grad():
        for i, toks in batches(train, a.forward_batch):
            model(toks.to(device), stop_at_layer=layer + 1, return_type=None)
            if (i // a.forward_batch) % 200 == 0:
                log(f"[{arm}]   {i:,}/{train.shape[0]:,} sequences")
    seconds = time.time() - t0
    rec = alignment_checks(a, arm, model, train, layer, hook, device, stem, layers)
    rec.update(spec=spec.name, layers=layers, capture_seconds=round(seconds, 1), forward_batch=a.forward_batch)
    marker.write_text(json.dumps(rec, indent=1))
    log(f"[{arm}] stored {rec['bytes_per_token_per_layer']:.1f} B/token/layer, {rec['ratio_vs_fp32']:.2f}x vs fp32 "
        f"({rec['ratio_vs_bf16']:.2f}x vs bf16) over layers {layers}; target layer {layer} is a "
        f"{'keyframe' if rec['target_is_keyframe'] else 'delta'}; rel L2 at target {rec['rel_l2_target_sample']:.3e} "
        f"(header {rec['rel_l2_target_header']:.3e}); {seconds:.0f}s")
    return rec


def alignment_checks(a, arm, model, train, layer, hook, device, stem, layers):
    """Reader tokens equal the input tokens (first and last sequence); rel L2 between stored and live activations
    at the target layer on a sample; bytes per token per layer and the ratios from the header."""
    with rc.open(stem) as r:
        h = r.header
        n = train.shape[0]
        if r.n_seq != n:
            raise AssertionError(f"[{arm}] reader holds {r.n_seq} sequences, expected {n}")
        for s in (0, n - 1):
            if not torch.equal(r.tokens(s), train[s]):
                raise AssertionError(f"[{arm}] stored tokens of sequence {s} differ from the input tokens")
        n_sample = min(n, 4 * a.forward_batch)
        se = ss = 0.0
        with torch.no_grad():
            for i, toks in batches(train[:n_sample], a.forward_batch):
                live = live_acts(model, toks.to(device), hook, layer)[:, 1:].cpu()          # BOS excluded
                stored = torch.stack(r.read(layer, seqs=list(range(i, i + toks.shape[0]))))[:, 1:]
                se += (live - stored).pow(2).sum().item()
                ss += live.pow(2).sum().item()
        idx = r.layers.index(layer)
        lrec = h["layers"][idx]
        b = h["bytes"]
        n_tok, n_layers, d = h["n_tokens"], len(r.layers), r.d
        stored_bytes = b["activations_stored"]
        return {"n_sequences": n, "n_tokens": n_tok, "stored_bytes": stored_bytes,
                "bytes_per_token_per_layer": stored_bytes / (n_tok * n_layers),
                "ratio_vs_fp32": n_tok * n_layers * d * 4 / stored_bytes, "ratio_vs_bf16": b["ratio_vs_bf16"],
                "target_is_keyframe": bool(lrec["keyframe"]), "target_quant": lrec["quant"],
                "rel_l2_target_header": lrec["fidelity"]["rel_l2"], "rel_l2_all_layers_header": h["fidelity"]["rel_l2"],
                "rel_l2_target_sample": math.sqrt(se / ss) if ss else 0.0, "rel_l2_sample_sequences": n_sample,
                "header_spec": h["spec"], "header_regime": h.get("regime")}


# ---------------------------------------------------------------- data providers

@torch.no_grad()          # as a decorator torch handles generators; a `with no_grad()` around a yield would leak grad mode on close
def baseline_chunks(model, train, layer, hook, device, forward_batch):
    """One pass: [B*(T-1), d] float32 activations per forward batch, BOS position dropped."""
    for _, toks in batches(train, forward_batch):
        x = live_acts(model, toks.to(device), hook, layer)[:, 1:]
        yield x.reshape(-1, x.shape[-1])


@torch.no_grad()
def reader_chunks(stem, layer, n, forward_batch, device):
    """One pass over a capture, in the chunks it was written in (each chunk decodes once)."""
    with rc.open(stem) as r:
        for i in range(0, n, forward_batch):
            ids = list(range(i, min(n, i + forward_batch)))
            x = torch.stack(r.read(layer, seqs=ids))[:, 1:]
            yield x.reshape(-1, x.shape[-1]).to(device)


def endless(make_chunks, total, label):
    """Loop over the chunk source for ever, logging progress against the sample budget."""
    served, next_mark, pass_no = 0, 0.1, 0
    while True:
        pass_no += 1
        for x in make_chunks():
            served += x.shape[0]
            if served >= next_mark * total:
                log(f"[{label}]   produced {served:,} samples ({100 * served / total:.0f}% of the budget, pass {pass_no})")
                next_mark += 0.1
            yield x


def own_rng(seed, make_iter):
    """Run an iterator under its own CPU RNG stream (seeded once), so the shuffles inside SAELens' mixing_buffer
    come out the same for every arm whatever else consumes the global RNG between draws."""
    outer = torch.random.get_rng_state()
    torch.manual_seed(seed)
    inner = torch.random.get_rng_state()
    torch.random.set_rng_state(outer)
    it = make_iter()
    while True:
        outer = torch.random.get_rng_state()
        torch.random.set_rng_state(inner)
        try:
            x = next(it)
        finally:
            inner = torch.random.get_rng_state()
            torch.random.set_rng_state(outer)
        yield x


def make_provider(a, arm, seed, model, train, layer, hook, device, out, total_samples):
    from sae_lens.training.mixing_buffer import mixing_buffer
    if arm == "baseline":
        chunks = lambda: baseline_chunks(model, train, layer, hook, device, a.forward_batch)
    else:
        stem = str(out / "captures" / arm / arm)
        chunks = lambda: reader_chunks(stem, layer, train.shape[0], a.forward_batch, device)
    src = endless(chunks, total_samples, f"{arm} s{seed}")
    return own_rng(seed + PROVIDER_SEED_OFFSET, lambda: mixing_buffer(a.buffer_tokens, a.batch_tokens, src))


# ---------------------------------------------------------------- training

def sae_path(out, arm, seed):
    return out / "saes" / f"{arm}_s{seed}.pt"


def build_sae(a, d, device):
    from sae_lens import TopKTrainingSAE, TopKTrainingSAEConfig
    cfg = TopKTrainingSAEConfig(d_in=d, d_sae=a.d_sae_mult * d, k=a.k, device=device, dtype="float32")
    return TopKTrainingSAE(cfg)


def train_sae(a, arm, seed, model, train, layer, hook, device, out):
    from sae_lens import LoggingConfig, SAETrainer
    from sae_lens.config import SAETrainerConfig
    d = model.cfg.d_model
    total = train.shape[0] * (a.context - 1)
    torch.manual_seed(seed)
    sae = build_sae(a, d, device)
    provider = make_provider(a, arm, seed, model, train, layer, hook, device, out, total)
    cfg = SAETrainerConfig(total_training_samples=total, train_batch_size_samples=a.batch_tokens, lr=a.lr, lr_end=a.lr / 10,
                           device=device, logger=LoggingConfig(log_to_wandb=False))
    log(f"[{arm} s{seed}] training TopK SAE d_sae={a.d_sae_mult * d} k={a.k} for {cfg.total_training_steps} steps of {a.batch_tokens}")
    t0 = time.time()
    sae = SAETrainer(cfg, sae, provider).fit()
    provider.close()
    seconds = time.time() - t0
    p = sae_path(out, arm, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in sae.state_dict().items()}, "cfg": sae.cfg.to_dict(), "arm": arm, "seed": seed,
                "train_seconds": seconds, "train_l0": sae.cfg.metadata.l0, "train_dead": sae.cfg.metadata.num_dead_features}, p)
    log(f"[{arm} s{seed}] trained in {seconds:.0f}s; last-window L0 {sae.cfg.metadata.l0}, dead {sae.cfg.metadata.num_dead_features}")
    return sae, {"train_seconds": round(seconds, 1)}


def load_sae(a, p, d, device):
    rec = torch.load(p, map_location="cpu")
    sae = build_sae(a, d, device)
    sae.load_state_dict(rec["state_dict"])
    return sae.to(device), rec


# ---------------------------------------------------------------- evaluation

def ce_and_logprobs(logits, toks):
    """Per-position cross-entropy and log-probs for predictions made from non-BOS positions (1 .. T-2)."""
    lp = torch.log_softmax(logits[:, 1:-1].float(), -1)
    tgt = toks[:, 2:]
    ce = -lp.gather(-1, tgt[..., None])[..., 0]
    return ce, lp


@torch.no_grad()
def evaluate(a, sae, model, ev, layer, hook, device):
    """On TRUE float32 activations of the held-out sequences, BOS excluded."""
    sae.eval()
    d_sae = sae.cfg.d_sae
    n = sse = sum_sq = l0 = 0.0
    sum_x = torch.zeros(model.cfg.d_model, dtype=torch.float64, device=device)
    fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    ce_clean = ce_spl = ce_zero = kl = 0.0
    n_pred = 0
    for _, toks in batches(ev, a.eval_batch):
        toks = toks.to(device)
        logits, cache = model.run_with_cache(toks, names_filter=hook)
        x = cache[hook].float()[:, 1:]                                  # [B, T-1, d], BOS excluded
        flat = x.reshape(-1, x.shape[-1])
        feats = sae.encode(flat)
        recon = sae.decode(feats)
        sse += (recon - flat).pow(2).sum().item()
        sum_sq += flat.double().pow(2).sum().item()
        sum_x += flat.double().sum(0)
        n += flat.shape[0]
        l0 += (feats != 0).sum().item()
        fired |= (feats != 0).any(0)
        recon = recon.view_as(x)

        def splice(act, hook, rep=recon):                               # TL calls hook_fn(act, hook=HookPoint)
            act = act.clone()
            act[:, 1:] = rep.to(act.dtype)
            return act

        def zero(act, hook):
            act = act.clone()
            act[:, 1:] = 0
            return act

        lg_spl = model.run_with_hooks(toks, fwd_hooks=[(hook, splice)])
        lg_zero = model.run_with_hooks(toks, fwd_hooks=[(hook, zero)])
        c0, lp0 = ce_and_logprobs(logits, toks)
        c1, lp1 = ce_and_logprobs(lg_spl, toks)
        c2, _ = ce_and_logprobs(lg_zero, toks)
        ce_clean += c0.sum().item()
        ce_spl += c1.sum().item()
        ce_zero += c2.sum().item()
        kl += (lp0.exp() * (lp0 - lp1)).sum(-1).sum().item()
        n_pred += c0.numel()
    mean_x = sum_x / n
    total_var = sum_sq - n * mean_x.pow(2).sum().item()
    ce_clean, ce_spl, ce_zero, kl = ce_clean / n_pred, ce_spl / n_pred, ce_zero / n_pred, kl / n_pred
    return {"fve": 1 - sse / total_var, "mse": sse / n, "l0": l0 / n, "dead_frac": 1 - fired.float().mean().item(),
            "ce_clean": ce_clean, "ce_spliced": ce_spl, "ce_zero": ce_zero, "delta_ce": ce_spl - ce_clean,
            "ce_recovered": (ce_zero - ce_spl) / (ce_zero - ce_clean) if ce_zero != ce_clean else float("nan"),
            "kl_clean_spliced": kl, "eval_tokens": int(n)}


@torch.no_grad()
def mean_max_cos(wa, wb, block=2048):
    """Mean over rows of A of the max cosine against rows of B (decoder rows, unit-normalised)."""
    a = torch.nn.functional.normalize(wa.float(), dim=-1)
    b = torch.nn.functional.normalize(wb.float(), dim=-1)
    best = []
    for i in range(0, a.shape[0], block):
        best.append((a[i: i + block] @ b.T).max(-1).values)
    return torch.cat(best).mean().item()


def feature_agreement(arms, seeds, out, device):
    """For each arm and each pair of distinct seeds (s, s'): arm seed s decoder rows against baseline seed s'."""
    dec = {}
    for arm in arms:
        for s in seeds:
            dec[(arm, s)] = torch.load(sae_path(out, arm, s), map_location="cpu")["state_dict"]["W_dec"].to(device)
    agree = {}
    for arm in arms:
        vals = [mean_max_cos(dec[(arm, s)], dec[("baseline", t)]) for s in seeds for t in seeds if s != t]
        agree[arm] = sum(vals) / len(vals)
    return agree


# ---------------------------------------------------------------- results

def load_results(path):
    done = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["seed"])] = r
    return done


def mean_range(xs):
    return sum(xs) / len(xs), max(xs) - min(xs)


def verdicts(results, agree, arms, seeds):
    base = [results[("baseline", s)] for s in seeds]
    b_dce, b_dce_r = mean_range([r["delta_ce"] for r in base])
    b_fve, b_fve_r = mean_range([r["fve"] for r in base])
    out = {}
    for arm in arms:
        if arm == "baseline":
            continue
        rs = [results[(arm, s)] for s in seeds]
        dce = mean_range([r["delta_ce"] for r in rs])[0]
        fve = mean_range([r["fve"] for r in rs])[0]
        checks = {"delta_ce": (abs(dce - b_dce), max(b_dce_r, DCE_FLOOR)),
                  "fve": (abs(fve - b_fve), max(b_fve_r, FVE_FLOOR)),
                  "feature_match": (agree[arm], agree["baseline"] - MATCH_SLACK)}
        ok = checks["delta_ce"][0] <= checks["delta_ce"][1] and checks["fve"][0] <= checks["fve"][1] \
            and checks["feature_match"][0] >= checks["feature_match"][1]
        out[arm] = {"pass": ok, "checks": checks}
    return out


def write_summary(a, out, results, caps, agree, verd, arms, seeds, model_desc):
    metrics = [("fve", "FVE", ".4f"), ("mse", "MSE", ".3f"), ("l0", "L0", ".2f"), ("dead_frac", "dead frac", ".4f"),
               ("ce_clean", "CE clean", ".4f"), ("ce_spliced", "CE spliced", ".4f"), ("ce_zero", "CE zero-abl", ".3f"),
               ("delta_ce", "delta CE", ".4f"), ("ce_recovered", "CE recovered", ".4f"), ("kl_clean_spliced", "KL(clean, spliced)", ".4f")]
    lines = [f"# rescomp x SAELens downstream check: {model_desc}", "",
             f"*{time.strftime('%Y-%m-%d %H:%M')}. Hook `{a.hook}`; {results[('baseline', seeds[0])]['train_tokens']:,} training tokens "
             f"({a.context}-token contexts, BOS at 0, never trained on); TopK SAE d_sae = {a.d_sae_mult} x d_model, k = {a.k}, "
             f"lr {a.lr}, batch {a.batch_tokens}; seeds {seeds}. Every number below is measured on true float32 activations of "
             f"{results[('baseline', seeds[0])]['eval_tokens']:,} held-out tokens (BOS excluded). Cells are mean (range) over seeds.*", ""]
    head = "| arm | spec | layers | target | B/tok/layer | ratio vs fp32 | rel L2 (target) | " + " | ".join(m[1] for m in metrics) + " | cross-seed match | verdict |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for arm in arms:
        rs = [results[(arm, s)] for s in seeds]
        c = caps.get(arm)
        cells = [arm, "fp32 live" if c is None else f"`{c['spec']}`", "-" if c is None else str(c["layers"]),
                 "-" if c is None else ("keyframe" if c["target_is_keyframe"] else "delta"),
                 "4 x d" if c is None else f"{c['bytes_per_token_per_layer']:.1f}",
                 "1.00" if c is None else f"{c['ratio_vs_fp32']:.2f}", "0" if c is None else f"{c['rel_l2_target_sample']:.2e}"]
        for key, _, fmt in metrics:
            m, r = mean_range([x[key] for x in rs])
            cells.append(f"{m:{fmt}} ({r:{fmt}})")
        cells.append(f"{agree[arm]:.4f}" + (" (noise floor)" if arm == "baseline" else ""))
        cells.append("-" if arm == "baseline" else ("PASS" if verd[arm]["pass"] else "FAIL"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Ratios and bytes per token cover the whole captured layer stack (keyframes included for `balanced` and `compact`),",
              "not the target layer alone. rel L2 is ||x - xhat|| / ||x|| at the target layer over the first "
              f"{next((c['rel_l2_sample_sequences'] for c in caps.values()), 0)} training sequences, BOS excluded (the sink's own "
              "figure in the capture header counts the BOS position too, whose norm is far larger than any other token's, so it reads "
              "lower: " + ", ".join(f"{arm} {c['rel_l2_target_header']:.2e}" for arm, c in caps.items()) + ").", "",
              "## Pass rule (fixed before running)", "",
              f"- |mean delta CE(arm) - mean delta CE(baseline)| <= max(baseline seed range, {DCE_FLOOR} nats)",
              f"- |mean FVE(arm) - mean FVE(baseline)| <= max(baseline seed range, {FVE_FLOOR})",
              f"- cross-seed feature match(arm) >= cross-seed match(baseline) - {MATCH_SLACK}", "",
              "Cross-seed feature match: for each pair of distinct seeds (s, s'), the mean over arm-seed-s decoder rows of the max cosine "
              "against baseline-seed-s' decoder rows, averaged over the pairs; the baseline row is the noise floor (baseline s vs baseline s').", "",
              "| arm | delta CE gap (tolerance) | FVE gap (tolerance) | feature match (floor) | verdict |", "|---|---|---|---|---|"]
    for arm, v in verd.items():
        c = v["checks"]
        lines.append(f"| {arm} | {c['delta_ce'][0]:.4f} ({c['delta_ce'][1]:.4f}) | {c['fve'][0]:.4f} ({c['fve'][1]:.4f}) | "
                     f"{c['feature_match'][0]:.4f} ({c['feature_match'][1]:.4f}) | {'PASS' if v['pass'] else 'FAIL'} |")
    lines += ["", "## Per seed", "", "| arm | seed | " + " | ".join(m[1] for m in metrics) + " | train s |",
              "|" + "---|" * (len(metrics) + 3)]
    for arm in arms:
        for s in seeds:
            r = results[(arm, s)]
            lines.append(f"| {arm} | {s} | " + " | ".join(f"{r[k]:{fmt}}" for k, _, fmt in metrics) + f" | {r.get('train_seconds', 0):.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=7, help="target layer: hook blocks.<layer>.hook_resid_post")
    ap.add_argument("--dataset", default="apollo-research/Skylion007-openwebtext-tokenizer-gpt2")
    ap.add_argument("--dataset-column", default="input_ids")
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--train-tokens", type=int, default=20_000_000, help="trainable (non-BOS) tokens")
    ap.add_argument("--eval-tokens", type=int, default=500_000)
    ap.add_argument("--arms", nargs="*", default=["faithful", "balanced", "compact"], help="rescomp regimes or spec strings")
    ap.add_argument("--capture-layers", type=int, nargs="*", default=None, help="override the captured layer stack")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--d-sae-mult", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--buffer-tokens", type=int, default=262_144, help="mixing buffer size (samples)")
    ap.add_argument("--forward-batch", type=int, default=128, help="sequences per model forward (= one rescomp chunk)")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--level", type=int, default=3, help="zstd level for the captures")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny random model and tokens on the CPU: plumbing only")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.layer, a.train_tokens, a.eval_tokens = "smoke-4L-64d", 3, 40_000, 4_000
        a.seeds, a.buffer_tokens, a.forward_batch, a.eval_batch, a.device = [0, 1], 8192, 16, 16, "cpu"
    if len(a.seeds) < 2:
        ap.error("the pass rule needs at least 2 seeds")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    a.hook = f"blocks.{a.layer}.hook_resid_post"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = ["baseline"] + [x for x in a.arms if x != "baseline"]
    (out / "config.json").write_text(json.dumps(dict(vars(a), device=device, arms_run=arms, pass_rule={
        "delta_ce_floor_nats": DCE_FLOOR, "fve_floor": FVE_FLOOR, "feature_match_slack": MATCH_SLACK}), indent=1))
    log(f"device {device}; out {out}; arms {arms}; seeds {a.seeds}")

    model = load_model(a, device)
    d = model.cfg.d_model
    bos_id = 0 if a.smoke else model.tokenizer.bos_token_id
    tok_path = out / "tokens.pt"
    toks = torch.load(tok_path) if tok_path.exists() else draw_tokens(a, bos_id, model.cfg.d_vocab, tok_path)
    train, ev = toks["train"], toks["eval"]
    if toks["context"] != a.context or int(train[0, 0]) != bos_id:
        raise ValueError(f"{tok_path} was drawn with other settings; remove it or use a new --out")
    train_tokens = train.shape[0] * (a.context - 1)
    log(f"model d_model={d} n_layers={model.cfg.n_layers}; {train.shape[0]:,} train sequences ({train_tokens:,} trainable tokens), "
        f"{ev.shape[0]:,} eval sequences; hook {a.hook}")

    caps = {arm: capture(a, arm, model, train, a.layer, a.hook, device, out) for arm in arms if arm != "baseline"}

    res_path = out / "results.jsonl"
    results = load_results(res_path)
    for seed in a.seeds:
        for arm in arms:
            if (arm, seed) in results:
                log(f"[{arm} s{seed}] done already, skipping")
                continue
            p = sae_path(out, arm, seed)
            if p.exists():
                log(f"[{arm} s{seed}] SAE weights found, evaluating only")
                sae, saved = load_sae(a, p, d, device)
                extra = {"train_seconds": saved.get("train_seconds")}
            else:
                sae, extra = train_sae(a, arm, seed, model, train, a.layer, a.hook, device, out)
            t0 = time.time()
            m = evaluate(a, sae, model, ev, a.layer, a.hook, device)
            rec = {"arm": arm, "seed": seed, "model": a.model, "hook": a.hook, "train_tokens": train_tokens, **m, **extra,
                   "eval_seconds": round(time.time() - t0, 1), "capture": caps.get(arm)}
            results[(arm, seed)] = rec
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"[{arm} s{seed}] FVE {m['fve']:.4f} L0 {m['l0']:.1f} dead {m['dead_frac']:.4f} CE {m['ce_clean']:.4f} -> "
                f"{m['ce_spliced']:.4f} (zero {m['ce_zero']:.3f}) dCE {m['delta_ce']:.4f} recovered {m['ce_recovered']:.4f} KL {m['kl_clean_spliced']:.4f}")
            del sae
            if device == "cuda":
                torch.cuda.empty_cache()

    agree = feature_agreement(arms, a.seeds, out, device)
    verd = verdicts(results, agree, arms, a.seeds)
    desc = f"{a.model} ({d}d, {model.cfg.n_layers} layers)"
    write_summary(a, out, results, caps, agree, verd, arms, a.seeds, desc)
    log("cross-seed feature match: " + ", ".join(f"{k} {v:.4f}" for k, v in agree.items()))
    for arm, v in verd.items():
        log(f"{arm}: {'PASS' if v['pass'] else 'FAIL'} " + json.dumps({k: [round(x, 5) for x in c] for k, c in v["checks"].items()}))
    log(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
