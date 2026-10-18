"""Markdown tables from results/<model>/results.jsonl files.

    python scripts/report.py results/gpt2 results/qwen3-1.7b ... > RESULTS_TABLES.md

Coherence statistics are recomputed from the saved generations (one averaged log-perplexity difference per
prompt, so seeds of the same prompt are not counted as independent observations)."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def fmt(x, f):
    return "-" if x is None else format(x, f)


def coherence_from_generations(path):
    """-> {ppl_ratio, t, n_prompts, rep4_codec, rep4_clean} from a generations_*.jsonl file."""
    recs = [json.loads(l) for l in open(path)]
    by_prompt = defaultdict(list)
    for r in recs:
        by_prompt[r["prompt_id"]].append(r["codec"]["nll"] - r["clean"]["nll"])
    d = [sum(v) / len(v) for v in by_prompt.values()]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else None
    se = math.sqrt(var / len(d)) if var else None
    return {"ppl_ratio": math.exp(mean), "t": (mean / se) if se else None, "n_prompts": len(d),
            "rep4_codec": sum(r["codec"]["rep4"] for r in recs) / len(recs),
            "rep4_clean": sum(r["clean"]["rep4"] for r in recs) / len(recs)}


def table(path):
    recs = [json.loads(l) for l in open(path)]
    if not recs:
        return ""
    r0 = recs[0]
    lines = [f"### {r0['model']}  (layers {r0['layers'][0]}-{r0['layers'][-1]}, {r0['n_seq']} seqs x {r0['seq_len']} tokens)", "",
             "| Spec | Ratio vs bf16 | rel L2 | KL (last layer) | ppl x (last layer) | KL (live) | ppl x (live) | top-1 | "
             "Coherence ppl x | t | rep4 codec/clean | Enc s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: r["ratio"]):
        fe, le = r["file_eval"], r.get("live_eval") or {}
        gen = path.parent / f"generations_{Path(r['file']).stem}.jsonl"
        co = coherence_from_generations(gen) if gen.exists() else {}
        lines.append("| `%s` | %.2f | %.1e | %.4f | %.4f | %s | %s | %.3f | %s | %s | %s | %s |" % (
            r["spec"], r["ratio"], r["fidelity"]["rel_l2"], fe["kl_mean"], fe["ppl_ratio"],
            fmt(le.get("kl_mean"), ".4f"), fmt(le.get("ppl_ratio"), ".4f"), fe["top1_agree"],
            fmt(co.get("ppl_ratio"), ".3f"), fmt(co.get("t"), ".1f"),
            f"{co['rep4_codec']:.3f}/{co['rep4_clean']:.3f}" if co else "-", fmt(r.get("encode_seconds"), ".0f")))
    return "\n".join(lines) + "\n"


def main():
    for d in sys.argv[1:]:
        p = Path(d) / "results.jsonl" if Path(d).is_dir() else Path(d)
        print(table(p))


if __name__ == "__main__":
    main()
