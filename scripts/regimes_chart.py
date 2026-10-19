"""docs/regimes.png: compression ratio against live perplexity cost for the four regimes on every measured model.

    python scripts/regimes_chart.py          # reads results/regimes.json (written by the snippet in RESULTS.md)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ["exact", "faithful", "balanced", "compact"]
COLORS = {"exact": "#86b6ef", "faithful": "#3987e5", "balanced": "#1c5cab", "compact": "#0d366b"}   # one hue, ordinal
NAMES = {"gpt2": "GPT-2", "qwen3-1.7b": "Qwen3-1.7B", "gemma-2-2b": "Gemma-2-2B", "qwen3-8b": "Qwen3-8B",
         "qwen3-14b": "Qwen3-14B", "gemma-2-9b": "Gemma-2-9B", "olmoe-1b-7b": "OLMoE-1B-7B", "qwen3-30b-a3b": "Qwen3-30B-A3B"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1a1a19", "#5c5b55", "#e6e5e0"


def main():
    rows = json.load(open(ROOT / "results" / "regimes.json"))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for reg in REGIMES:
        pts = [r for r in rows if r["regime"] == reg]
        ax.scatter([r["ratio"] for r in pts], [100 * (r["ppl_live"] - 1) for r in pts], s=46, color=COLORS[reg],
                   edgecolor=SURFACE, linewidth=1.2, label=reg, zorder=3)
    # direct labels on the regime clusters (never a number on every point); a few model labels where it matters
    label = {"exact": (1.51, 0.0), "faithful": (2.58, 0.25), "balanced": (2.93, 1.45), "compact": (5.06, 5.3)}
    for reg, (x, y) in label.items():
        ax.annotate(reg, (x, y), xytext=(8, 10), textcoords="offset points", color=INK, fontsize=10, weight="bold")
    offsets = {"gpt2": (7, -3), "qwen3-1.7b": (7, -10), "gemma-2-2b": (7, 4), "gemma-2-9b": (7, -3)}
    for r in rows:
        if r["regime"] == "compact" and r["model"] in offsets:
            ax.annotate(NAMES[r["model"]], (r["ratio"], 100 * (r["ppl_live"] - 1)), xytext=offsets[r["model"]],
                        textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("compression ratio vs bf16 (×)", color=INK)
    ax.set_ylabel("perplexity increase, live codec (%)", color=INK)
    ax.set_title("Four regimes on eight models (late third of layers, wikitext)", color=INK, fontsize=11, loc="left")
    ax.set_xlim(1.2, 5.8)
    ax.set_ylim(-0.8, 8)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(title="regime (one point per model)", loc="upper left", frameon=False, fontsize=9, title_fontsize=9)
    leg.get_title().set_color(INK2)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out = ROOT / "docs" / "regimes.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
