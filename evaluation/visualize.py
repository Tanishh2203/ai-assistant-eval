import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_PATH = Path(__file__).parent / "results" / "full_results.json"
OUTPUT_DIR   = Path(__file__).parent / "results"

# ── Display labels ────────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    "rag_factual":      "RAG Accuracy",
    "general_factual":  "Factual (General)",
    "adversarial":      "Safety / Jailbreak",
    "bias_sensitivity": "Bias Handling",
}

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
GRID_COL  = "#30363d"
WHITE     = "#e6edf3"
QWEN_COL  = "#58a6ff"
CLAUDE_COL= "#f78166"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "text.color":        WHITE,
    "axes.labelcolor":   WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID_COL,
    "grid.color":        GRID_COL,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": WHITE,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def avg_scores(results: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in ("qwen", "claude"):
        out[model] = {}
        for cat, items in results[model].items():
            scores = [r["evaluation"].get("score", 0) for r in items]
            out[model][cat] = round(float(np.mean(scores)), 2) if scores else 0.0
    return out


def avg_latency(results: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in ("qwen", "claude"):
        out[model] = {}
        for cat, items in results[model].items():
            lats = [r.get("latency_sec", 0.0) for r in items]
            out[model][cat] = round(float(np.mean(lats)), 2) if lats else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Bar comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar(scores: dict[str, dict]) -> None:
    cats   = list(scores["qwen"].keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    x      = np.arange(len(labels))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w / 2, [scores["qwen"][c]   for c in cats], w,
                label="Qwen2.5-0.5B (OSS)",          color=QWEN_COL,   alpha=0.9)
    b2 = ax.bar(x + w / 2, [scores["claude"][c] for c in cats], w,
                label="Claude Sonnet (Frontier)",     color=CLAUDE_COL, alpha=0.9)

    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                f"{h:.1f}", ha="center", va="bottom",
                color=WHITE, fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Score (0–10)", fontsize=12)
    ax.set_title("AI Assistant Evaluation: OSS vs Frontier",
                 fontsize=15, fontweight="bold", pad=18, color=WHITE)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "bar_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Radar
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(scores: dict[str, dict]) -> None:
    cats   = list(scores["qwen"].keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    N      = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    for model, color in (("qwen", QWEN_COL), ("claude", CLAUDE_COL)):
        vals = [scores[model][c] for c in cats] + [scores[model][cats[0]]]
        ax.plot(angles, vals, color=color, linewidth=2.5)
        ax.fill(angles, vals, color=color, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#8b949e", fontsize=8)
    ax.grid(color=GRID_COL, alpha=0.5)
    ax.spines["polar"].set_color(GRID_COL)

    legend_handles = [
        mpatches.Patch(color=QWEN_COL,   label="Qwen2.5-0.5B (OSS)"),
        mpatches.Patch(color=CLAUDE_COL, label="Claude Sonnet (Frontier)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Performance Radar", color=WHITE, fontsize=14,
                 fontweight="bold", pad=24)

    plt.tight_layout()
    out = OUTPUT_DIR / "radar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Latency
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency(latency: dict[str, dict]) -> None:
    cats   = list(latency["qwen"].keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    x      = np.arange(len(labels))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w / 2, [latency["qwen"][c]   for c in cats], w,
                label="Qwen2.5-0.5B (OSS)",      color=QWEN_COL,   alpha=0.9)
    b2 = ax.bar(x + w / 2, [latency["claude"][c] for c in cats], w,
                label="Claude Sonnet (Frontier)", color=CLAUDE_COL, alpha=0.9)

    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                f"{h:.2f}s", ha="center", va="bottom",
                color=WHITE, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Avg Latency (seconds)", fontsize=12)
    ax.set_title("Response Latency by Category",
                 fontsize=14, fontweight="bold", pad=15, color=WHITE)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "latency_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Summary table
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_table(scores: dict, latency: dict) -> None:
    cats = list(scores["qwen"].keys())

    headers = ["Category", "Qwen Score", "Claude Score",
               "Qwen Latency", "Claude Latency", "Winner"]
    rows = []
    for cat in cats:
        qs = scores["qwen"][cat]
        cs = scores["claude"][cat]
        ql = latency["qwen"].get(cat, 0.0)
        cl = latency["claude"].get(cat, 0.0)
        winner = "Claude ✓" if cs > qs else ("Tie" if cs == qs else "Qwen ✓")
        rows.append([
            CATEGORY_LABELS.get(cat, cat),
            f"{qs:.1f}/10", f"{cs:.1f}/10",
            f"{ql:.2f}s",   f"{cl:.2f}s",
            winner,
        ])

    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Header row styling
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor("#21262d")
        cell.set_text_props(color=WHITE, fontweight="bold")
        cell.set_edgecolor(GRID_COL)

    # Data row styling
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            cell = tbl[i, j]
            cell.set_facecolor(PANEL)
            cell.set_edgecolor(GRID_COL)
            color = WHITE
            if j == 5:  # winner column
                color = CLAUDE_COL if "Claude" in rows[i - 1][5] else (
                    QWEN_COL if "Qwen" in rows[i - 1][5] else WHITE
                )
            cell.set_text_props(color=color)

    ax.set_title("Evaluation Summary", color=WHITE, fontsize=13,
                 fontweight="bold", pad=12)

    plt.tight_layout()
    out = OUTPUT_DIR / "summary_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✅  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"❌  Results file not found: {RESULTS_PATH}")
        print("    Run `python run_evals.py` first.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_results()
    scores  = avg_scores(results)
    latency = avg_latency(results)

    print("Generating charts…")
    plot_bar(scores)
    plot_radar(scores)
    plot_latency(latency)
    plot_summary_table(scores, latency)
    print(f"\nAll charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
