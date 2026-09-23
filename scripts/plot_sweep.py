#!/usr/bin/env python3
"""Plot a sweep's accuracy and AUC against difficulty.

    python scripts/plot_sweep.py sweeps/overlap/sweep_results.json

Separate from sweep.py so a finished sweep can be replotted without re-running
hours of training. Writes sweep.png next to the input JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Accuracy and AUC are both probabilities, so they share one axis -- a second
# y-scale would imply a difference that is not there.
ACCURACY_COLOUR = "#2a78d6"
AUC_COLOUR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"

# the accuracy window Phase 3 is looking for: hard enough to discriminate
# between models, not so hard that the task is noise
BAND = (0.75, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="a sweep_results.json")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output PNG (default: sweep.png beside the input)")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    rows = [r for r in payload.get("results", []) if "error" not in r]
    failed = [r for r in payload.get("results", []) if "error" in r]
    if not rows:
        raise SystemExit(f"{args.results}: no successful levels to plot")
    rows.sort(key=lambda r: r["d"])

    d = [r["d"] for r in rows]
    accuracy = [r["accuracy"] for r in rows]
    auc = [r["auc"] for r in rows]

    figure, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.axhspan(*BAND, color=INK, alpha=0.05, zorder=0)
    ax.annotate(f"target band {BAND[0]:.2f}-{BAND[1]:.2f}",
                xy=(0.015, BAND[1]), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color=MUTED)

    ax.plot(d, accuracy, color=ACCURACY_COLOUR, linewidth=2, marker="o",
            markersize=6, label="accuracy", zorder=3)
    ax.plot(d, auc, color=AUC_COLOUR, linewidth=2, marker="s", markersize=5,
            label="ROC-AUC", zorder=2)

    # label the endpoints only: a number on every point is noise. The two
    # series are offset in opposite directions so equal values do not collide.
    for series, colour, dy in ((accuracy, ACCURACY_COLOUR, 10), (auc, AUC_COLOUR, -14)):
        for i in (0, len(d) - 1):
            ax.annotate(f"{series[i]:.3f}", (d[i], series[i]),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=colour)

    chosen = payload.get("chosen_d", payload.get("operating_point"))
    if chosen is not None:
        chosen_d = chosen.get("d") if isinstance(chosen, dict) else chosen
        ax.axvline(chosen_d, color=INK, linewidth=1, linestyle="--", alpha=0.55,
                   zorder=1)
        ax.annotate(f"operating point  d={chosen_d:g}", xy=(chosen_d, 0.02),
                    xycoords=("data", "axes fraction"), rotation=90,
                    va="bottom", ha="right", fontsize=8, color=MUTED)

    ax.set_xlabel("difficulty  d", color=INK)
    ax.set_ylabel("score", color=INK)
    ax.set_title("Baseline accuracy against class overlap", color=INK, fontsize=12)
    ax.set_ylim(min(0.5, min(accuracy + auc) - 0.05), 1.02)
    ax.set_xlim(min(d) - 0.03, max(d) + 0.03)

    ax.grid(axis="y", color=MUTED, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)

    if failed:
        figure.text(0.99, 0.01,
                    f"{len(failed)} level(s) failed and are not plotted",
                    ha="right", fontsize=8, color=MUTED)

    figure.tight_layout()
    out = args.out or args.results.parent / "sweep.png"
    figure.savefig(out, dpi=args.dpi)
    print(f"wrote {out}  ({len(rows)} levels"
          f"{f', {len(failed)} failed' if failed else ''})")


if __name__ == "__main__":
    main()
