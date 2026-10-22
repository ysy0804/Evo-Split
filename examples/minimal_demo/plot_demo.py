#!/usr/bin/env python3
"""Render the Evo-Split minimal demo as a compact three-panel figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.input)
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    colors = {"Evo-Split": "#2878B5", "Cold-Restart": "#D65F5F"}

    action_ids: dict[str, int] = {}
    for method, values in by_method.items():
        for value in values:
            action_ids.setdefault(value["action"], len(action_ids))
        axes[0].step(
            [int(v["request"]) for v in values],
            [action_ids[v["action"]] for v in values],
            where="post", label=method, color=colors[method], linewidth=1.8)
    axes[0].set_ylabel("Action ID")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("Online action adaptation")

    for method, values in by_method.items():
        cumulative, total = [], 0.0
        for value in values:
            total += float(value["regret_ms"])
            cumulative.append(total)
        axes[1].plot([int(v["request"]) for v in values], cumulative,
                     label=method, color=colors[method], linewidth=2.0)
    axes[1].set_ylabel("Cumulative regret (ms)")
    axes[1].set_xlabel("Request")
    axes[1].set_title("Adaptation cost")

    update = next(row for row in rows if row["method"] == "Evo-Split" and row["update_event"] == "1")
    labels = ["Reused", "Recalibrated", "Removed"]
    values = [int(update["reused"]), int(update["reset"]), int(update["removed"])]
    axes[2].bar(labels, values, color=["#4E9F3D", "#F3A712", "#C94C4C"], width=0.55)
    axes[2].set_ylabel("Action states")
    axes[2].set_title("Minimal-impact recovery at request 21")
    axes[2].set_xlabel("Recovery class")

    for axis in axes[:2]:
        axis.axvline(11, color="#888888", linestyle="--", linewidth=1)
        axis.axvline(21, color="#E07A2D", linestyle="--", linewidth=1.2)
        axis.grid(axis="y", alpha=0.22)
    axes[2].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"FIGURE={args.output}")


if __name__ == "__main__":
    main()

