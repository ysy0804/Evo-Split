#!/usr/bin/env python3
"""Generate compact adaptation and update-recovery figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_adaptation(out: Path) -> None:
    datasets = {
        "Letv / MobileNetV2": DATA / "e3_letv_mobilenet_summary.csv",
        "Letv / ResNet50": DATA / "e3_letv_resnet50_summary.csv",
        "Honor / EfficientNet-B0": DATA / "e3_honor_efficientnet_summary.csv",
        "vivo / EfficientNet-B0": DATA / "e3_vivo_efficientnet_summary.csv",
    }
    methods = ["always_local", "fixed_split", "offline_profile", "pfhoc_cpp"]
    labels = ["Always Local", "Fixed Split", "Offline Profile", "Evo-Split"]
    colors = ["#8C8C8C", "#E09F3E", "#6A5ACD", "#2878B5"]
    values = {method: [] for method in methods}
    for path in datasets.values():
        rows = read_csv(path)
        for method in methods:
            selected = [float(row["normalized_regret"]) for row in rows
                        if row["controller"] == method]
            values[method].append(sum(selected) / len(selected))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = list(range(len(datasets)))
    width = 0.18
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        ax.bar([item + (index - 1.5) * width for item in x], values[method],
               width=width, label=label, color=color)
    ax.set_xticks(x, datasets.keys(), rotation=10, ha="right")
    ax.set_ylabel("Mean normalized cumulative regret")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "adaptation_regret.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(out: Path) -> None:
    rows = read_csv(DATA / "e5_recovery_summary.csv")
    updates = ["U0", "U1", "U2", "U3", "U4"]
    methods = ["Cold-Restart", "Full-Reprofile", "V-MICA"]
    labels = ["Cold Restart", "Full Reprofile", "Evo-Split Recovery"]
    colors = ["#8C8C8C", "#D65F5F", "#4E9F3D"]
    lookup = {(row["update_id"], row["method"]): row for row in rows}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = list(range(len(updates)))
    width = 0.24
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = []
        for update in updates:
            text = lookup.get((update, method), {}).get("mean_probe_requests", "")
            values.append(float(text) if text and not math.isnan(float(text)) else 0.0)
        ax.bar([item + (index - 1) * width for item in x], values,
               width=width, label=label, color=color)
    ax.set_xticks(x, updates)
    ax.set_xlabel("Model update type")
    ax.set_ylabel("Mean recovery probe requests")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "update_recovery_probes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_adaptation(args.out)
    plot_recovery(args.out)
    print(f"FIGURES={args.out}")


if __name__ == "__main__":
    main()

