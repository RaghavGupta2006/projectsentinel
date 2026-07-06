"""Generate clean visualization plots for Project Sentinel evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path("outputs")


def plot_reliability_over_time(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        print(f"Skipping: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    scenarios = df["scenario"].unique()

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario].sort_values("window")
        plt.plot(
            scenario_df["window"],
            scenario_df["semantic_reliability_score"],
            marker="o",
            linewidth=2,
            label=scenario.replace("_", " ").title(),
        )

    plt.axvline(x=4, color="red", linestyle="--", alpha=0.7, label="Degradation Start")
    plt.title("Semantic Reliability Score (SRS) Over Time Windows", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time Window", fontsize=11, labelpad=10)
    plt.ylabel("Semantic Reliability Score", fontsize=11, labelpad=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved reliability plot to {output_path}")


def plot_ablation_comparison(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        print(f"Skipping: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Filter out full score or re-arrange for plotting
    df = df.sort_values("accuracy", ascending=False)

    plt.figure(figsize=(10, 5))
    colors = ["#2563eb" if x == "full_score" else "#6b7280" for x in df["variant"]]

    # Clean variant names
    labels = [x.replace("_", " ").title() for x in df["variant"]]

    bars = plt.bar(labels, df["accuracy"], color=colors, width=0.6)

    # Add accuracy values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Degradation Detection Accuracy Across Ablation Variants", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Signal Variant", fontsize=11, labelpad=10)
    plt.ylabel("Detection Accuracy", fontsize=11, labelpad=10)
    plt.ylim(0.0, 1.15)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ablation plot to {output_path}")


def plot_embedding_comparison(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        print(f"Skipping: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Re-structure data to plot side-by-side comparison
    # We want scenario vs accuracy for each backend
    pivoted = df.pivot(index="scenario", columns="embedding_backend", values="accuracy")

    # Re-order and clean index names
    pivoted.index = [x.replace("_", " ").title() for x in pivoted.index]

    plt.figure(figsize=(10, 5))
    ax = pivoted.plot(kind="bar", width=0.6, color=["#10b981", "#3b82f6"], edgecolor="none", figsize=(10, 5))

    plt.title("Detection Accuracy Comparison by Embedding Backend", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Scenario", fontsize=11, labelpad=10)
    plt.ylabel("Accuracy", fontsize=11, labelpad=10)
    plt.ylim(0.0, 1.15)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Embedding Backend", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved embedding comparison plot to {output_path}")


def main() -> None:
    # Synthetic MVP plots
    plot_reliability_over_time(
        csv_path=OUTPUT_DIR / "mvp_results.csv",
        output_path=OUTPUT_DIR / "reliability_over_time.png",
    )
    plot_ablation_comparison(
        csv_path=OUTPUT_DIR / "ablation_metrics.csv",
        output_path=OUTPUT_DIR / "ablation_comparison.png",
    )

    # Real model plots
    plot_reliability_over_time(
        csv_path=OUTPUT_DIR / "real_model_analysis_results.csv",
        output_path=OUTPUT_DIR / "real_model_reliability_over_time.png",
    )
    plot_ablation_comparison(
        csv_path=OUTPUT_DIR / "real_model_ablation_metrics.csv",
        output_path=OUTPUT_DIR / "real_model_ablation_comparison.png",
    )

    # Embedding benchmark plots
    plot_embedding_comparison(
        csv_path=OUTPUT_DIR / "embedding_benchmark_results.csv",
        output_path=OUTPUT_DIR / "embedding_comparison.png",
    )


if __name__ == "__main__":
    main()
