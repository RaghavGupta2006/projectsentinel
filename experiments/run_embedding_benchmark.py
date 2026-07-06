"""Run a comparative benchmark of embedding backends for Project Sentinel.

This script evaluates how different embedding backends (Hashed Bag-of-Words vs. Sentence-Transformers)
perform across evaluation metrics (accuracy, precision, recall, false positive rate, detection lead time).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_mvp import run_all_experiments, compute_metrics
from sentinel.data import load_prompt_cases

OUTPUT_DIR = Path("outputs")
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "embedding_benchmark_results.csv"


def run_benchmark(dataset: str, limit: int, csv_path: Path | None = None) -> list[dict[str, object]]:
    cases = load_prompt_cases(source=dataset, limit=limit, csv_path=csv_path)

    # 1. Evaluate with Hashed Bag-of-Words
    print("Evaluating with Hashed Bag-of-Words backend...", flush=True)
    os.environ["SENTINEL_EMBEDDINGS"] = "hashed"
    hashed_rows = run_all_experiments(cases=cases, dataset_label=dataset)
    hashed_metrics = compute_metrics(hashed_rows)

    # 2. Evaluate with Sentence-Transformers
    print("Evaluating with Sentence-Transformers backend...", flush=True)
    os.environ["SENTINEL_EMBEDDINGS"] = "sentence-transformers"
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print(
            "Error: sentence-transformers is not installed. Please run `pip install sentence-transformers` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    st_rows = run_all_experiments(cases=cases, dataset_label=dataset)
    st_metrics = compute_metrics(st_rows)

    # 3. Combine results
    comparison_rows: list[dict[str, object]] = []

    for metric in hashed_metrics:
        comparison_rows.append(
            {
                "embedding_backend": "hashed_bow",
                "scenario": metric["scenario"],
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "false_positive_rate": metric["false_positive_rate"],
                "detection_lead_time_windows": metric["detection_lead_time_windows"],
            }
        )

    for metric in st_metrics:
        comparison_rows.append(
            {
                "embedding_backend": "sentence-transformers",
                "scenario": metric["scenario"],
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "false_positive_rate": metric["false_positive_rate"],
                "detection_lead_time_windows": metric["detection_lead_time_windows"],
            }
        )

    return comparison_rows


def run_captured_benchmark(captured_path: Path) -> list[dict[str, object]]:
    from sentinel.responses import read_captured_response_rows
    from experiments.analyze_captured import analyze_captured_rows

    captured_rows = read_captured_response_rows(captured_path)

    # 1. Evaluate Hashed Bag-of-Words
    print(f"Evaluating Hashed Bag-of-Words on captured responses from {captured_path}...", flush=True)
    os.environ["SENTINEL_EMBEDDINGS"] = "hashed"
    hashed_rows = analyze_captured_rows(captured_rows)
    hashed_metrics = compute_metrics(hashed_rows)

    # 2. Evaluate Sentence-Transformers
    print(f"Evaluating Sentence-Transformers on captured responses from {captured_path}...", flush=True)
    os.environ["SENTINEL_EMBEDDINGS"] = "sentence-transformers"
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print(
            "Error: sentence-transformers is not installed. Please run `pip install sentence-transformers` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    st_rows = analyze_captured_rows(captured_rows)
    st_metrics = compute_metrics(st_rows)

    # 3. Combine results
    comparison_rows: list[dict[str, object]] = []

    for metric in hashed_metrics:
        comparison_rows.append(
            {
                "embedding_backend": "hashed_bow",
                "scenario": metric["scenario"],
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "false_positive_rate": metric["false_positive_rate"],
                "detection_lead_time_windows": metric["detection_lead_time_windows"],
            }
        )

    for metric in st_metrics:
        comparison_rows.append(
            {
                "embedding_backend": "sentence-transformers",
                "scenario": metric["scenario"],
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "false_positive_rate": metric["false_positive_rate"],
                "detection_lead_time_windows": metric["detection_lead_time_windows"],
            }
        )

    return comparison_rows


def write_benchmark_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_benchmark_results(rows: list[dict[str, object]]) -> None:
    print("\nProject Sentinel Embedding Backend Benchmark Comparison", flush=True)
    print("=" * 110, flush=True)
    print(
        f"{'Embedding Backend':<25}{'Scenario':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'FPR':<8}{'Lead Time':<10}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for row in rows:
        print(
            f"{row['embedding_backend']:<25}"
            f"{row['scenario']:<22}"
            f"{row['accuracy']:<10}"
            f"{row['precision']:<11}"
            f"{row['recall']:<9}"
            f"{row['false_positive_rate']:<8}"
            f"{str(row['detection_lead_time_windows']):<10}",
            flush=True,
        )
    print("=" * 110, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding benchmark comparison.")
    parser.add_argument("--dataset", choices=["sample", "truthfulqa", "gsm8k", "csv"], default="sample")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--input", type=Path, default=None, help="Path to captured responses CSV")
    parser.add_argument("--output", type=Path, default=BENCHMARK_RESULTS_PATH)
    args = parser.parse_args()

    try:
        if args.input:
            comparison_rows = run_captured_benchmark(args.input)
        else:
            comparison_rows = run_benchmark(args.dataset, args.limit, args.csv_path)
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        sys.exit(1)

    write_benchmark_csv(args.output, comparison_rows)
    print_benchmark_results(comparison_rows)
    print(f"Saved benchmark comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
