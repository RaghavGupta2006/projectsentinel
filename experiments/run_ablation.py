"""Ablation study for Project Sentinel reliability signals."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_mvp import SCENARIOS, compute_metrics, group_by_window
from sentinel.data import load_prompt_cases
from sentinel.scoring import detect_degradation, score_window
from sentinel.signals import extract_window_signals
from sentinel.simulator import simulate_windows
from sentinel.responses import read_captured_response_rows, rows_to_model_responses


SIGNALS = [
    "semantic_consistency",
    "response_stability",
    "embedding_drift",
    "confidence_proxy",
    "task_compliance",
]
OUTPUT_PATH = Path("outputs/ablation_metrics.csv")


def run_ablation(dataset: str = "sample", limit: int = 32, csv_path: Path | None = None) -> list[dict[str, object]]:
    cases = load_prompt_cases(source=dataset, limit=limit, csv_path=csv_path)
    rows: list[dict[str, object]] = []

    rows.extend(_run_variant("full_score", dataset, cases, active_signal=None))
    for signal_name in SIGNALS:
        rows.extend(_run_variant(signal_name, dataset, cases, active_signal=signal_name))

    return rows


def _run_variant(dataset: str, dataset_label: str, cases: list, active_signal: str | None) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        responses = simulate_windows(cases=cases, degradation=scenario)
        windows = group_by_window(responses)
        baseline = windows[1]
        baseline_signals = extract_window_signals(baseline, baseline)
        baseline_score = _variant_score(baseline_signals, active_signal)

        for window, window_responses in windows.items():
            signals = extract_window_signals(window_responses, baseline)
            reliability_score = _variant_score(signals, active_signal)
            actual_degraded = window_responses[0].degradation != "none"
            output_rows.append(
                {
                    "variant": dataset,
                    "dataset": dataset_label,
                    "scenario": scenario,
                    "window": window,
                    "actual_degradation": window_responses[0].degradation,
                    "actual_degraded": actual_degraded,
                    "semantic_reliability_score": reliability_score,
                    "degradation_alert": detect_degradation(reliability_score, baseline_score),
                }
            )
    return output_rows


def run_captured_ablation(captured_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(_run_captured_variant("full_score", captured_rows, active_signal=None))
    for signal_name in SIGNALS:
        rows.extend(_run_captured_variant(signal_name, captured_rows, active_signal=signal_name))
    return rows


def _run_captured_variant(variant_name: str, captured_rows: list[dict[str, str]], active_signal: str | None) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    from experiments.analyze_captured import group_rows, group_responses_by_window

    for (dataset, model, scenario), group in group_rows(captured_rows).items():
        responses = rows_to_model_responses(group)
        windows = group_responses_by_window(responses)
        baseline_win = min(windows)
        baseline = windows[baseline_win]
        baseline_signals = extract_window_signals(baseline, baseline)
        baseline_score = _variant_score(baseline_signals, active_signal)

        for window, window_responses in windows.items():
            signals = extract_window_signals(window_responses, baseline)
            reliability_score = _variant_score(signals, active_signal)
            actual_degraded = window_responses[0].degradation != "none"
            output_rows.append(
                {
                    "variant": variant_name,
                    "dataset": dataset,
                    "model": model,
                    "scenario": scenario,
                    "window": window,
                    "actual_degradation": window_responses[0].degradation,
                    "actual_degraded": actual_degraded,
                    "semantic_reliability_score": reliability_score,
                    "degradation_alert": detect_degradation(reliability_score, baseline_score),
                }
            )
    return output_rows


def _variant_score(signals: dict[str, float], active_signal: str | None) -> float:
    if active_signal is None:
        return score_window(signals)
    return round(signals[active_signal], 4)


def summarize_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    variants = sorted({str(row["variant"]) for row in rows})
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        metric_row = compute_metrics(variant_rows)[0]
        summaries.append(
            {
                "variant": variant,
                "accuracy": metric_row["accuracy"],
                "precision": metric_row["precision"],
                "recall": metric_row["recall"],
                "false_positive_rate": metric_row["false_positive_rate"],
                "detection_lead_time_windows": metric_row["detection_lead_time_windows"],
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signal ablation study.")
    parser.add_argument("--dataset", choices=["sample", "truthfulqa", "gsm8k", "csv"], default="sample")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--input", type=Path, default=None, help="Path to captured responses CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input:
        captured_rows = read_captured_response_rows(args.input)
        ablation_rows = run_captured_ablation(captured_rows)
    else:
        ablation_rows = run_ablation(dataset=args.dataset, limit=args.limit, csv_path=args.csv_path)

    summary_rows = summarize_ablation(ablation_rows)
    write_csv(args.output, summary_rows)
    print("Project Sentinel Ablation Metrics")
    print("=" * 82)
    for row in summary_rows:
        print(row)
    print(f"Saved ablation metrics to {args.output}")