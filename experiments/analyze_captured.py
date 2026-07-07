"""Analyze captured monitored-model responses with Sentinel.

This script computes Sentinel signals over already-captured model outputs. It does
not call any LLM and does not perform answer verification.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_mvp import compute_metrics, print_results, write_csv
from sentinel.responses import read_captured_response_rows, rows_to_model_responses
from sentinel.scoring import detect_degradation, score_window
from sentinel.signals import extract_window_signals
from sentinel.simulator import ModelResponse


DEFAULT_INPUT = Path("outputs/captured_responses.csv")
DEFAULT_RESULTS = Path("outputs/captured_analysis_results.csv")
DEFAULT_METRICS = Path("outputs/captured_analysis_metrics.csv")


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", "unknown"), row.get("model", "unknown"), row.get("scenario", "unknown"))
        grouped[key].append(row)
    return grouped


def group_responses_by_window(responses: list[ModelResponse]) -> dict[int, list[ModelResponse]]:
    grouped: dict[int, list[ModelResponse]] = defaultdict(list)
    for response in responses:
        grouped[response.window].append(response)
    return dict(sorted(grouped.items()))


def analyze_captured_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for (dataset, model, scenario), group in group_rows(rows).items():
        responses = rows_to_model_responses(group)
        windows = group_responses_by_window(responses)
        baseline = windows[min(windows)]
        baseline_signals = extract_window_signals(baseline, baseline)
        baseline_score = score_window(baseline_signals)

        for window, window_responses in windows.items():
            signals = extract_window_signals(window_responses, baseline)
            reliability_score = score_window(signals)
            actual_degraded = window_responses[0].degradation != "none"
            output_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scenario": scenario,
                    "window": window,
                    "actual_degradation": window_responses[0].degradation,
                    "actual_degraded": actual_degraded,
                    "semantic_consistency": round(signals["semantic_consistency"], 4),
                    "response_stability": round(signals["response_stability"], 4),
                    "embedding_drift": round(signals["embedding_drift"], 4),
                    "confidence_proxy": round(signals["confidence_proxy"], 4),
                    "task_compliance": round(signals["task_compliance"], 4),
                    "semantic_reliability_score": reliability_score,
                    "degradation_alert": detect_degradation(reliability_score, baseline_score),
                }
            )
    return output_rows


def write_analysis_outputs(results_path: Path, metrics_path: Path, rows: list[dict[str, object]]) -> None:
    metrics = compute_metrics(rows)
    write_csv(results_path, rows)
    write_csv(metrics_path, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze captured monitored-model responses.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captured_rows = read_captured_response_rows(args.input)
    analysis_rows = analyze_captured_rows(captured_rows)
    metrics = compute_metrics(analysis_rows)
    write_csv(args.results_path, analysis_rows)
    write_csv(args.metrics_path, metrics)
    print_results(analysis_rows, metrics, args.results_path, args.metrics_path)
    print("Analysis used Sentinel signals only; no LLM judge or answer verifier was called.")


if __name__ == "__main__":
    main()