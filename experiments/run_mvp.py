"""Run the Project Sentinel 7-day MVP experiment suite."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.data import PromptCase, load_prompt_cases
from sentinel.scoring import detect_degradation, score_window
from sentinel.signals import extract_window_signals
from sentinel.simulator import ModelResponse, simulate_windows


SCENARIOS = ["prompt_corruption", "context_truncation", "temperature_spike"]
DEGRADATION_START = 4
RESULTS_PATH = Path("outputs/mvp_results.csv")
METRICS_PATH = Path("outputs/mvp_metrics.csv")


def group_by_window(responses: list[ModelResponse]) -> dict[int, list[ModelResponse]]:
    grouped: dict[int, list[ModelResponse]] = defaultdict(list)
    for response in responses:
        grouped[response.window].append(response)
    return dict(sorted(grouped.items()))


def run_experiment(
    scenario: str = "prompt_corruption",
    cases: list[PromptCase] | None = None,
    dataset_label: str = "sample",
    zscore_k: float | None = None,
) -> list[dict[str, object]]:
    responses = simulate_windows(cases=cases, degradation=scenario, degradation_start=DEGRADATION_START)
    windows = group_by_window(responses)
    baseline = windows[1]

    rows: list[dict[str, object]] = []
    baseline_signals = extract_window_signals(baseline, baseline)
    baseline_score = score_window(baseline_signals)

    history_scores: list[float] = []

    for window, window_responses in windows.items():
        signals = extract_window_signals(window_responses, baseline)
        reliability_score = score_window(signals)
        alert = detect_degradation(
            reliability_score,
            baseline_score,
            history_scores=history_scores,
            zscore_k=zscore_k,
        )
        actual_degraded = window_responses[0].degradation != "none"

        rows.append(
            {
                "dataset": dataset_label,
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
                "degradation_alert": alert,
            }
        )
        history_scores.append(reliability_score)

    return rows


def run_all_experiments(
    scenarios: list[str] | None = None,
    cases: list[PromptCase] | None = None,
    dataset_label: str = "sample",
    zscore_k: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios or SCENARIOS:
        rows.extend(run_experiment(scenario, cases=cases, dataset_label=dataset_label, zscore_k=zscore_k))
    return rows


def compute_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scenario_names = sorted({str(row["scenario"]) for row in rows})
    metric_rows = [_compute_metric_row("all", rows)]
    for scenario in scenario_names:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        metric_rows.append(_compute_metric_row(scenario, scenario_rows))
    return metric_rows


def _compute_metric_row(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    tp = sum(1 for row in rows if row["actual_degraded"] and row["degradation_alert"])
    fp = sum(1 for row in rows if not row["actual_degraded"] and row["degradation_alert"])
    tn = sum(1 for row in rows if not row["actual_degraded"] and not row["degradation_alert"])
    fn = sum(1 for row in rows if row["actual_degraded"] and not row["degradation_alert"])

    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    detection_lead_time = _detection_lead_time(rows)

    return {
        "scenario": label,
        "windows": total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "detection_lead_time_windows": detection_lead_time,
    }


def _detection_lead_time(rows: list[dict[str, object]]) -> int | str:
    positives = [row for row in rows if row["actual_degraded"]]
    if not positives:
        return "n/a"

    degradation_start = min(int(row["window"]) for row in positives)
    alerts = [int(row["window"]) for row in positives if row["degradation_alert"]]
    if not alerts:
        return "missed"

    return min(alerts) - degradation_start


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: list[dict[str, object]], metrics: list[dict[str, object]], results_path: Path, metrics_path: Path) -> None:
    write_csv(results_path, rows)
    write_csv(metrics_path, metrics)


def print_results(rows: list[dict[str, object]], metrics: list[dict[str, object]], results_path: Path, metrics_path: Path) -> None:
    print("Project Sentinel Experiment Results")
    print("=" * 114)
    print(
        f"{'Dataset':<18}{'Scenario':<20}{'Window':<8}{'Actual':<20}{'SRS':<10}{'Alert':<8}"
        f"{'Consistency':<14}{'DriftScore':<12}{'Compliance':<12}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<18}"
            f"{row['scenario']:<20}"
            f"{row['window']:<8}"
            f"{row['actual_degradation']:<20}"
            f"{row['semantic_reliability_score']:<10}"
            f"{str(row['degradation_alert']):<8}"
            f"{row['semantic_consistency']:<14}"
            f"{row['embedding_drift']:<12}"
            f"{row['task_compliance']:<12}"
        )

    print("=" * 114)
    print("Evaluation Metrics")
    print("-" * 104)
    print(
        f"{'Scenario':<20}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}"
        f"{'FPR':<8}{'Lead Time':<10}{'TP':<5}{'FP':<5}{'TN':<5}{'FN':<5}"
    )
    for row in metrics:
        print(
            f"{row['scenario']:<20}"
            f"{row['accuracy']:<10}"
            f"{row['precision']:<11}"
            f"{row['recall']:<9}"
            f"{row['false_positive_rate']:<8}"
            f"{str(row['detection_lead_time_windows']):<10}"
            f"{row['true_positive']:<5}"
            f"{row['false_positive']:<5}"
            f"{row['true_negative']:<5}"
            f"{row['false_negative']:<5}"
        )

    print("=" * 104)
    print(f"Saved window results to {results_path}")
    print(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project Sentinel experiments.")
    parser.add_argument("--dataset", choices=["sample", "truthfulqa", "gsm8k", "csv"], default="sample")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--zscore-k", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_prompt_cases(source=args.dataset, limit=args.limit, csv_path=args.csv_path)
    source_labels = sorted({case.source for case in cases})
    dataset_label = args.dataset if source_labels == [args.dataset] else f"{args.dataset}({'+'.join(source_labels)})"

    experiment_rows = run_all_experiments(cases=cases, dataset_label=dataset_label, zscore_k=args.zscore_k)
    metric_rows = compute_metrics(experiment_rows)
    write_outputs(experiment_rows, metric_rows, args.results_path, args.metrics_path)
    print_results(experiment_rows, metric_rows, args.results_path, args.metrics_path)


if __name__ == "__main__":
    main()