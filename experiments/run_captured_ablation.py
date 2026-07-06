"""Reproducible script to run Sentinel signal ablation over captured model outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation import run_captured_ablation, summarize_ablation, write_csv
from sentinel.responses import read_captured_response_rows

DEFAULT_INPUT = Path("outputs/real_model_responses.csv")
DEFAULT_OUTPUT = Path("outputs/real_model_ablation_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run signal ablation study on captured responses.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to captured responses CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save ablation metrics CSV")
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"Error: Input file {args.input} does not exist. Run the response capture script first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading captured responses from {args.input}...", flush=True)
    captured_rows = read_captured_response_rows(args.input)

    print("Running ablation study variants...", flush=True)
    ablation_rows = run_captured_ablation(captured_rows)

    print("Computing metrics...", flush=True)
    summary_rows = summarize_ablation(ablation_rows)

    write_csv(args.output, summary_rows)

    print("\nProject Sentinel Real-Model Ablation Metrics", flush=True)
    print("=" * 82, flush=True)
    for row in summary_rows:
        print(row, flush=True)
    print(f"\nSaved ablation metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
