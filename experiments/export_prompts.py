"""Export prompt cases for Project Sentinel experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.data import load_prompt_cases, write_prompt_cases_csv


DEFAULT_OUTPUT = Path("data/prompt_cases.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prompt cases to CSV.")
    parser.add_argument("--source", choices=["sample", "truthfulqa"], default="sample")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cases = load_prompt_cases(source=args.source, limit=args.limit)
    write_prompt_cases_csv(cases, args.output)
    sources = sorted({case.source for case in cases})
    print(f"Exported {len(cases)} prompt cases to {args.output}")
    print(f"Source labels: {', '.join(sources)}")