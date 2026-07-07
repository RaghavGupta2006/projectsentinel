"""Utilities for captured model-response datasets.

Captured responses are outputs from the model being monitored. They are not
labels, judgments, or correctness scores from another LLM.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sentinel.simulator import ModelResponse


CAPTURE_FIELDS = [
    "dataset",
    "model",
    "provider",
    "scenario",
    "window",
    "case_id",
    "question",
    "variant",
    "degradation",
    "output",
]


def write_captured_responses(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_captured_response_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rows_to_model_responses(rows: list[dict[str, str]]) -> list[ModelResponse]:
    responses: list[ModelResponse] = []
    for row in rows:
        responses.append(
            ModelResponse(
                window=int(row["window"]),
                case_id=row["case_id"],
                question=row["question"],
                variant=int(row["variant"]),
                degradation=row["degradation"],
                output=row["output"],
            )
        )
    return responses