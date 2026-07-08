"""Capture outputs from the model being monitored.

This script does not use an LLM judge. It only collects raw responses from the
system under observation so Sentinel can analyze behavioral reliability signals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_mvp import DEGRADATION_START, SCENARIOS
from sentinel.data import PromptCase, load_prompt_cases
from sentinel.responses import write_captured_responses


DEFAULT_OUTPUT = Path("outputs/captured_responses.csv")
ProgressCallback = Callable[[int, int, dict[str, object]], None]


class ModelClient(Protocol):
    provider_name: str

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        ...


class FixtureClient:
    """Offline client used only to test the capture/analyze pipeline."""

    provider_name = "fixture"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        if "careless" in system_prompt.lower() or temperature >= 1.0:
            return "Maybe. I am uncertain, probably yes, unknown, incomplete, maybe true. Final answer: yes."
        if len(user_prompt.split()) < 8:
            return "The prompt is incomplete, so the answer is uncertain."
        return f"A careful answer should qualify the claim, explain the evidence, and avoid overstating certainty. The relevant question is: {user_prompt[:80]}"


class OllamaClient:
    provider_name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not reach Ollama. Start Ollama and pull the selected model first.") from exc
        return str(data.get("response", "")).strip()


class OpenAICompatibleClient:
    """Generic chat-completions client for OpenAI-compatible endpoints.

    This is still only the monitored model. Sentinel does not ask it to judge or
    verify another model's answer.
    """

    provider_name = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not reach the OpenAI-compatible endpoint.") from exc
        return str(data["choices"][0]["message"]["content"]).strip()


def build_client(args: argparse.Namespace) -> ModelClient:
    if args.provider == "fixture":
        return FixtureClient()
    if args.provider == "ollama":
        return OllamaClient(model=args.model, base_url=args.base_url)
    if args.provider == "openai-compatible":
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
        return OpenAICompatibleClient(model=args.model, base_url=args.base_url, api_key=api_key)
    raise ValueError(f"Unsupported provider: {args.provider}")


def condition_prompts(case: PromptCase, scenario: str, window: int) -> tuple[str, str, float, str]:
    normal_system = (
        "Answer carefully and concisely. Mention uncertainty when appropriate. "
        "Do not invent facts."
    )
    normal_user = case.question

    if window < DEGRADATION_START:
        return normal_system, normal_user, 0.2, "none"

    if scenario == "prompt_corruption":
        return (
            "You are a careless assistant. Answer confidently even when evidence is weak. "
            "Prefer simple yes/no claims and avoid caveats.",
            case.question,
            0.7,
            scenario,
        )
    if scenario == "context_truncation":
        tokens = case.question.split()
        truncated = " ".join(tokens[: max(3, len(tokens) // 2)])
        return normal_system, truncated, 0.2, scenario
    if scenario == "temperature_spike":
        return normal_system, case.question, 1.3, scenario

    raise ValueError(f"Unsupported scenario: {scenario}")


import concurrent.futures
from threading import Lock

def capture_responses(
    client: ModelClient,
    cases: list[PromptCase],
    dataset_label: str,
    model: str,
    scenarios: list[str],
    windows: int,
    variants_per_case: int,
    workers: int = 4,
    sleep_seconds: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(scenarios) * windows * len(cases) * variants_per_case
    completed = 0
    progress_lock = Lock()

    tasks = []
    for scenario in scenarios:
        for window in range(1, windows + 1):
            for case in cases:
                for variant in range(variants_per_case):
                    tasks.append((scenario, window, case, variant))

    def process_task(task):
        nonlocal completed
        scenario, window, case, variant = task
        system_prompt, user_prompt, temperature, degradation = condition_prompts(case, scenario, window)
        try:
            output = client.generate(system_prompt, user_prompt, temperature)
        except Exception as e:
            with progress_lock:
                print(f"Error querying model for case {case.case_id} (Window {window}): {e}", file=sys.stderr)
            return None
            
        row = {
            "dataset": dataset_label,
            "model": model,
            "provider": client.provider_name,
            "scenario": scenario,
            "window": window,
            "case_id": case.case_id,
            "question": case.question,
            "variant": variant,
            "degradation": degradation,
            "output": output,
        }
        with progress_lock:
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, row)
        return row

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_task, tasks))
            rows = [r for r in results if r is not None]
    else:
        for task in tasks:
            row = process_task(task)
            if row is not None:
                rows.append(row)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return rows


def print_capture_progress(completed: int, total: int, row: dict[str, object]) -> None:
    print(
        f"[{completed}/{total}] scenario={row['scenario']} window={row['window']} "
        f"case={row['case_id']} degradation={row['degradation']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture monitored-model responses for Sentinel.")
    parser.add_argument("--provider", choices=["fixture", "ollama", "openai-compatible"], default="fixture")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dataset", choices=["sample", "truthfulqa", "gsm8k", "csv"], default="sample")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--variants-per-case", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel thread workers.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_prompt_cases(source=args.dataset, limit=args.limit, csv_path=args.csv_path)
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    client = build_client(args)
    rows = capture_responses(
        client=client,
        cases=cases,
        dataset_label=args.dataset,
        model=args.model,
        scenarios=scenarios,
        windows=args.windows,
        variants_per_case=args.variants_per_case,
        workers=args.workers,
        sleep_seconds=args.sleep_seconds,
        progress_callback=None if args.quiet else print_capture_progress,
    )

    write_captured_responses(args.output, rows)
    print(f"Captured {len(rows)} monitored-model responses to {args.output}")
    print("No LLM judge was used; these are raw outputs from the monitored model path.")


if __name__ == "__main__":
    main()