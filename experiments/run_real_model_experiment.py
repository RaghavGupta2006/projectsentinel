"""Run a monitored-model experiment end to end.

This is the real-output experiment path: capture raw outputs from the monitored
model, then analyze those outputs with Sentinel. It does not use an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_captured import analyze_captured_rows
from experiments.capture_responses import build_client, capture_responses, print_capture_progress
from experiments.run_mvp import SCENARIOS, compute_metrics, print_results, write_csv
from sentinel.data import load_prompt_cases
from sentinel.responses import read_captured_response_rows, write_captured_responses


RESPONSES_PATH = Path("outputs/real_model_responses.csv")
RESULTS_PATH = Path("outputs/real_model_analysis_results.csv")
METRICS_PATH = Path("outputs/real_model_analysis_metrics.csv")


def ollama_available(base_url: str, model_name: str | None = None) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:
            if response.status == 200:
                if model_name:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    found = False
                    normalized_target = model_name.lower()
                    if ":" not in normalized_target:
                        normalized_target = f"{normalized_target}:latest"
                    for m in models:
                        name = str(m).lower()
                        if name == normalized_target or name == model_name.lower():
                            found = True
                            break
                    if not found:
                        return False, f"Ollama is running, but model '{model_name}' was not found. Please run `ollama pull {model_name}` first."
                return True, "Ollama server is reachable and model is available."
            return False, f"Ollama returned HTTP {response.status}."
    except urllib.error.URLError as exc:
        return False, f"Ollama is not reachable at {base_url}: {exc}"


def provider_ready(args: argparse.Namespace) -> tuple[bool, str]:
    if args.provider == "ollama":
        return ollama_available(args.base_url, args.model)
    if args.provider == "openai-compatible":
        if os.getenv(args.api_key_env):
            return True, f"Environment variable {args.api_key_env} is set."
        return False, f"Missing API key environment variable: {args.api_key_env}"
    if args.provider == "fixture":
        return True, "Fixture provider is available for smoke tests only."
    return False, f"Unsupported provider: {args.provider}"


def run_real_model_experiment(args: argparse.Namespace) -> None:
    ready, message = provider_ready(args)
    print(message, flush=True)
    if args.check_only:
        return
    if not ready:
        raise RuntimeError("Provider is not ready. Fix the setup issue above, then rerun this command.")

    cases = load_prompt_cases(source=args.dataset, limit=args.limit, csv_path=args.csv_path)
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    client = build_client(args)
    total_generations = len(scenarios) * args.windows * len(cases) * args.variants_per_case
    print(f"Capturing {total_generations} monitored-model responses...", flush=True)

    captured_rows = capture_responses(
        client=client,
        cases=cases,
        dataset_label=args.dataset,
        model=args.model,
        scenarios=scenarios,
        windows=args.windows,
        variants_per_case=args.variants_per_case,
        sleep_seconds=args.sleep_seconds,
        progress_callback=None if args.quiet else print_capture_progress,
    )
    write_captured_responses(args.responses_path, captured_rows)
    print(f"Saved raw monitored-model outputs to {args.responses_path}", flush=True)

    reloaded_rows = read_captured_response_rows(args.responses_path)
    analysis_rows = analyze_captured_rows(reloaded_rows)
    metric_rows = compute_metrics(analysis_rows)
    write_csv(args.results_path, analysis_rows)
    write_csv(args.metrics_path, metric_rows)

    print_results(analysis_rows, metric_rows, args.results_path, args.metrics_path)
    print("No LLM judge or answer verifier was used.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real monitored-model Sentinel experiment.")
    parser.add_argument("--provider", choices=["ollama", "openai-compatible", "fixture"], default="ollama")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dataset", choices=["sample", "truthfulqa", "gsm8k", "csv"], default="sample")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="prompt_corruption")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--variants-per-case", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--responses-path", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_real_model_experiment(parse_args())