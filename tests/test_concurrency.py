"""Concurrency, race condition, and failure handling tests for the capture responses harness."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.data import PromptCase
from experiments.capture_responses import capture_responses, ModelClient

class MockModelClient:
    """Mock LLM client that returns mock text or throws exceptions to test safety."""
    provider_name = "mock"

    def __init__(self, failure_case_id: str | None = None, delay: float = 0.0) -> None:
        self.failure_case_id = failure_case_id
        self.delay = delay

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        # Simulate network delay if specified
        if self.delay > 0:
            time.sleep(self.delay)

        # Trigger mock timeout/error if this is the target failure case
        if self.failure_case_id and self.failure_case_id in user_prompt:
            raise RuntimeError("Mock network timeout communicating with Ollama!")
            
        return f"Mock response content for prompt: {user_prompt[:30]}"

class TestConcurrencyHarness(unittest.TestCase):

    def setUp(self) -> None:
        self.cases = [
            PromptCase("c1", "What is the capital of France?", "capital France"),
            PromptCase("c2", "Does knuckle cracking cause arthritis?", "knuckles arthritis"),
            PromptCase("c3", "Is the Great Wall of China visible from space?", "great wall space"),
            PromptCase("c4", "Write a short paragraph about dogs.", "dog topic"),
        ]
        self.scenarios = ["prompt_corruption"]
        self.windows = 2
        self.variants = 2
        # Total expected outputs = 1 scenario * 2 windows * 4 cases * 2 variants = 16 outputs

    def test_no_race_conditions_multithreaded(self):
        """Harness should complete all tasks without drops, duplicates, or row corruption under 4 threads."""
        client = MockModelClient()
        rows = capture_responses(
            client=client,
            cases=self.cases,
            dataset_label="test",
            model="mock-model",
            scenarios=self.scenarios,
            windows=self.windows,
            variants_per_case=self.variants,
            workers=4
        )
        
        # 1. Assert no rows were dropped
        expected_total = len(self.scenarios) * self.windows * len(self.cases) * self.variants
        self.assertEqual(len(rows), expected_total)

        # 2. Assert no duplicate variants were created (every row should have a unique combination of keys)
        seen_keys = set()
        for r in rows:
            key = (r["scenario"], r["window"], r["case_id"], r["variant"])
            self.assertNotIn(key, seen_keys, f"Duplicate variant row detected: {key}")
            seen_keys.add(key)

        # 3. Assert contents aren't corrupted or mixed up
        for r in rows:
            self.assertIn("Mock response content", str(r["output"]))

    def test_deterministic_outputs(self):
        """Harness runs should be completely deterministic, producing identical results across runs regardless of scheduling."""
        client = MockModelClient()
        
        # Run 1
        rows_1 = capture_responses(
            client=client,
            cases=self.cases,
            dataset_label="test",
            model="mock-model",
            scenarios=self.scenarios,
            windows=self.windows,
            variants_per_case=self.variants,
            workers=4
        )
        
        # Run 2
        rows_2 = capture_responses(
            client=client,
            cases=self.cases,
            dataset_label="test",
            model="mock-model",
            scenarios=self.scenarios,
            windows=self.windows,
            variants_per_case=self.variants,
            workers=4
        )

        # Sort both runs to ensure index-independent comparison
        sort_key = lambda x: (x["scenario"], x["window"], x["case_id"], x["variant"])
        sorted_1 = sorted(rows_1, key=sort_key)
        sorted_2 = sorted(rows_2, key=sort_key)

        self.assertEqual(len(sorted_1), len(sorted_2))
        for r1, r2 in zip(sorted_1, sorted_2):
            self.assertEqual(r1["output"], r2["output"])
            self.assertEqual(r1["degradation"], r2["degradation"])

    def test_graceful_failure_handling(self):
        """Harness should degrade gracefully: if one thread throws an exception, other queries should still complete."""
        # Setup client to crash ONLY on the prompt for case "c2" ("Does knuckle cracking...")
        client = MockModelClient(failure_case_id="knuckle cracking")
        
        # Run with 4 threads. One task (c2, which has 1 scenario * 2 windows * 2 variants = 4 rows) will fail.
        # The remaining tasks (c1, c3, c4 = 12 rows) should complete successfully without the whole script crashing!
        rows = capture_responses(
            client=client,
            cases=self.cases,
            dataset_label="test",
            model="mock-model",
            scenarios=self.scenarios,
            windows=self.windows,
            variants_per_case=self.variants,
            workers=4
        )

        # Assert failed case rows are excluded, but successful ones are preserved
        expected_success_count = 12  # (4 cases - 1 failing case) * 2 windows * 2 variants
        self.assertEqual(len(rows), expected_success_count)

        # Ensure no c2 rows are in the outputs, but c1, c3, c4 are
        for r in rows:
            self.assertNotEqual(r["case_id"], "c2", "Failing case c2 should have been safely skipped.")
            self.assertIn(r["case_id"], ["c1", "c3", "c4"])

if __name__ == "__main__":
    unittest.main()
