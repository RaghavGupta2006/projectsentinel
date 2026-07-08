"""Unit tests for Sentinel scoring, signals, and Z-score anomaly detection logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force offline hashed embeddings for fast testing
import os
os.environ["SENTINEL_EMBEDDINGS"] = "hashed"

from sentinel.simulator import ModelResponse
from sentinel.scoring import detect_degradation, score_window
from sentinel.signals import is_compliant, semantic_consistency_score, task_compliance_score


class TestScoringLogic(unittest.TestCase):

    def test_identical_responses_consistency(self):
        """Feeding identical compliant responses should yield a consistency score of 1.0 (max)."""
        responses = [
            ModelResponse(window=1, case_id="case_1", question="What is health?", variant=0, degradation="none", output="Health is a state of complete physical and mental well-being."),
            ModelResponse(window=1, case_id="case_1", question="What is health?", variant=1, degradation="none", output="Health is a state of complete physical and mental well-being.")
        ]
        score = semantic_consistency_score(responses)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_unrelated_responses_consistency(self):
        """Feeding completely unrelated words should yield a low consistency score."""
        responses = [
            ModelResponse(window=1, case_id="case_1", question="Test?", variant=0, degradation="none", output="apple banana cherry grape orange lemon strawberry blueberry"),
            ModelResponse(window=1, case_id="case_1", question="Test?", variant=1, degradation="none", output="computer motherboard monitor keyboard mouse printer scanner router")
        ]
        score = semantic_consistency_score(responses)
        # Hashed BoW cosine similarity of completely disjoint word lists is 0.0,
        # which scales to (0.0 + 1.0) / 2.0 = 0.5 consistency.
        self.assertLess(score, 0.6)

    def test_non_compliant_responses_penalization(self):
        """Non-compliant responses (e.g. failing formatting rules) should be heavily penalized to 0.0 consistency."""
        responses = [
            ModelResponse(window=1, case_id="case_1", question="exactly 2 words please", variant=0, degradation="none", output="This is a long non-compliant answer that fails word limit constraints."),
            ModelResponse(window=1, case_id="case_1", question="exactly 2 words please", variant=1, degradation="none", output="This is a long non-compliant answer that fails word limit constraints.")
        ]
        # Although the texts are identical, they both fail the 'exactly 2 words' constraint.
        # Pairwise similarity should be penalized to -1.0, scaling to 0.0 consistency score.
        score = semantic_consistency_score(responses)
        self.assertEqual(score, 0.0)

    def test_is_compliant_edge_cases(self):
        """Test compliance rules on various formatting constraints and edge cases."""
        # Standard compliant text
        r_ok = ModelResponse(1, "c1", "Normal QA query?", 0, "none", "This is a detailed and compliant response that is long enough.")
        self.assertTrue(is_compliant(r_ok))

        # Too short (< 8 words)
        r_short = ModelResponse(1, "c1", "Normal QA query?", 0, "none", "Too short.")
        self.assertFalse(is_compliant(r_short))

        # Evasive phrase
        r_evasive = ModelResponse(1, "c1", "Normal QA", 0, "none", "I have unknown not enough information to answer.")
        self.assertFalse(is_compliant(r_evasive))

        # Dynamic exact word count constraint
        r_word_ok = ModelResponse(1, "c1", "Answer in exactly 2 words.", 0, "none", "Hello world")
        r_word_fail = ModelResponse(1, "c1", "Answer in exactly 2 words.", 0, "none", "Hello world this is a test")
        self.assertTrue(is_compliant(r_word_ok))
        self.assertFalse(is_compliant(r_word_fail))

        # Dynamic start-letter constraint
        r_letter_ok = ModelResponse(1, "c1", "every single word must start with the letter 'd'", 0, "none", "dog dig day dug dog dig day dug")
        r_letter_fail = ModelResponse(1, "c1", "every single word must start with the letter 'd'", 0, "none", "dog cat day dug dog dig day dug")
        self.assertTrue(is_compliant(r_letter_ok))
        self.assertFalse(is_compliant(r_letter_fail))

    def test_zscore_anomaly_detection(self):
        """Verify that rolling Z-score flags statistical outliers but ignores normal noise."""
        baseline_score = 0.95
        
        # 1. Normal noise: scores fluctuate slightly around 0.95. Z-score should not trigger.
        history_normal = [0.95, 0.96, 0.94, 0.95]
        alert_normal = detect_degradation(
            reliability_score=0.945,
            baseline_score=baseline_score,
            history_scores=history_normal,
            zscore_k=2.0
        )
        self.assertFalse(alert_normal)

        # 2. Obvious outlier: a sudden drop to 0.75. Z-score should flag it immediately.
        # History mean is ~0.95, std dev is very small (~0.007). Drop is >20 standard deviations.
        history_outlier = [0.95, 0.96, 0.95, 0.94]
        alert_outlier = detect_degradation(
            reliability_score=0.75,
            baseline_score=baseline_score,
            history_scores=history_outlier,
            zscore_k=2.0
        )
        self.assertTrue(alert_outlier)

if __name__ == "__main__":
    unittest.main()
