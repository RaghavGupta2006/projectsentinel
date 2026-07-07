"""Semantic Reliability Score computation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityWeights:
    semantic_consistency: float = 0.25
    response_stability: float = 0.15
    embedding_drift: float = 0.25
    confidence_proxy: float = 0.15
    task_compliance: float = 0.20


def score_window(
    signals: dict[str, float],
    weights: ReliabilityWeights | None = None,
) -> float:
    weights = weights or ReliabilityWeights()
    weighted_sum = (
        weights.semantic_consistency * signals["semantic_consistency"]
        + weights.response_stability * signals["response_stability"]
        + weights.embedding_drift * signals["embedding_drift"]
        + weights.confidence_proxy * signals["confidence_proxy"]
        + weights.task_compliance * signals["task_compliance"]
    )
    return round(weighted_sum, 4)


def detect_degradation(
    reliability_score: float,
    baseline_score: float,
    absolute_threshold: float = 0.70,
    relative_drop_threshold: float = 0.15,
    history_scores: list[float] | None = None,
    zscore_k: float | None = None,
) -> bool:
    """Detect reliability degradation using thresholding or rolling z-scores.

    If zscore_k is provided and history_scores has enough data points (>= 2),
    it flags an anomaly if the current score is zscore_k standard deviations
    below the historical mean. Otherwise, it falls back to absolute thresholding
    and relative drop checks.
    """
    if zscore_k is not None and history_scores and len(history_scores) >= 2:
        import statistics

        mean_val = statistics.mean(history_scores)
        std_val = statistics.pstdev(history_scores)
        if std_val > 0.0001:
            z_score = (reliability_score - mean_val) / std_val
            if z_score < -zscore_k:
                return True

    relative_drop = baseline_score - reliability_score
    return reliability_score < absolute_threshold or relative_drop >= relative_drop_threshold