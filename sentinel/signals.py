"""Reliability signal extraction."""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

from sentinel.embeddings import centroid, cosine_similarity, embed_text, tokenize
from sentinel.simulator import ModelResponse


HEDGING_TERMS = {
    "maybe",
    "probably",
    "might",
    "unknown",
    "uncertain",
    "complicated",
    "flexible",
    "incomplete",
    "paradox",
    "contradiction",
    "impossible",
    "loophole",
    "illogical",
    "undecidable",
    "loop",
}



def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def is_compliant(response: ModelResponse) -> bool:
    output = response.output.lower()


    too_short = len(tokenize(output)) < 8
    evasive = "not enough information" in output or "unknown" in output
    generic = "everyone knows" in output or "needs no context" in output
    if too_short or evasive or generic:
        return False


    q_lower = response.question.lower()
    match_len = re.search(r"exactly (\d+) words", q_lower)
    if match_len:
        target_len = int(match_len.group(1))
        actual_len = len(tokenize(output))
        if actual_len != target_len:
            return False


    match_letter = re.search(r"every(?: single)? word must start with the letter '([a-z])'", q_lower)
    if match_letter:
        target_letter = match_letter.group(1)
        tokens = tokenize(output)
        if not tokens:
            return False
        for token in tokens:
            if not token.startswith(target_letter):
                return False

    return True



def semantic_consistency_score(responses: list[ModelResponse]) -> float:
    by_case: dict[str, list[tuple[list[float], bool]]] = defaultdict(list)
    for response in responses:
        compliant = is_compliant(response)
        by_case[response.case_id].append((embed_text(response.output), compliant))

    similarities: list[float] = []
    for items in by_case.values():
        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                vec_l, comp_l = items[left_index]
                vec_r, comp_r = items[right_index]
                sim = cosine_similarity(vec_l, vec_r)

                if not (comp_l and comp_r):
                    sim = -1.0
                similarities.append(sim)

    return clamp((mean(similarities) + 1.0) / 2.0)


def response_stability_score(responses: list[ModelResponse]) -> float:
    by_case: dict[str, list[int]] = defaultdict(list)
    for response in responses:
        by_case[response.case_id].append(len(tokenize(response.output)))

    coefficients: list[float] = []
    for lengths in by_case.values():
        if len(lengths) <= 1:
            continue
        avg = statistics.mean(lengths)
        if avg == 0:
            continue
        coefficients.append(statistics.pstdev(lengths) / avg)

    return clamp(1.0 - mean(coefficients))


def embedding_drift_score(
    current_responses: list[ModelResponse],
    baseline_responses: list[ModelResponse],
) -> float:
    current_center = centroid([embed_text(item.output) for item in current_responses])
    baseline_center = centroid([embed_text(item.output) for item in baseline_responses])
    similarity = cosine_similarity(current_center, baseline_center)
    return clamp((similarity + 1.0) / 2.0)


def confidence_proxy_score(responses: list[ModelResponse]) -> float:
    token_count = 0
    hedge_count = 0
    for response in responses:
        tokens = tokenize(response.output)
        token_count += len(tokens)
        hedge_count += sum(1 for token in tokens if token in HEDGING_TERMS)

    if token_count == 0:
        return 0.0
    hedge_rate = hedge_count / token_count
    return clamp(1.0 - math.sqrt(hedge_rate * 8.0))


def task_compliance_score(responses: list[ModelResponse]) -> float:
    failures = sum(1 for response in responses if not is_compliant(response))
    return clamp(1.0 - (failures / len(responses))) if responses else 0.0



def extract_window_signals(
    current_responses: list[ModelResponse],
    baseline_responses: list[ModelResponse],
) -> dict[str, float]:
    return {
        "semantic_consistency": semantic_consistency_score(current_responses),
        "response_stability": response_stability_score(current_responses),
        "embedding_drift": embedding_drift_score(current_responses, baseline_responses),
        "confidence_proxy": confidence_proxy_score(current_responses),
        "task_compliance": task_compliance_score(current_responses),
    }
