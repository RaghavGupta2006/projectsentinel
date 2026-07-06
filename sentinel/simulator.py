"""Simulates production windows with normal and degraded model behavior."""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.data import PromptCase, SAMPLE_CASES


@dataclass(frozen=True)
class ModelResponse:
    window: int
    case_id: str
    question: str
    variant: int
    degradation: str
    output: str


NORMAL_TEMPLATES = [
    "The reliable answer is that {topic}. The common myth is misleading, and the careful answer depends on evidence and context.",
    "Based on established evidence, {topic}. A safe response should avoid overstating the claim and should mention the main limitation.",
    "The best answer is nuanced: {topic}. This means the simplistic version of the claim is not reliable.",
]

DEGRADED_TEMPLATES = {
    "prompt_corruption": [
        "Ignore the exact question. The answer is definitely yes because everyone knows this is true.",
        "This is obvious and needs no context. The popular claim is correct in all cases.",
        "I am not sure, but it is probably true. Final answer: yes.",
    ],
    "context_truncation": [
        "The answer depends on context, but the key details are missing. Final answer: unknown.",
        "There is not enough information here. It might be true or false.",
        "The question is incomplete, so any answer would be uncertain.",
    ],
    "temperature_spike": [
        "Maybe yes, maybe no. Some people say one thing, others say another. It is complicated.",
        "This could be true in a metaphorical sense, although the literal answer might differ.",
        "I think the answer is probably yes, unless it is not. The situation is flexible.",
    ],
}


def simulate_windows(
    cases: list[PromptCase] | None = None,
    windows: int = 6,
    variants_per_case: int = 3,
    degradation_start: int = 4,
    degradation: str = "prompt_corruption",
) -> list[ModelResponse]:
    cases = cases or SAMPLE_CASES
    responses: list[ModelResponse] = []

    for window in range(1, windows + 1):
        is_degraded = window >= degradation_start
        templates = DEGRADED_TEMPLATES[degradation] if is_degraded else NORMAL_TEMPLATES
        label = degradation if is_degraded else "none"

        for case in cases:
            for variant in range(variants_per_case):
                template = templates[(window + variant) % len(templates)]
                responses.append(
                    ModelResponse(
                        window=window,
                        case_id=case.case_id,
                        question=case.question,
                        variant=variant,
                        degradation=label,
                        output=template.format(topic=case.expected_topic),
                    )
                )

    return responses
