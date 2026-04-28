"""Rule-based routing engine.

Given a classifier output and the capability table, picks the best model under
a chosen policy (cost / latency / quality / balanced).

Pipeline:
    1. filter — drop models that don't clear required_quality (and SLA, if set)
    2. score — order remaining models by the chosen policy
    3. emit  — RoutingDecision with selected model, estimates, and reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from capability.loader import ModelCard

Policy = Literal["cost_first", "latency_first", "quality_first", "balanced"]
ALL_POLICIES: Tuple[Policy, ...] = ("cost_first", "latency_first", "quality_first", "balanced")

# Balanced-policy weights (Q minus normalized C and L)
BAL_W_QUALITY = 0.55
BAL_W_COST = 0.25
BAL_W_LATENCY = 0.20


@dataclass
class Objective:
    policy: Policy = "balanced"
    latency_sla_ms: Optional[int] = None


@dataclass
class RoutingDecision:
    selected_model: str
    provider: str
    policy: Policy
    estimated_quality: float
    estimated_cost_usd: float
    estimated_latency_ms: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    routing_reason: str
    eligible: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    fallback: bool = False


def count_input_tokens(prompt: str) -> int:
    """Token count via tiktoken cl100k_base; falls back to whitespace if tiktoken absent."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt))
    except Exception:
        return max(1, len(prompt.split()))


def _balanced_score(quality: float, cost: float, latency: float,
                    cost_max: float, latency_max: float) -> float:
    cost_norm = cost / cost_max if cost_max > 0 else 0.0
    lat_norm = latency / latency_max if latency_max > 0 else 0.0
    return BAL_W_QUALITY * quality - BAL_W_COST * cost_norm - BAL_W_LATENCY * lat_norm


def route(
    classifier_output: dict,
    models: List[ModelCard],
    objective: Objective,
    input_tokens: int,
) -> RoutingDecision:
    task_type: str = classifier_output["task_type"]
    difficulty_bucket: str = classifier_output["difficulty_bucket"]
    required_quality: float = float(classifier_output["required_quality"])
    expected_output_tokens: int = int(classifier_output["expected_output_tokens"])

    # Step 1 — filter
    eligible: List[Tuple[ModelCard, float, float, int]] = []
    rejected: Dict[str, str] = {}
    for m in models:
        q = m.quality_for(task_type, difficulty_bucket)
        cost = m.estimate_cost_usd(input_tokens, expected_output_tokens)
        latency = m.avg_latency_ms

        if q < required_quality:
            rejected[m.name] = (
                f"quality={q:.2f} < required={required_quality:.2f} "
                f"on {task_type}/{difficulty_bucket}"
            )
            continue
        if objective.latency_sla_ms is not None and latency > objective.latency_sla_ms:
            rejected[m.name] = f"latency={latency}ms > SLA={objective.latency_sla_ms}ms"
            continue
        eligible.append((m, q, cost, latency))

    # Step 1b — fallback if nothing eligible
    if not eligible:
        # Pick highest-quality model (regardless of bar) so the caller still gets *some* answer
        best = max(models, key=lambda m: m.quality_for(task_type, difficulty_bucket))
        best_q = best.quality_for(task_type, difficulty_bucket)
        cost = best.estimate_cost_usd(input_tokens, expected_output_tokens)
        return RoutingDecision(
            selected_model=best.name,
            provider=best.provider,
            policy=objective.policy,
            estimated_quality=best_q,
            estimated_cost_usd=cost,
            estimated_latency_ms=best.avg_latency_ms,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=expected_output_tokens,
            routing_reason=(
                f"FALLBACK: no model met required_quality={required_quality:.2f} "
                f"on {task_type}/{difficulty_bucket}; picked highest-quality available "
                f"({best.name}, q={best_q:.2f}). Consider lowering the bar or expanding the fleet."
            ),
            eligible=[],
            rejected=rejected,
            fallback=True,
        )

    # Step 2 — score by policy
    if objective.policy == "cost_first":
        eligible.sort(key=lambda t: t[2])
        reason = f"cost_first: cheapest of {len(eligible)} eligible (quality ≥ {required_quality:.2f})"
    elif objective.policy == "latency_first":
        eligible.sort(key=lambda t: t[3])
        reason = f"latency_first: fastest of {len(eligible)} eligible (quality ≥ {required_quality:.2f})"
    elif objective.policy == "quality_first":
        eligible.sort(key=lambda t: t[1], reverse=True)
        reason = f"quality_first: highest-quality of {len(eligible)} eligible"
    elif objective.policy == "balanced":
        cost_max = max(c for _, _, c, _ in eligible)
        lat_max = max(l for _, _, _, l in eligible)
        scored = [
            (m, q, c, l, _balanced_score(q, c, l, cost_max, lat_max))
            for (m, q, c, l) in eligible
        ]
        scored.sort(key=lambda t: t[4], reverse=True)
        eligible = [(m, q, c, l) for (m, q, c, l, _) in scored]
        reason = (
            f"balanced (Q×{BAL_W_QUALITY} − Cnorm×{BAL_W_COST} − Lnorm×{BAL_W_LATENCY}): "
            f"top of {len(eligible)} eligible"
        )
    else:
        raise ValueError(f"Unknown policy: {objective.policy}")

    chosen, q, cost, latency = eligible[0]
    return RoutingDecision(
        selected_model=chosen.name,
        provider=chosen.provider,
        policy=objective.policy,
        estimated_quality=q,
        estimated_cost_usd=cost,
        estimated_latency_ms=latency,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=expected_output_tokens,
        routing_reason=reason,
        eligible=[m.name for (m, _, _, _) in eligible],
        rejected=rejected,
        fallback=False,
    )


def route_prompt(
    classifier,
    models: List[ModelCard],
    prompt: str,
    objective: Objective,
) -> Tuple[dict, RoutingDecision]:
    """End-to-end: classify the prompt, count tokens, route. Returns both."""
    classifier_output = classifier.predict(prompt)
    input_tokens = count_input_tokens(prompt)
    decision = route(classifier_output, models, objective, input_tokens)
    return classifier_output, decision
