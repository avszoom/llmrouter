"""Routing rules — filter, policy selection, fallback."""

from __future__ import annotations

import pytest

from router.engine import Objective, route


# ─── easy prompt: all 3 eligible (quality bar 0.80) ──────────────────────


def test_cost_first_picks_cheapest_eligible(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="cost_first"), input_tokens=100)
    assert decision.selected_model == "cheap-fast"
    assert decision.eligible == ["cheap-fast", "mid", "premium"]
    assert not decision.fallback
    assert decision.rejected == {}


def test_latency_first_picks_fastest_eligible(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="latency_first"), input_tokens=100)
    assert decision.selected_model == "cheap-fast"
    # cheap-fast (400) < mid (1500) < premium (3000)
    assert decision.eligible[0] == "cheap-fast"


def test_quality_first_picks_highest_quality_eligible(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="quality_first"), input_tokens=100)
    assert decision.selected_model == "premium"
    assert decision.eligible[0] == "premium"


def test_balanced_picks_score_winner(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="balanced"), input_tokens=100)
    # All are eligible; balanced rewards quality but penalizes cost & latency.
    # cheap-fast: 0.55*0.85 - 0.25*(0.005/0.066) - 0.20*(400/3000) = 0.467 - 0.019 - 0.027 = 0.421
    # mid:       0.55*0.92 - 0.25*(0.0061/0.066) - 0.20*(1500/3000) = 0.506 - 0.023 - 0.10 = 0.383
    # premium:   0.55*0.97 - 0.25*1.0 - 0.20*1.0 = 0.534 - 0.25 - 0.20 = 0.084
    # → cheap-fast wins (highest balanced score on cheap inputs)
    assert decision.selected_model == "cheap-fast"


# ─── hard prompt: only 'premium' clears 0.90 on coding/hard ──────────────


def test_hard_prompt_filters_cheap_models(models, cls_hard):
    decision = route(cls_hard, models, Objective(policy="cost_first"), input_tokens=100)
    # Only premium (quality_hard=0.91) clears 0.90
    assert decision.selected_model == "premium"
    assert decision.eligible == ["premium"]
    assert "cheap-fast" in decision.rejected
    assert "mid" in decision.rejected
    assert "0.55" in decision.rejected["cheap-fast"]
    assert "0.78" in decision.rejected["mid"]


def test_hard_prompt_quality_first_same_winner(models, cls_hard):
    decision = route(cls_hard, models, Objective(policy="quality_first"), input_tokens=100)
    assert decision.selected_model == "premium"


# ─── SLA filter ─────────────────────────────────────────────────────────


def test_sla_excludes_slow_models(models, cls_easy):
    # Only cheap-fast (400ms) clears a 500ms SLA
    decision = route(
        cls_easy, models,
        Objective(policy="quality_first", latency_sla_ms=500),
        input_tokens=100,
    )
    assert decision.selected_model == "cheap-fast"
    assert decision.eligible == ["cheap-fast"]
    assert "latency=1500ms" in decision.rejected["mid"]
    assert "latency=3000ms" in decision.rejected["premium"]


def test_sla_combined_with_quality_filter(models, cls_hard):
    # Hard prompt needs 0.90 quality; only premium qualifies, but premium violates 1000ms SLA
    decision = route(
        cls_hard, models,
        Objective(policy="cost_first", latency_sla_ms=1000),
        input_tokens=100,
    )
    assert decision.fallback is True
    # Fallback picks the highest-quality model regardless of SLA
    assert decision.selected_model == "premium"


# ─── unmeetable required_quality → fallback path ─────────────────────────


def test_no_eligible_triggers_fallback(models, cls_unmeetable):
    decision = route(cls_unmeetable, models, Objective(policy="cost_first"), input_tokens=100)
    assert decision.fallback is True
    assert decision.selected_model == "premium"  # highest quality_hard = 0.91
    assert decision.eligible == []
    assert "FALLBACK" in decision.routing_reason
    assert len(decision.rejected) == 3


# ─── cost calculation sanity ────────────────────────────────────────────


def test_cost_calculation_matches_pricing(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="cost_first"), input_tokens=1000)
    # cheap-fast: in=$0.10/1M, out=$0.30/1M; expected_output=200
    expected = 1000 * 0.10 / 1_000_000 + 200 * 0.30 / 1_000_000
    assert decision.estimated_cost_usd == pytest.approx(expected)


def test_decision_carries_token_counts(models, cls_easy):
    decision = route(cls_easy, models, Objective(policy="balanced"), input_tokens=42)
    assert decision.estimated_input_tokens == 42
    assert decision.estimated_output_tokens == 200


# ─── invariants ─────────────────────────────────────────────────────────


def test_all_policies_pick_eligible_when_any_exist(models, cls_easy):
    for pol in ("cost_first", "latency_first", "quality_first", "balanced"):
        decision = route(cls_easy, models, Objective(policy=pol), input_tokens=100)
        assert decision.fallback is False
        assert decision.selected_model in decision.eligible


def test_unknown_policy_raises(models, cls_easy):
    with pytest.raises(ValueError, match="Unknown policy"):
        route(cls_easy, models, Objective(policy="random"), input_tokens=100)  # type: ignore[arg-type]
