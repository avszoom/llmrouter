"""Capability table loader and validator tests using the real seed CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from capability.loader import (
    DIFFICULTY_BUCKETS,
    DEFAULT_CSV,
    load_capability_table,
    validate_table,
)


REAL_CSV = Path(__file__).resolve().parent.parent / DEFAULT_CSV


@pytest.fixture(scope="module")
def real_models():
    return load_capability_table(REAL_CSV)


def test_loader_returns_ten_models(real_models):
    assert len(real_models) == 10


def test_each_model_has_ten_task_types(real_models):
    for m in real_models:
        tasks = {t for (t, _) in m.quality}
        assert len(tasks) == 10, f"{m.name} has {len(tasks)} task_types, expected 10"


def test_each_task_has_three_difficulty_buckets(real_models):
    for m in real_models:
        tasks = {t for (t, _) in m.quality}
        for t in tasks:
            buckets = {b for (tt, b) in m.quality if tt == t}
            assert buckets == set(DIFFICULTY_BUCKETS), (
                f"{m.name}/{t} has {buckets}, expected {DIFFICULTY_BUCKETS}"
            )


def test_real_csv_has_no_validation_issues(real_models):
    issues = validate_table(real_models)
    assert issues == [], "\n".join(issues)


def test_quality_lookup_returns_zero_for_unknown_combo(real_models):
    m = real_models[0]
    assert m.quality_for("nonexistent_task", "easy") == 0.0
    assert m.quality_for("coding", "ultra") == 0.0


def test_cost_calculation(real_models):
    m = next(m for m in real_models if m.name == "claude-haiku-4.5")
    # input=$1.0/1M, output=$5.0/1M
    cost = m.estimate_cost_usd(input_tokens=1000, output_tokens=500)
    assert cost == pytest.approx(0.001 + 0.0025)


def test_validate_catches_monotone_violation():
    from capability.loader import ModelCard
    bad = ModelCard(
        name="bad", provider="x",
        input_usd_per_1m=1.0, output_usd_per_1m=1.0,
        avg_latency_ms=1000, output_tps=100, notes="",
        quality={
            ("coding", "easy"): 0.50,
            ("coding", "medium"): 0.80,  # not monotone!
            ("coding", "hard"): 0.30,
        },
    )
    issues = validate_table([bad])
    assert any("not monotone" in s for s in issues)


def test_validate_catches_out_of_range():
    from capability.loader import ModelCard
    bad = ModelCard(
        name="bad", provider="x",
        input_usd_per_1m=1.0, output_usd_per_1m=1.0,
        avg_latency_ms=1000, output_tps=100, notes="",
        quality={
            ("coding", "easy"): 1.5,  # out of [0,1]
            ("coding", "medium"): 0.80,
            ("coding", "hard"): 0.30,
        },
    )
    issues = validate_table([bad])
    assert any("outside [0, 1]" in s for s in issues)
