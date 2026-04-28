"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable in tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest

from capability.loader import ModelCard


def _card(
    name: str,
    inp: float,
    outp: float,
    latency_ms: int,
    quality_easy: float,
    quality_medium: float,
    quality_hard: float,
    task: str = "coding",
) -> ModelCard:
    return ModelCard(
        name=name,
        provider="test",
        input_usd_per_1m=inp,
        output_usd_per_1m=outp,
        avg_latency_ms=latency_ms,
        output_tps=100,
        notes="",
        quality={
            (task, "easy"): quality_easy,
            (task, "medium"): quality_medium,
            (task, "hard"): quality_hard,
        },
    )


@pytest.fixture
def models():
    """A small synthetic fleet for routing tests.

    All have coding quality on (easy, medium, hard); designed so each policy
    has an obvious winner.
    """
    return [
        # cheap & fast but mediocre quality
        _card("cheap-fast",  inp=0.10, outp=0.30, latency_ms=400,  quality_easy=0.85, quality_medium=0.75, quality_hard=0.55),
        # mid cost, mid latency, good quality
        _card("mid",         inp=1.00, outp=3.00, latency_ms=1500, quality_easy=0.92, quality_medium=0.86, quality_hard=0.78),
        # expensive but best quality, slowest
        _card("premium",     inp=5.00, outp=15.0, latency_ms=3000, quality_easy=0.97, quality_medium=0.94, quality_hard=0.91),
    ]


@pytest.fixture
def cls_easy():
    return {
        "task_type": "coding",
        "difficulty": 0.20,
        "difficulty_bucket": "easy",
        "required_quality": 0.80,
        "risk_level": "low",
        "expected_output_tokens": 200,
        "latency_sensitivity": "high",
    }


@pytest.fixture
def cls_hard():
    return {
        "task_type": "coding",
        "difficulty": 0.85,
        "difficulty_bucket": "hard",
        "required_quality": 0.90,
        "risk_level": "high",
        "expected_output_tokens": 1000,
        "latency_sensitivity": "low",
    }


@pytest.fixture
def cls_unmeetable():
    """No model can clear quality 0.99."""
    return {
        "task_type": "coding",
        "difficulty": 0.85,
        "difficulty_bucket": "hard",
        "required_quality": 0.99,
        "risk_level": "high",
        "expected_output_tokens": 500,
        "latency_sensitivity": "low",
    }
