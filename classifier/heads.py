"""Per-output head specifications.

Each head is a small sklearn model that reads a 384-dim embedding and
predicts one classifier output. LogisticRegression for categorical labels;
GradientBoostingRegressor for continuous targets; RandomForestRegressor for
expected_output_tokens (heavier-tailed, benefits from bagging).
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, TypedDict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


class HeadSpec(TypedDict):
    kind: Literal["classification", "regression"]
    make: Callable[[], object]


HEADS: Dict[str, HeadSpec] = {
    "task_type": {
        "kind": "classification",
        "make": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    },
    "difficulty": {
        "kind": "regression",
        "make": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    },
    "required_quality": {
        "kind": "regression",
        "make": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    },
    "risk_level": {
        "kind": "classification",
        "make": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    },
    "expected_output_tokens": {
        "kind": "regression",
        "make": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
    },
    "latency_sensitivity": {
        "kind": "classification",
        "make": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    },
}
