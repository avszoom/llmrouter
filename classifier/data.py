"""Dataset loading and stratified train/val/test splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = (
    "prompt",
    "task_type",
    "difficulty",
    "difficulty_bucket",
    "required_quality",
    "risk_level",
    "expected_output_tokens",
    "latency_sensitivity",
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num} invalid JSON: {exc}") from exc
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_col: str = "task_type",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("split sizes must sum to 1.0")
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df[stratify_col],
        random_state=seed,
    )
    val_fraction_of_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_fraction_of_temp,
        stratify=temp_df[stratify_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
