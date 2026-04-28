"""Capability table loader.

Reads the long-format CSV (one row per model × task_type) and builds a list of
ModelCard objects with a fast (task_type, difficulty_bucket) → quality lookup.

CLI:
    python -m capability.loader --validate
        prints any range or monotonicity violations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_CSV = "llm_router_poc_spec_and_dataset/model_capability_table_seed.csv"
DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


@dataclass
class ModelCard:
    name: str
    provider: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    avg_latency_ms: int
    output_tps: int
    notes: str
    quality: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def quality_for(self, task_type: str, difficulty_bucket: str) -> float:
        return self.quality.get((task_type, difficulty_bucket), 0.0)

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_1m / 1_000_000
            + output_tokens * self.output_usd_per_1m / 1_000_000
        )


def load_capability_table(path: str | Path = DEFAULT_CSV) -> List[ModelCard]:
    df = pd.read_csv(path)
    required = {
        "model", "provider",
        "input_usd_per_1m", "output_usd_per_1m",
        "avg_latency_ms_seed", "output_tps_seed",
        "task_type", "quality_easy", "quality_medium", "quality_hard", "notes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    by_name: Dict[str, ModelCard] = {}
    for _, row in df.iterrows():
        name = str(row["model"])
        if name not in by_name:
            by_name[name] = ModelCard(
                name=name,
                provider=str(row["provider"]),
                input_usd_per_1m=float(row["input_usd_per_1m"]),
                output_usd_per_1m=float(row["output_usd_per_1m"]),
                avg_latency_ms=int(row["avg_latency_ms_seed"]),
                output_tps=int(row["output_tps_seed"]),
                notes=str(row["notes"]),
            )
        task = str(row["task_type"])
        by_name[name].quality[(task, "easy")] = float(row["quality_easy"])
        by_name[name].quality[(task, "medium")] = float(row["quality_medium"])
        by_name[name].quality[(task, "hard")] = float(row["quality_hard"])
    return list(by_name.values())


def validate_table(models: List[ModelCard]) -> List[str]:
    """Return a list of issue strings; empty list means valid."""
    issues: List[str] = []
    for m in models:
        # Range check
        for (task, bucket), q in m.quality.items():
            if not 0.0 <= q <= 1.0:
                issues.append(f"{m.name}/{task}/{bucket}: quality {q} outside [0, 1]")
        # Monotonicity easy >= medium >= hard
        tasks = {t for (t, _) in m.quality}
        for t in tasks:
            e = m.quality.get((t, "easy"))
            md = m.quality.get((t, "medium"))
            h = m.quality.get((t, "hard"))
            if None in (e, md, h):
                issues.append(f"{m.name}/{t}: missing one of easy/medium/hard")
                continue
            if not (e >= md >= h):
                issues.append(
                    f"{m.name}/{t}: not monotone (easy={e}, medium={md}, hard={h})"
                )
        # Pricing sanity
        if m.input_usd_per_1m < 0 or m.output_usd_per_1m < 0:
            issues.append(f"{m.name}: negative pricing")
        if m.avg_latency_ms <= 0:
            issues.append(f"{m.name}: non-positive latency {m.avg_latency_ms}")
    return issues


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Load and validate the capability table.")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--validate", action="store_true", help="Run validation and exit nonzero if issues found")
    args = ap.parse_args()

    models = load_capability_table(args.csv)
    print(f"Loaded {len(models)} models from {args.csv}")
    for m in models:
        print(f"  {m.name:<25} {m.provider:<18} ${m.input_usd_per_1m:.3f}/${m.output_usd_per_1m:.3f}/1M  {m.avg_latency_ms}ms  ({len(m.quality)} quality entries)")

    if args.validate:
        issues = validate_table(models)
        print(f"\nValidation: {'OK' if not issues else f'{len(issues)} issues'}")
        for issue in issues:
            print(f"  - {issue}")
        if issues:
            raise SystemExit(1)


if __name__ == "__main__":
    _cli()
