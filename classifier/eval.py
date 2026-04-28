"""Evaluate the trained classifier.

- Held-out test set: report accuracy / F1 (classification) and MAE / MAPE / R² (regression).
- 5-fold cross-validation on the full dataset for an honest mean ± std on each output.

Usage:
    python -m classifier.eval --data llm_router_poc_spec_and_dataset/prompt_classifier_dataset_1000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .data import load_dataset
from .embed import Embedder
from .heads import HEADS


PASS_BAR = {
    "task_type": ("macro_f1", 0.80, ">="),
    "risk_level": ("macro_f1", 0.80, ">="),
    "latency_sensitivity": ("macro_f1", 0.80, ">="),
    "difficulty": ("mae", 0.10, "<="),
    "required_quality": ("mae", 0.10, "<="),
    "expected_output_tokens": ("mape_pct", 25.0, "<="),
}


def _check(name: str, metric_value: float) -> str:
    metric_name, threshold, op = PASS_BAR[name]
    if op == ">=":
        ok = metric_value >= threshold
    else:
        ok = metric_value <= threshold
    return "PASS" if ok else "FAIL"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained classifier.")
    ap.add_argument(
        "--data",
        default="llm_router_poc_spec_and_dataset/prompt_classifier_dataset_1000.jsonl",
        help="Full labeled dataset (used for cross-validation)",
    )
    ap.add_argument("--artifacts", default="classifier/artifacts")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    if not (artifacts / "heads").exists():
        raise SystemExit(f"No artifacts found in {artifacts}. Run `python -m classifier.train` first.")

    # 1) Held-out test evaluation using already-trained heads
    print("=" * 60)
    print("Held-out test set (1 split, deterministic seed=42)")
    print("=" * 60)
    test_df = pd.read_parquet(artifacts / "test_df.parquet")
    X_test = np.load(artifacts / "X_test.npy")
    print(f"Test rows: {len(test_df)}\n")

    test_summary = {}
    for name, spec in HEADS.items():
        head = joblib.load(artifacts / "heads" / f"{name}.joblib")
        if spec["kind"] == "classification":
            le = joblib.load(artifacts / "label_encoders" / f"{name}.joblib")
            y_true = le.transform(test_df[name])
            y_pred = head.predict(X_test)
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro")
            verdict = _check(name, macro_f1)
            test_summary[name] = {"accuracy": float(acc), "macro_f1": float(macro_f1), "verdict": verdict}
            print(f"  {name:<25} acc={acc:.3f}  macro_f1={macro_f1:.3f}  [{verdict}]")
        else:
            y_true = test_df[name].astype(float).values
            y_pred = head.predict(X_test)
            mae = float(np.mean(np.abs(y_pred - y_true)))
            mape_pct = float(
                np.mean(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), 1e-3, None))) * 100
            )
            denom = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else 0.0
            metric_for_bar = mape_pct if PASS_BAR[name][0] == "mape_pct" else mae
            verdict = _check(name, metric_for_bar)
            test_summary[name] = {
                "mae": mae,
                "mape_pct": mape_pct,
                "r2": r2,
                "verdict": verdict,
            }
            print(f"  {name:<25} MAE={mae:.4f}  MAPE={mape_pct:5.1f}%  R²={r2:+.3f}  [{verdict}]")

    # 2) k-fold cross-validation on the full dataset
    print("\n" + "=" * 60)
    print(f"{args.folds}-fold cross-validation on full dataset")
    print("=" * 60)
    df = load_dataset(args.data)
    print(f"Embedding {len(df)} prompts (one pass)...")
    embedder = Embedder()
    X = embedder.embed(df["prompt"].tolist())
    print()

    cv_summary = {}
    for name, spec in HEADS.items():
        scores = []
        if spec["kind"] == "classification":
            le = LabelEncoder()
            y = le.fit_transform(df[name])
            kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
            for tr_idx, va_idx in kf.split(X, y):
                model = spec["make"]()
                model.fit(X[tr_idx], y[tr_idx])
                y_pred = model.predict(X[va_idx])
                scores.append(f1_score(y[va_idx], y_pred, average="macro"))
            mean, std = float(np.mean(scores)), float(np.std(scores))
            cv_summary[name] = {
                "metric": "macro_f1",
                "mean": mean,
                "std": std,
                "folds": [float(s) for s in scores],
            }
            folds_str = ", ".join(f"{s:.3f}" for s in scores)
            print(f"  {name:<25} macro_f1 = {mean:.3f} ± {std:.3f}   folds=[{folds_str}]")
        else:
            y = df[name].astype(float).values
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
            for tr_idx, va_idx in kf.split(X):
                model = spec["make"]()
                model.fit(X[tr_idx], y[tr_idx])
                y_pred = model.predict(X[va_idx])
                scores.append(np.mean(np.abs(y_pred - y[va_idx])))
            mean, std = float(np.mean(scores)), float(np.std(scores))
            cv_summary[name] = {
                "metric": "mae",
                "mean": mean,
                "std": std,
                "folds": [float(s) for s in scores],
            }
            folds_str = ", ".join(f"{s:.4f}" for s in scores)
            print(f"  {name:<25} MAE      = {mean:.4f} ± {std:.4f} folds=[{folds_str}]")

    report = {"test": test_summary, "cv": cv_summary, "pass_bar": PASS_BAR}
    (artifacts / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull report → {artifacts}/eval_report.json")


if __name__ == "__main__":
    main()
