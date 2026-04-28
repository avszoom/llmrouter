"""Train the prompt classifier: embed prompts, fit 6 sklearn heads, save artifacts.

Usage:
    python -m classifier.train --data llm_router_poc_spec_and_dataset/prompt_classifier_dataset_1000.jsonl

Programmatic use (e.g. Streamlit cold-start bootstrap):
    from classifier.train import train_classifier
    train_classifier(data_path=..., artifacts_dir=..., log=lambda m: ...)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .data import load_dataset, stratified_split
from .embed import Embedder
from .heads import HEADS


def train_classifier(
    data_path: str | Path,
    artifacts_dir: str | Path = "classifier/artifacts",
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Train the multi-output classifier and persist artifacts.

    `log` is the sink for progress messages — defaults to `print`. The Streamlit
    UI passes a closure that writes into a `st.status` widget instead.

    Returns a dict with `meta` (saved alongside the artifacts) and `summary`
    (per-head validation metrics).
    """
    artifacts = Path(artifacts_dir)
    (artifacts / "heads").mkdir(parents=True, exist_ok=True)
    (artifacts / "label_encoders").mkdir(parents=True, exist_ok=True)

    log(f"[1/5] Loading dataset: {data_path}")
    df = load_dataset(data_path)
    log(f"      → {len(df)} rows | columns: {len(df.columns)}")

    log(f"[2/5] Stratified split 80/10/10 (stratify=task_type, seed={seed})")
    train_df, val_df, test_df = stratified_split(df, seed=seed)
    log(f"      → train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    log("[3/5] Loading embedding model: all-MiniLM-L6-v2")
    t0 = time.time()
    embedder = Embedder()
    log(f"      → loaded in {time.time() - t0:.1f}s | dim={embedder.dim}")

    log("[4/5] Embedding prompts (batched)")
    t0 = time.time()
    X_train = embedder.embed(train_df["prompt"].tolist())
    X_val = embedder.embed(val_df["prompt"].tolist())
    X_test = embedder.embed(test_df["prompt"].tolist())
    total = len(X_train) + len(X_val) + len(X_test)
    log(f"      → embedded {total} prompts in {time.time() - t0:.1f}s")

    np.save(artifacts / "X_train.npy", X_train)
    np.save(artifacts / "X_val.npy", X_val)
    np.save(artifacts / "X_test.npy", X_test)
    train_df.to_parquet(artifacts / "train_df.parquet")
    val_df.to_parquet(artifacts / "val_df.parquet")
    test_df.to_parquet(artifacts / "test_df.parquet")

    log(f"[5/5] Training {len(HEADS)} heads")
    summary: List[Dict[str, Any]] = []
    for name, spec in HEADS.items():
        t0 = time.time()
        if spec["kind"] == "classification":
            le = LabelEncoder()
            y_train = le.fit_transform(train_df[name])
            y_val = le.transform(val_df[name])
            estimator = spec["make"]()
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, average="macro")
            joblib.dump(estimator, artifacts / "heads" / f"{name}.joblib")
            joblib.dump(le, artifacts / "label_encoders" / f"{name}.joblib")
            elapsed = time.time() - t0
            summary.append({
                "name": name, "kind": "cls",
                "val_acc": float(val_acc), "val_macro_f1": float(val_f1),
                "elapsed_s": float(elapsed),
            })
            log(f"      {name:<25} cls  val_acc={val_acc:.3f} f1={val_f1:.3f}  ({elapsed:.1f}s)")
        else:
            y_train = train_df[name].astype(float).values
            y_val = val_df[name].astype(float).values
            estimator = spec["make"]()
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)
            mae = float(np.mean(np.abs(y_pred - y_val)))
            joblib.dump(estimator, artifacts / "heads" / f"{name}.joblib")
            elapsed = time.time() - t0
            summary.append({
                "name": name, "kind": "reg",
                "val_mae": mae, "elapsed_s": float(elapsed),
            })
            log(f"      {name:<25} reg  val_mae={mae:.4f}                  ({elapsed:.1f}s)")

    meta = {
        "embedding_model": embedder.model_name,
        "embedding_dim": int(embedder.dim),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "seed": seed,
        "heads": list(HEADS.keys()),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Artifacts written → {artifacts.resolve()}")
    return {"meta": meta, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train multi-output prompt classifier.")
    ap.add_argument("--data", required=True, help="Path to prompt_classifier_dataset_1000.jsonl")
    ap.add_argument("--artifacts", default="classifier/artifacts", help="Where to save trained models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_classifier(data_path=args.data, artifacts_dir=args.artifacts, seed=args.seed)
    print("Next: python -m classifier.eval")


if __name__ == "__main__":
    main()
