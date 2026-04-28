"""Runtime classifier — single-prompt inference.

Loaded once at app/API startup; called per request.

CLI usage:
    python -m classifier.predict "Design a real-time chat system"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from .embed import Embedder
from .heads import HEADS


def bucketize(difficulty: float) -> str:
    if difficulty < 0.4:
        return "easy"
    if difficulty < 0.7:
        return "medium"
    return "hard"


class Classifier:
    def __init__(self, artifacts_dir: str | Path = "classifier/artifacts"):
        self.artifacts = Path(artifacts_dir)
        if not (self.artifacts / "heads").exists():
            raise FileNotFoundError(
                f"Artifacts not found at {self.artifacts}. Run `python -m classifier.train` first."
            )
        self.embedder = Embedder()
        self.heads: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Any] = {}
        for name, spec in HEADS.items():
            self.heads[name] = joblib.load(self.artifacts / "heads" / f"{name}.joblib")
            if spec["kind"] == "classification":
                self.label_encoders[name] = joblib.load(
                    self.artifacts / "label_encoders" / f"{name}.joblib"
                )

    def predict(self, prompt: str) -> Dict[str, Any]:
        emb = self.embedder.embed([prompt])
        out: Dict[str, Any] = {}
        for name, spec in HEADS.items():
            head = self.heads[name]
            if spec["kind"] == "classification":
                le = self.label_encoders[name]
                pred_idx = int(head.predict(emb)[0])
                out[name] = str(le.inverse_transform([pred_idx])[0])
                if hasattr(head, "predict_proba"):
                    probs = head.predict_proba(emb)[0]
                    out[f"{name}_proba"] = {
                        str(le.inverse_transform([i])[0]): float(p)
                        for i, p in enumerate(probs)
                    }
            else:
                out[name] = float(head.predict(emb)[0])
        out["difficulty_bucket"] = bucketize(float(out["difficulty"]))
        # Clamp regression outputs to reasonable ranges
        out["difficulty"] = float(np.clip(out["difficulty"], 0.0, 1.0))
        out["required_quality"] = float(np.clip(out["required_quality"], 0.0, 1.0))
        out["expected_output_tokens"] = max(1, int(round(float(out["expected_output_tokens"]))))
        return out


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Single-prompt classifier inference.")
    ap.add_argument("prompt", help="Prompt text to classify")
    ap.add_argument("--artifacts", default="classifier/artifacts")
    args = ap.parse_args()

    clf = Classifier(args.artifacts)
    print(json.dumps(clf.predict(args.prompt), indent=2))


if __name__ == "__main__":
    _cli()
