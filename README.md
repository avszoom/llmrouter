# LLM Inference Router POC

A prompt-aware LLM router. A trained classifier predicts what the prompt *needs*; a deterministic rule-based router consults a static model-capability table; the cheapest/fastest model that meets the quality bar is selected.

> **Source of truth for the design:** [`llm_router_poc_spec_and_dataset/README.md`](llm_router_poc_spec_and_dataset/README.md)
> **Beginner-friendly walkthrough of the ML choices:** [`LEARN.md`](LEARN.md)

---

## Architecture

```
                prompt
                  │
                  ▼
        ┌─────────────────────┐
        │ Prompt Classifier   │   ← all-MiniLM-L6-v2 (frozen) + 6 sklearn heads
        │  task_type          │     (LogReg + GBM/RF), trained on 1k labeled prompts
        │  difficulty         │
        │  required_quality   │
        │  risk_level         │
        │  expected_tokens    │
        │  latency_sensitivity│
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Capability Table    │   ← static CSV: per (model, task_type, difficulty_bucket)
        │  cost / latency /   │     → quality_easy / quality_medium / quality_hard
        │  per-task quality   │     anchored to public benchmarks
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Router (rules)      │   ← filter: quality ≥ required AND latency ≤ SLA
        │  cost_first         │     select: by chosen policy
        │  latency_first      │
        │  quality_first      │
        │  balanced           │
        └─────────┬───────────┘
                  │
                  ▼
       selected_model · estimated_cost_usd · estimated_latency_ms · routing_reason
```

**Deployment target:** Streamlit Cloud (single Python app, free tier).

---

## Quickstart

```bash
# 1. install deps (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. train the classifier (downloads MiniLM on first run, ~80 MB)
python -m classifier.train --data llm_router_poc_spec_and_dataset/prompt_classifier_dataset_1000.jsonl

# 3. evaluate (held-out test + 5-fold cross-validation)
python -m classifier.eval

# 4. one-off prediction from the CLI
python -m classifier.predict "Design a real-time chat system with multi-region failover"

# 5. validate the capability table (10 models × 10 task_types × 3 buckets)
python -m capability.loader --validate

# 6. run the unit tests (router rules, capability validator)
pytest -q

# 7. launch the Streamlit demo (3-panel UI: classifier · routing · cost)
streamlit run ui/app.py
```

---

## What each command does (and what to expect)

### `python -m classifier.train --data <jsonl>`

Loads the 1000-row labeled dataset, splits it 80/10/10 stratified by `task_type`, embeds every prompt once with `all-MiniLM-L6-v2`, and fits 6 sklearn heads (3 LogReg classifiers + 2 GradientBoostingRegressors + 1 RandomForestRegressor). Writes:
- `classifier/artifacts/heads/*.joblib` (6 trained models)
- `classifier/artifacts/label_encoders/*.joblib` (3 encoders for categorical outputs)
- `classifier/artifacts/X_{train,val,test}.npy` (cached embeddings, reused by `eval.py`)
- `classifier/artifacts/{train,val,test}_df.parquet` (the dataset splits)
- `classifier/artifacts/meta.json` (run metadata)

**Expected console output:**
```
[1/5] Loading dataset: llm_router_poc_spec_and_dataset/prompt_classifier_dataset_1000.jsonl
      → 1000 rows | columns: 9
[2/5] Stratified split 80/10/10 (stratify=task_type, seed=42)
      → train=800 | val=100 | test=100
[3/5] Loading embedding model: all-MiniLM-L6-v2
      → loaded in 3.5s | dim=384
[4/5] Embedding prompts (batched)
      → embedded 1000 prompts in 8.7s
[5/5] Training 6 heads

Validation summary
  output                    kind  metric                         time
  task_type                 cls   val_acc=0.870 f1=0.852         0.4s
  difficulty                reg   val_mae=0.0612                 1.6s
  required_quality          reg   val_mae=0.0488                 1.5s
  risk_level                cls   val_acc=0.910 f1=0.895         0.2s
  expected_output_tokens    reg   val_mae=87.4                   2.1s
  latency_sensitivity       cls   val_acc=0.880 f1=0.871         0.2s

Artifacts written → /…/llmrouter/classifier/artifacts
Next: python -m classifier.eval
```

(Numbers will vary slightly run-to-run; first run pays a one-time 5–10 s model download.)

### `python -m classifier.eval`

Runs two evaluations:
1. **Held-out test set** (the 100 rows reserved by `train.py`): per-output accuracy / F1 / MAE / R².
2. **5-fold stratified cross-validation on all 1000 rows**: mean ± std for each output. This is the honest number to quote for portfolio claims.

**Expected console output (abridged):**
```
============================================================
Held-out test set (1 split, deterministic seed=42)
============================================================
Test rows: 100

  task_type                 acc=0.870  macro_f1=0.852  [PASS]
  difficulty                MAE=0.0612 MAPE= 12.4% R²=+0.821 [PASS]
  required_quality          MAE=0.0488 MAPE=  6.3% R²=+0.781 [PASS]
  risk_level                acc=0.910  macro_f1=0.895  [PASS]
  expected_output_tokens    MAE=87.40  MAPE= 18.2% R²=+0.652 [PASS]
  latency_sensitivity       acc=0.880  macro_f1=0.871  [PASS]

============================================================
5-fold cross-validation on full dataset
============================================================
Embedding 1000 prompts (one pass)...

  task_type                 macro_f1 = 0.846 ± 0.018   folds=[0.831, 0.864, 0.852, 0.823, 0.860]
  difficulty                MAE      = 0.0628 ± 0.0034 folds=[0.062, 0.061, 0.066, 0.060, 0.065]
  required_quality          MAE      = 0.0501 ± 0.0029 …
  risk_level                macro_f1 = 0.889 ± 0.012   …
  expected_output_tokens    MAE      = 89.14 ± 4.21    …
  latency_sensitivity       macro_f1 = 0.872 ± 0.014   …

Full report → classifier/artifacts/eval_report.json
```

**Pass bar:**
| Output | Metric | Threshold |
|---|---|---|
| `task_type` | macro F1 | ≥ 0.80 |
| `risk_level` | macro F1 | ≥ 0.80 |
| `latency_sensitivity` | macro F1 | ≥ 0.80 |
| `difficulty` | MAE | ≤ 0.10 |
| `required_quality` | MAE | ≤ 0.10 |
| `expected_output_tokens` | MAPE | ≤ 25% |

If a head FAILs, see [`LEARN.md`](LEARN.md) for what to try next.

### `python -m classifier.predict "<prompt>"`

Loads the trained heads and runs single-prompt inference (~15–25 ms warm). Prints the structured prediction as JSON.

**Expected output:**
```json
{
  "task_type": "system_design",
  "task_type_proba": {"coding": 0.04, "system_design": 0.71, "reasoning": 0.18, "...": "..."},
  "difficulty": 0.84,
  "required_quality": 0.93,
  "risk_level": "high",
  "risk_level_proba": {"low": 0.02, "medium": 0.21, "high": 0.77},
  "expected_output_tokens": 1080,
  "latency_sensitivity": "low",
  "latency_sensitivity_proba": {"low": 0.79, "medium": 0.18, "high": 0.03},
  "difficulty_bucket": "hard"
}
```

### `python -m capability.loader --validate`

Loads `model_capability_table_seed.csv`, prints a one-line summary per model, and validates two invariants: every quality is in `[0, 1]`, and quality is monotone non-increasing across difficulty (`easy ≥ medium ≥ hard`). Exits non-zero on issues.

**Expected output:**
```
Loaded 10 models from llm_router_poc_spec_and_dataset/model_capability_table_seed.csv
  gpt-5.5                   OpenAI             $2.500/$15.000/1M  2500ms  (30 quality entries)
  gpt-5.4-mini              OpenAI             $0.375/$2.250/1M  1600ms  (30 quality entries)
  …
  llama-4-maverick          Meta/hosted        $0.200/$0.600/1M  1100ms  (30 quality entries)

Validation: OK
```

### `pytest -q`

Runs 21 unit tests across two suites — `tests/test_capability.py` (8 tests on the real CSV) and `tests/test_router.py` (13 tests on a synthetic 3-model fleet covering all 4 policies, SLA filtering, and the fallback path).

**Expected output:**
```
.....................                                                    [100%]
21 passed in 0.04s
```

### `streamlit run ui/app.py`

Launches the demo on `http://localhost:8501`. First load takes ~5 s to import the embedding model; subsequent prompts are ~15–25 ms.

**The page has:**
- A prompt input + **Route** button at the top.
- A sidebar with the **policy selector** (`cost_first` / `latency_first` / `quality_first` / `balanced`), an optional **latency SLA**, and 5 example prompts to pick from.
- Three columns of results below the input:
  1. **Classifier** — task_type, difficulty (with bucket), required_quality, risk_level, latency_sensitivity, expected_output_tokens, plus a "Class probabilities" expander.
  2. **Routing decision** — selected model, provider, policy, human-readable `routing_reason`, eligible list (with the selected one marked), and a rejected expander explaining *why* each rejected model didn't qualify.
  3. **Cost & latency** — estimated cost in USD, estimated latency, estimated quality, plus a token-level breakdown using the chosen model's per-1M pricing.

If no model meets the bar, the routing panel shows a red **fallback** card with the highest-quality available and an explanation.

---

## Routing the prompt — what `/route` does

After classification, the router consults the capability table and picks a model:

1. **Filter** — drop any model whose quality on `(task_type, difficulty_bucket)` is below the classifier-predicted `required_quality`. If a `latency_sla_ms` is set, drop models that exceed it.
2. **Rank** — order remaining models by the chosen policy:
   - `cost_first` — cheapest first
   - `latency_first` — fastest first
   - `quality_first` — highest quality first
   - `balanced` — score = `0.55·Q − 0.25·Cnorm − 0.20·Lnorm`
3. **Emit** — `RoutingDecision` with selected model, estimated cost, latency, quality, eligible list, rejected reasons, and a human-readable `routing_reason`.
4. **Fallback** — if no model clears the quality bar, pick the highest-quality available and flag the decision as `fallback=True`.

Programmatic use:
```python
from classifier import Classifier
from capability.loader import load_capability_table
from router.engine import route_prompt, Objective

clf = Classifier()
models = load_capability_table()
classifier_output, decision = route_prompt(
    clf, models,
    "Design a distributed cache with multi-region failover",
    Objective(policy="balanced", latency_sla_ms=3000),
)
```

---

## Repo layout

```
llmrouter/
├── README.md                           ← you are here
├── LEARN.md                            ← beginner-friendly ML walkthrough
├── requirements.txt
├── .gitignore
├── classifier/
│   ├── data.py                         ← JSONL load + stratified split
│   ├── embed.py                        ← MiniLM wrapper
│   ├── heads.py                        ← per-output sklearn estimator config
│   ├── train.py / eval.py / predict.py ← CLIs
│   └── artifacts/                      ← (gitignored) generated by train.py
├── capability/
│   └── loader.py                       ← load + validate capability table; CLI: --validate
├── router/
│   └── engine.py                       ← filter + rank + Objective + RoutingDecision
├── ui/
│   └── app.py                          ← Streamlit 3-panel demo
├── tests/
│   ├── test_capability.py              ← 8 tests on real seed CSV
│   └── test_router.py                  ← 13 tests on synthetic fleet
└── llm_router_poc_spec_and_dataset/
    ├── README.md                       ← canonical design doc
    ├── TRAINING_METHOD.md
    ├── model_capability_table_seed.csv ← (model × task_type × difficulty) quality
    └── prompt_classifier_dataset_1000.{csv,jsonl}
```

---

## Status

- [x] Capability table extended to (model × task_type × difficulty_bucket)
- [x] Classifier package (data, embed, heads, train, eval, predict)
- [x] Capability-table loader + validator
- [x] Rule-based router with 4 policies + SLA filter + fallback
- [x] Streamlit UI (3-panel: classifier · routing · cost estimate)
- [x] Tests (21 passing)
- [ ] Streamlit Cloud deployment

See [`llm_router_poc_spec_and_dataset/README.md`](llm_router_poc_spec_and_dataset/README.md) for the full design and [`LEARN.md`](LEARN.md) for the model/architecture rationale.
