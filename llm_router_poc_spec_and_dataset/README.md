# LLM Inference Router POC

## Goal

Build a proof-of-concept LLM inference router that sends each prompt to the cheapest and/or fastest model that can still satisfy the required quality.

The core belief is:

> The latest/biggest model is usually the quality ceiling, but many prompts do not require that ceiling.

This POC uses:

1. A Prompt Classifier trained on synthetic POC data.
2. A Model Capability Table seeded from public pricing and benchmark-style assumptions.
3. A Routing Engine that matches prompt requirements to model capabilities.
4. Optional confidence checks and fallback logic.

---

## Architecture

```text
User Prompt
  -> Prompt Classifier
  -> {task_type, difficulty, required_quality, risk_level, expected_output_tokens, latency_sensitivity}
  -> Model Capability Table
  -> Router
  -> Selected Model
  -> Optional confidence check / fallback
```

---

## Prompt Classifier

The Prompt Classifier does not choose the model. It answers:

> What does this prompt need?

### Input

```json
{
  "prompt": "Design a multi-region LLM inference router optimized for latency and cost."
}
```

### Output

```json
{
  "task_type": "system_design",
  "difficulty": 0.87,
  "difficulty_bucket": "hard",
  "required_quality": 0.93,
  "risk_level": "high",
  "expected_output_tokens": 1100,
  "latency_sensitivity": "low"
}
```

### Supported task types

```text
writing
summarization
extraction
factual
coding
debugging
math
reasoning
system_design
creative
```

---

## Why there is no `cheap_good_enough`

This dataset is for the Prompt Classifier, not for a direct router classifier.

So the label is not:

```text
cheap_good_enough
```

The labels are:

```text
task_type
difficulty
difficulty_bucket
required_quality
risk_level
expected_output_tokens
latency_sensitivity
```

A future direct-router model could learn:

```text
prompt -> cheap_good_enough
```

But this POC intentionally keeps the system explainable:

```text
Prompt Classifier = what the prompt needs
Capability Table = what each model can handle
Router = match need to capability
```

---

## Dataset

Files:

- `prompt_classifier_dataset_1000.jsonl`
- `prompt_classifier_dataset_1000.csv`

Example row:

```json
{
  "id": "pc_0001",
  "prompt": "Design a scalable URL shortener for global traffic.",
  "task_type": "system_design",
  "difficulty": 0.82,
  "difficulty_bucket": "hard",
  "required_quality": 0.93,
  "risk_level": "high",
  "expected_output_tokens": 1100,
  "latency_sensitivity": "low"
}
```

This is synthetic POC training data. It is useful for demonstrating the architecture and training method. It is not production ground truth.

---

## Training Method

Train a multi-output prompt classifier.

| Output | Model type |
|---|---|
| task_type | multiclass classifier |
| difficulty | regression |
| difficulty_bucket | derived from difficulty |
| required_quality | regression |
| risk_level | multiclass classifier |
| expected_output_tokens | regression |
| latency_sensitivity | multiclass classifier |

### Simple baseline

Use TF-IDF + scikit-learn:

```text
prompt text -> TF-IDF -> classifiers/regressors
```

Recommended:

- `TfidfVectorizer`
- `LogisticRegression` for task/risk/latency
- `GradientBoostingRegressor` or `RandomForestRegressor` for difficulty/quality/tokens

### Stronger POC

Use embeddings:

```text
prompt -> embedding -> XGBoost/sklearn models
```

Add simple features:

```text
prompt length
code keyword count
reasoning keyword count
contains JSON requirement
contains correctness words
contains latency words
```

Why this works well for a POC:

- Cheap to train
- Explainable
- Easy to debug
- No transformer fine-tuning needed
- Good enough for 1,000 labeled samples

---

## Model Capability Table

File:

- `model_capability_table_seed.csv`

**Schema (long format, one row per `(model, task_type)` pair):**

```text
model
provider
input_usd_per_1m
output_usd_per_1m
avg_latency_ms_seed
output_tps_seed
task_type
quality_easy
quality_medium
quality_hard
notes
```

10 models × 10 task_types = **100 data rows**. Each row carries three quality scores — one per difficulty bucket — so the router can match `(task_type, difficulty_bucket)` to a per-model expected quality.

Quality scores are normalized to `[0, 1]` and are **monotone non-increasing** with difficulty: `quality_easy ≥ quality_medium ≥ quality_hard`.

**Anchoring:** seed values are extrapolated from public benchmarks — SWE-bench Verified and BigCodeBench (coding/debugging), GPQA-Diamond and BBH (reasoning), AIME and MATH-500 (math), MMLU-Pro and SimpleQA (factual), HumanEval and MBPP (coding/easy), MT-Bench and AlpacaEval (writing/creative), IFEval and BFCL (extraction). Cross-checked against Artificial Analysis and LiveBench leaderboards. Frontier-tier models hold 0.85–0.92 on hard tasks; cheap-tier models drop to 0.46–0.66.

> Pricing is seeded from public provider pricing pages where available. Latency and task quality scores are POC seed estimates, not measured production data.

For a real system, replace the seed values with your own eval results (see "How to build the real model-quality mapping" below).

---

## How to build the real model-quality mapping

The proper way to build:

```text
model + task_type + difficulty_bucket -> average_quality, p50_latency, p95_latency, avg_cost
```

### Step 1: Create benchmark prompts

Create prompts across every task and difficulty bucket:

```text
writing/easy
writing/medium
coding/easy
coding/hard
system_design/hard
math/hard
...
```

### Step 2: Run every model

For each prompt:

```text
model -> answer
```

Collect:

```text
latency
input_tokens
output_tokens
cost
answer
```

### Step 3: Judge quality

Use different scoring methods by task:

| Task type | Scoring method |
|---|---|
| coding | unit tests + LLM judge |
| extraction | schema validation + field accuracy |
| math | exact answer / symbolic check |
| summarization | LLM judge + coverage |
| writing | LLM judge + preference score |
| system_design | rubric-based LLM judge |
| factual | fact coverage + judge |

Normalize:

```text
quality = judge_score / 10
```

Or relative to the best model:

```text
relative_quality = model_score / best_model_score
```

### Step 4: Aggregate

```text
task_type + difficulty_bucket + model -> avg_quality
```

That final table drives routing.

---

## Routing Algorithm

### Step 1: classify

```json
{
  "task_type": "coding",
  "difficulty": 0.71,
  "difficulty_bucket": "hard",
  "required_quality": 0.90,
  "expected_output_tokens": 700,
  "latency_sensitivity": "low"
}
```

### Step 2: estimate each model

For every model, look up quality at the prompt's task type **and difficulty bucket**:

```text
expected_quality = capability_table[model][task_type][difficulty_bucket]
estimated_cost   = input_tokens * input_price + expected_output_tokens * output_price
estimated_latency = avg_latency_ms_seed
```

### Step 3: filter

```text
expected_quality >= required_quality
```

Also apply latency SLA if given:

```text
estimated_latency <= latency_sla_ms
```

A model is **eligible** only if it clears both filters. The harder the prompt, the more cheap models drop out — by design.

### Step 4: select

#### cost_first

```text
choose min(cost) from eligible models
```

#### latency_first

```text
choose min(latency) from eligible models
```

#### quality_first

```text
choose max(expected_quality)
```

#### balanced

```text
score = 0.55 * expected_quality
      - 0.25 * normalized_cost
      - 0.20 * normalized_latency
```

Choose highest score.

---

## Example

Prompt:

```text
Write a short birthday message
```

Classifier:

```json
{
  "task_type": "writing",
  "difficulty": 0.12,
  "required_quality": 0.60,
  "latency_sensitivity": "high"
}
```

Router:

```text
Use a cheap/fast model because writing quality exceeds the requirement.
```

Prompt:

```text
Design a distributed cache with strong consistency and multi-region failover
```

Classifier:

```json
{
  "task_type": "system_design",
  "difficulty": 0.94,
  "required_quality": 0.94,
  "latency_sensitivity": "low"
}
```

Router:

```text
Use a frontier model because cheap models do not meet the required quality threshold.
```

---

## MVP Implementation Plan

1. Train Prompt Classifier.
2. Load Model Capability Table.
3. Implement routing policy.
4. Expose `/classify` and `/route`.
5. Log decisions and estimated cost.
6. Add optional fallback later.

### `POST /classify`

Request:

```json
{
  "prompt": "Design a real-time chat system"
}
```

Response:

```json
{
  "task_type": "system_design",
  "difficulty": 0.85,
  "difficulty_bucket": "hard",
  "required_quality": 0.93,
  "risk_level": "high",
  "expected_output_tokens": 1000,
  "latency_sensitivity": "low"
}
```

### `POST /route`

Request:

```json
{
  "prompt": "Design a real-time chat system",
  "policy": "balanced",
  "latency_sla_ms": 2500
}
```

Response:

```json
{
  "selected_model": "claude-sonnet-4.6",
  "provider": "Anthropic",
  "estimated_quality": 0.94,
  "estimated_cost_usd": 0.018,
  "estimated_latency_ms": 2200,
  "routing_reason": "system_design/hard requires quality 0.93; selected lowest-cost eligible model under SLA"
}
```

---

## Future Improvements

1. Replace synthetic labels with real labeled prompts.
2. Replace seed model quality with measured evals.
3. Add prompt caching.
4. Add fallback:
   ```text
   if confidence low -> escalate to stronger model
   ```
5. Add online learning:
   ```text
   user feedback + judge score -> update capability table
   ```
6. Add per-domain capability tables:
   ```text
   coding_quality != legal_quality != writing_quality
   ```

---

## Key Claim

This POC demonstrates:

> Route by prompt requirements, not fixed model preference.

The router estimates prompt need, consults model capability, and chooses the cheapest/fastest model that satisfies the quality bar.
