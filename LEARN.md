# LEARN — beginner-friendly walkthrough

This doc explains *why* every architectural choice in this repo was made, with diagrams. If you want the spec, see [`llm_router_poc_spec_and_dataset/README.md`](llm_router_poc_spec_and_dataset/README.md). If you want code, read the files in `classifier/`. This doc is for the *mental model*.

---

## 1. The big picture

A router has to answer one question per prompt: **"which model is the cheapest one that's still good enough for this prompt?"** Two equally-bad strategies bracket the answer:

```
Always frontier:   ┌──────────────┐                                 ┌──────────────┐
                   │ user prompt  │ ──────────────────────────────► │ GPT-5 / Opus │   expensive, sometimes overkill
                   └──────────────┘                                 └──────────────┘

Always cheapest:   ┌──────────────┐                                 ┌──────────────┐
                   │ user prompt  │ ──────────────────────────────► │ tiny-fast    │   cheap, fails on hard prompts
                   └──────────────┘                                 └──────────────┘
```

A good router routes prompt-by-prompt:

```
                ┌─────────────────┐    needs are easy        ┌──────────────┐
                │ "what is 2+2?"  │ ───────────────────────► │ tiny-fast    │   $0.0001
                └─────────────────┘                          └──────────────┘

                ┌─────────────────┐    needs are mid         ┌──────────────┐
                │ "summarize..."  │ ───────────────────────► │ haiku/mini   │   $0.001
                └─────────────────┘                          └──────────────┘

                ┌─────────────────┐    needs are hard        ┌──────────────┐
                │ "design distrib │ ───────────────────────► │ opus / GPT-5 │   $0.05
                │  cache..."      │                          └──────────────┘
                └─────────────────┘
```

The trick is figuring out the *needs* of a prompt **before** running the expensive model. That's what the classifier does.

---

## 2. The pipeline, end to end

```
  ┌───────────────────────────┐
  │ User prompt (string)      │
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Prompt Classifier                                                 │
  │                                                                   │
  │   prompt → MiniLM (frozen)  → 384-dim vector ──┐                  │
  │                                                ├─► task_type      │
  │                                                ├─► difficulty     │
  │                          6 sklearn heads ──────┼─► required_qual  │
  │                                                ├─► risk_level     │
  │                                                ├─► output_tokens  │
  │                                                └─► latency_sens.  │
  └─────────────┬─────────────────────────────────────────────────────┘
                │  structured needs
                ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Capability Table (static CSV — 100 rows)                          │
  │                                                                   │
  │   model        task        easy    medium    hard                 │
  │   ─────────    ─────────   ────    ──────    ────                 │
  │   gpt-5.5      coding      0.97    0.93      0.87                 │
  │   gpt-5.5      math        0.98    0.95      0.92                 │
  │   haiku-4.5    coding      0.90    0.82      0.71                 │
  │   nano         coding      0.81    0.69      0.52                 │
  │   …            …           …       …         …                    │
  │                                                                   │
  │   plus per-model price ($/1M tokens) and latency (ms)             │
  └─────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Router (rules)                                                    │
  │                                                                   │
  │   1. filter:  expected_quality ≥ required_quality                 │
  │               estimated_latency ≤ latency_sla                     │
  │                                                                   │
  │   2. select by policy:                                            │
  │      ┌────────────┬─────────────────────────────────────────┐     │
  │      │ cost       │ pick the cheapest eligible model        │     │
  │      │ latency    │ pick the fastest eligible model         │     │
  │      │ quality    │ pick the highest-quality eligible model │     │
  │      │ balanced   │ score = 0.55·Q − 0.25·C − 0.20·L        │     │
  │      └────────────┴─────────────────────────────────────────┘     │
  └─────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
       selected_model · estimated_cost · estimated_latency · routing_reason
```

**Why split it this way?** The classifier knows about *prompts*. The capability table knows about *models*. The router is the only thing that knows about both — and it's just rules. You can:
- swap the classifier without touching the table
- update the table without retraining the classifier
- add a new policy without touching either

That's the entire architectural argument.

---

## 3. Why two stages and not one big model?

A natural alternative: train one model that takes a prompt and directly outputs `model_id`. This is **direct routing** and it's what RouteLLM/Not Diamond do at scale. It works — but for a POC, the two-stage version wins on three counts:

| | Two-stage (this repo) | Direct routing |
|---|---|---|
| Explainability | "Picked Haiku because the prompt is easy coding (q≥0.71) and Haiku is cheapest among eligible." | "Picked Haiku." (opaque) |
| Adapt to new models | Add a row in the CSV. No retraining. | Retrain on new preference data. |
| Adapt to new policies | Edit `router/engine.py`. | Retrain. |
| Training data needed | Per-prompt labels (1k synthetic rows). | Pairwise preference data on real model outputs (10k+). |
| Best at scale | Limited by capability-table accuracy | Higher ceiling once you have data |

For 1k synthetic rows and zero real-model evals, two-stage is the right call. Once we have real eval data, the capability table becomes accurate, and we can also compare to a learned direct router as a future Phase-3.

---

## 4. The classifier — three model options compared

The classifier converts a string of text into the 6 structured outputs. There are three common ways to do this:

```
Option A — TF-IDF + sklearn          Option B — Embeddings + sklearn        Option C — Fine-tune DistilBERT

  prompt                                prompt                                  prompt
    │                                     │                                       │
    ▼                                     ▼                                       ▼
  ┌────────────────┐                  ┌────────────────┐                  ┌────────────────┐
  │ TfidfVectorizer│                  │ MiniLM         │                  │ DistilBERT     │
  │  word counts   │                  │  (frozen)      │                  │  (trained)     │
  │  weighted      │                  │  pretrained on │                  │  starts from   │
  │                │                  │  1B sent. pairs│                  │  pretrained,   │
  │                │                  │                │                  │  fine-tuned    │
  └────────┬───────┘                  └────────┬───────┘                  │  on our 1k     │
           │  ~20k-dim                         │  384-dim                 │                │
           │  sparse                           │  dense                   └────────┬───────┘
           ▼                                   ▼                                   │  768-dim
   ┌───────────────┐                   ┌───────────────┐                           ▼
   │ 6 sklearn     │                   │ 6 sklearn     │                  ┌────────────────┐
   │ heads         │                   │ heads         │                  │ 6 Linear       │
   └───────┬───────┘                   └───────┬───────┘                  │ heads          │
           ▼                                   ▼                          │ (trained)      │
        outputs                             outputs                       └───────┬────────┘
                                                                                  ▼
                                                                                outputs

  trainable: 1000s                     trainable: 1000s                    trainable: 66M+
  needs data:  ~hundreds                needs data: ~hundreds              needs data: 10k+
  paraphrases: bad                     paraphrases: good                  paraphrases: best
  inference:   ~3 ms                    inference:  ~15 ms                 inference: ~50 ms
  cold start:  fast                     cold start: 3-5 s                  cold start: 5-10 s
  quality on 1k rows: lowest            quality on 1k rows: SWEET SPOT     quality on 1k rows: overfits
```

We picked **Option B**.

### Why TF-IDF falls short

TF-IDF treats text as a bag of words, with each word getting a "rarity weight" (common words like "the" matter less; rare words like "kubernetes" matter more). The vector for a prompt has one slot per vocabulary word, almost all zero.

```
"build a chat app"
        │
        ▼
{ "build": 0.45, "a": 0.02, "chat": 0.61, "app": 0.42 }   ← every other slot is 0


"design a messaging system"
        │
        ▼
{ "design": 0.51, "a": 0.02, "messaging": 0.71, "system": 0.39 }


→ shared words: just "a". Cosine similarity ≈ 0.04.
  TF-IDF thinks these are completely different prompts.
```

A user phrasing the same intent two ways throws TF-IDF off because it has no notion of *meaning*, only of *which words appeared*.

### Why embeddings work

The embedding model has been pretrained on **a billion sentence pairs** so that semantically similar text produces nearby vectors:

```
                          "build a chat app"   •         • "design a messaging system"
                                                ╲       ╱
                                                 ╲     ╱
                                                  ╲   ╱
                                                  cosine ≈ 0.93
                                                                                    "explain quantum entanglement"
                                                                                                  •
                                                                                            cosine ≈ 0.12
```

Now the classifier head trains on a *consistent* signal: prompts that mean the same thing land near each other in vector space, so a simple linear classifier can carve them into the right buckets.

### Why DistilBERT fine-tuning isn't worth it here

DistilBERT has **66 million trainable parameters**. With 1000 training examples, it has enough capacity to *memorize the entire training set* — which is exactly the failure mode called overfitting. The model gets 100% on the training data and falls apart on anything new.

The frozen-embedding approach has the embedding model do the hard work (it was trained on a billion sentences elsewhere), and our heads only learn the small mapping `embedding → label` with a few thousand parameters. Far less prone to overfitting on 1k rows.

**Rule of thumb:**
- Have 100s–1000s of labeled rows? Frozen embeddings + sklearn heads.
- Have 10k+? Consider fine-tuning.
- Have 1M+? Consider a custom model.

---

## 5. Inside the classifier — six heads, one shared body

```
                              ┌────────────────────────────────────────────────┐
                              │                                                │
   prompt                     │             ┌─► LogReg          ─► task_type   │
     │                        │             │                                  │
     ▼                        │             ├─► GBM (regression)─► difficulty  │
  ┌──────────────────────┐    │             │                                  │
  │ MiniLM, frozen       │    │             ├─► GBM (regression)─► required_q  │
  │  → 384-dim vector    ├────┤  same vector                                   │
  │  ~10 ms CPU          │    │             ├─► LogReg          ─► risk_level  │
  └──────────────────────┘    │             │                                  │
                              │             ├─► RF (regression) ─► out_tokens  │
                              │             │                                  │
                              │             └─► LogReg          ─► latency_s.  │
                              │                                                │
                              └────────────────────────────────────────────────┘
                                          (six heads, trained independently)
```

Why **six** small heads instead of one model with six outputs?
- The outputs are different *types*: 3 classifications + 3 regressions. A single sklearn estimator can't handle both natively.
- They train independently — a problem with one head doesn't break the others. You can swap RandomForest for XGBoost on `expected_output_tokens` without retouching anything else.
- All six combined are <5 MB. Free architectural simplicity.

### Logistic Regression in 30 seconds

A weighted sum of input features → squashed into a probability:

```
   embedding[0] ─── × w₀ ┐
   embedding[1] ─── × w₁ │
   embedding[2] ─── × w₂ ├─► z = Σ wᵢ·xᵢ + b ─► σ(z) = 1/(1+e⁻ᶻ) ─► P(class=coding) = 0.84
       ⋯               │
   embedding[383] ─ × w₃₈₃ ┘

For 10 classes (task_type), you have 10 sets of weights and a softmax instead of sigmoid.
```

It draws a hyperplane through the embedding space. Because the embedding space is already structured (semantic regions form clusters), a straight line is usually enough. **Sub-millisecond inference.**

### Gradient Boosting in 30 seconds (used for regression heads)

You can't draw a straight line and predict a continuous number well, so we use a tree ensemble:

```
   tree₁ predicts:  difficulty ≈ 0.65
                                   ↓ residual = +0.18 (true was 0.83)
   tree₂ predicts the residual:    +0.12
                                   ↓ remaining residual = +0.06
   tree₃ predicts the residual:    +0.05
                                   …
   final = tree₁ + 0.1·tree₂ + 0.1·tree₃ + … + 0.1·tree₂₀₀  ≈ 0.82
```

Each tree is a series of `if feature > threshold` rules — they capture nonlinear patterns. Adding many shallow trees in series boosts accuracy. **Few-millisecond inference per head.**

---

## 6. Cross-validation — how we trust our accuracy number

If you train on 80% and test on 20%, you get **one number**. That number could be lucky (the 20% happened to be easy) or unlucky. With **5-fold cross-validation**:

```
   1000 rows, shuffled, split into 5 folds of 200
   ────────────────────────────────────────────────
   Fold:           A      B      C      D      E
   ────────────────────────────────────────────────
   iter 1:       TEST   train  train  train  train   → score₁
   iter 2:       train  TEST   train  train  train   → score₂
   iter 3:       train  train  TEST   train  train   → score₃
   iter 4:       train  train  train  TEST   train   → score₄
   iter 5:       train  train  train  train  TEST    → score₅

   reported: mean(scores) ± std(scores)
```

Every row gets used as test once. The mean is your estimate of generalization, the std tells you how stable your model is. We report both in `eval.py`.

**Important:** cross-validation is for **estimating accuracy**. The model you ship is trained on **all 1000 rows** — the CV number just tells you what to expect from it.

---

## 7. The capability table — why we treat it as ground truth

```
   ┌──────────────────────────────────────────────────────┐
   │ For each model: 10 task types × 3 difficulties = 30  │
   │ quality numbers, normalized to [0, 1].               │
   │                                                      │
   │ Total table size: 10 models × 30 = 300 numbers       │
   │ + cost & latency per model.                          │
   └──────────────────────────────────────────────────────┘

   Sources we leaned on for seed values:
   ┌────────────────────────────────────────────────────────┐
   │ coding/debugging  → SWE-bench Verified, BigCodeBench,  │
   │                     HumanEval, MBPP                    │
   │ math              → GSM8K (easy), MATH-500 (medium),   │
   │                     AIME-2025 (hard)                   │
   │ reasoning         → BBH, ARC-Challenge, GPQA-Diamond   │
   │ factual           → MMLU-Pro, SimpleQA                 │
   │ writing/creative  → MT-Bench, AlpacaEval               │
   │ extraction        → IFEval, BFCL, JSON-mode benchmarks │
   │ system_design     → no clean benchmark; estimated      │
   │                     from reasoning + coding composite  │
   └────────────────────────────────────────────────────────┘
```

The honest caveat: these are **POC seed values**. They're plausible (better-anchored for coding/math/reasoning, weaker for system_design/extraction). Replacing them with measured evals on a held-out prompt set is a Phase-2 improvement, not a Phase-1 requirement. The point of this POC is to demonstrate that the routing pipeline is **well-defined and easy to update once real numbers exist**.

---

## 8. Routing rules — filter then select

Given the classifier output and the capability table, routing is deterministic:

```
   classifier says:  task_type = coding, difficulty_bucket = hard,
                     required_quality = 0.85, latency_sensitivity = low

   for each model in the table:
       q = capability[model][coding][hard]                        ← look up quality
       cost = input_tokens·input_$/1M + output_tokens·output_$/1M ← compute cost
       latency = avg_latency_ms_seed                               ← look up latency

       eligible if  q ≥ 0.85  AND  latency ≤ sla_ms (if given)

   eligible models, sorted by chosen policy:
   ┌────────────┬────────────────────────────────────────┐
   │ cost_first │ ascending cost                         │
   │ latency_   │ ascending latency                      │
   │ quality_   │ descending quality                     │
   │ balanced   │ score = 0.55·Q − 0.25·Cnorm − 0.20·Lnorm │
   └────────────┴────────────────────────────────────────┘

   pick the top one. Emit `routing_reason` so the UI can explain why.
```

**Worked example.** Prompt: *"Design a distributed cache with strong consistency."*

```
classifier output:
  task_type = system_design, difficulty = 0.94 → bucket=hard, required_quality = 0.94

filter (need quality ≥ 0.94 on system_design/hard):
  gpt-5.5            quality=0.86  ❌
  gpt-5.4-mini       quality=0.69  ❌
  gpt-5.4-nano       quality=0.46  ❌
  claude-opus-4.7    quality=0.87  ❌
  claude-sonnet-4.6  quality=0.78  ❌
  claude-haiku-4.5   quality=0.62  ❌
  gemini-3.1-pro     quality=0.85  ❌
  gemini-flash-lite  quality=0.48  ❌
  deepseek-r1        quality=0.76  ❌
  llama-4-maverick   quality=0.58  ❌

→ NONE pass. Router emits: "no model meets required_quality=0.94 on system_design/hard;
                            best available: claude-opus-4.7 at 0.87.
                            Either lower the bar (probabilistic match) or escalate."

(In a real system this surfaces as an alert: maybe the threshold needs calibration,
or the prompt category genuinely exceeds your fleet.)
```

Same prompt with `required_quality = 0.85` (calibrated lower):

```
filter:
  claude-opus-4.7    quality=0.87  ✅  cost=$0.052  latency=3200ms
  gemini-3.1-pro     quality=0.85  ✅  cost=$0.012  latency=2800ms
  gpt-5.5            quality=0.86  ✅  cost=$0.025  latency=2500ms

policy = cost_first → gemini-3.1-pro selected
policy = latency_first → gpt-5.5 selected
policy = quality_first → claude-opus-4.7 selected
```

The same prompt produces different routes depending on the **objective** — and the user gets to pick.

---

## 9. Glossary

| Term | One-line meaning |
|---|---|
| Embedding | A dense vector that represents a piece of text. |
| Sentence transformer | A model trained to output one vector per sentence (vs per word). |
| Frozen model | A pretrained model whose weights you don't update during your training. |
| Head | A small output layer attached to a shared body (the embedder). |
| Logistic regression | A linear classifier — weighted sum of inputs → probability. |
| Gradient boosting | An ensemble of small decision trees, each fixing the previous trees' errors. |
| Random forest | An ensemble of independent trees, prediction = average. |
| Cross-validation | Average accuracy over k different train/test splits — robust estimate. |
| Macro-F1 | Average F1 score across classes, weighting each class equally. |
| MAE | Mean absolute error — average distance between prediction and truth. |
| MAPE | Mean absolute percentage error — used for output-token regression. |
| R² | Fraction of target variance explained by the model. 1.0 = perfect, 0 = no better than predicting the mean. |
| Stratified split | Split that preserves class proportions in each part. |
| Required quality | Classifier-predicted minimum quality the prompt needs (a 0-1 number). |
| Difficulty bucket | `easy` (<0.4), `medium` (<0.7), or `hard` — the bucket the difficulty score falls into. |
| Filter-then-rank | Routing pattern: drop ineligible models first, rank what remains by policy. |

---

## 10. What to read next

- `classifier/heads.py` — the per-output estimator config in 30 lines
- `classifier/train.py` — the full training pipeline as a CLI
- `classifier/eval.py` — how cross-validation is wired up
- `llm_router_poc_spec_and_dataset/README.md` — the canonical design doc
- [RouteLLM paper](https://arxiv.org/abs/2406.18665) — the matrix-factorization router (Phase-3 idea)
- [FrugalGPT paper](https://arxiv.org/abs/2305.05176) — cascade routing with a learned scorer
- [Hybrid LLM paper](https://arxiv.org/abs/2404.14618) — pseudo-label difficulty from model-vs-model score gaps
