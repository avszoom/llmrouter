# LLM Inference Router Simulator

> Cost / latency / quality-aware routing across mocked LLM providers, with a reproducible benchmark harness.

**Status:** design draft. Iterate freely — every section below is up for debate.

---

## 1. Why this exists

Build a portfolio-grade routing service that demonstrates real infra thinking:

- prompt-aware routing (not hard-coded model picks)
- explicit cost / latency / quality tradeoffs under a user-chosen objective
- caching, fallback, circuit-breaker
- a **benchmark** that *proves* the router beats baselines and approaches the oracle

All providers are mocked (deterministic, no API spend) so we can iterate the routing logic without burning credits and the benchmark stays reproducible.

---

## 2. Design pattern (grounded in literature)

Two-stage routing: **feature extraction → per-model quality prediction → objective-aware reranking**.

This pattern is well-supported:

| Paper / system | Idea we borrow |
|---|---|
| **RouteLLM** (LMSYS, [arxiv 2406.18665](https://arxiv.org/abs/2406.18665)) | Matrix-factorization router: `embed(prompt) · W · embed(model) → P(strong wins)`. Threshold sweep on Pareto curve. |
| **FrugalGPT** (Stanford, [arxiv 2305.05176](https://arxiv.org/abs/2305.05176)) | Sequential cascade: try cheapest → score answer → escalate if below threshold. |
| **Hybrid LLM** (MSR ICLR'24, [arxiv 2404.14618](https://arxiv.org/html/2404.14618)) | Pseudo-label difficulty from `score_large − score_small`; train cheap classifier on embedding. |
| **AutoMix** (NeurIPS'23, [arxiv 2310.12963](https://arxiv.org/abs/2310.12963)) | Self-verification + meta-verifier (POMDP/KNN) for noisy escalation. |
| **BaRP** ([arxiv 2510.07429](https://arxiv.org/abs/2510.07429)) | Contextual bandit (LinUCB) over (prompt features, user-preference vector). |
| **Portkey Gateway** ([github](https://github.com/Portkey-AI/gateway)) | Composable routing: fallback ⊃ load-balancer ⊃ conditional ⊃ leaf rules. Plugin model. |
| **LiteLLM** ([github](https://github.com/BerriAI/litellm)) | Cooldown-on-429, priority fallback chains, lowest-TPM strategy. |
| **RouterBench** (Martian, [arxiv 2403.12031](https://arxiv.org/abs/2403.12031)) | 405k inferences × 11 models × 8 datasets — gold benchmark we can replay against. |

---

## 3. Architecture

```
POST /route {prompt, objective}
        │
        ▼
┌─────────────────────┐
│ FeatureExtractor[]  │  length, task_type, embedding, cacheability
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Cache.lookup        │  exact-hash (P1) → semantic cosine ≥ 0.85 (P2)
└─────────┬───────────┘
   miss   │   hit ──► return
          ▼
┌─────────────────────┐
│ QualityPredictor    │  {model_id: predicted_quality}
└─────────┬───────────┘    plugins: oracle | rule_based | cascade | matrix_factorization
          ▼
┌─────────────────────┐
│ Reranker(objective) │  ordered candidates
└─────────┬───────────┘    plugins: cost / latency / quality / balanced / pareto
          ▼
┌─────────────────────┐
│ Provider.call       │  retry → CircuitBreaker → next candidate (fallback chain)
└─────────┬───────────┘    plugins: cheap-fast, fast, premium, local-llama-sim
          ▼
   Cache.store + Logger.log(decision, cost, latency, quality)
```

---

## 4. Plugin model (the core abstraction)

Five `Protocol`s + decorator-based registry. Compose via YAML; swap any layer without touching others.

```python
# llmrouter/core/types.py
class FeatureExtractor(Protocol):  name: str; def extract(prompt, ctx) -> dict
class QualityPredictor(Protocol):  name: str; def predict(features, models) -> dict[str, float]
class Reranker(Protocol):          name: str; def rerank(qualities, metas, objective) -> list[ScoredModel]
class Provider(Protocol):          name: str; meta: ModelMeta; async def call(prompt) -> Response
class CachePolicy(Protocol):       async def lookup(...); async def store(...)

# llmrouter/core/registry.py
@register("provider", "cheap-fast-mock")
class CheapFastMock(Provider): ...
```

YAML wires them together:

```yaml
router:
  features: [length, task_type, embedding]
  quality_predictor: rule_based       # oracle | rule_based | cascade | matrix_factorization
  reranker: balanced                  # cost_first | latency_first | quality_first | balanced | pareto
  cache: exact_sqlite                 # exact_sqlite | semantic_cosine
  fallback: cooldown_breaker

objective:
  mode: balanced                      # cost | latency | quality | balanced
  weights: { cost: 0.3, latency: 0.2, quality: 0.5 }
  budget_usd: 0.01
  sla_ms: 1500

providers:
  - { name: cheap-fast-model, cost_in_per_1m: 0.05, cost_out_per_1m: 0.20,  latency_p50: 80,   latency_p95: 200,  fail_rate: 0.02,  quality: { qa: 0.60, code: 0.40, reasoning: 0.30, creative: 0.50 } }
  - { name: fast-model,       cost_in_per_1m: 0.15, cost_out_per_1m: 0.60,  latency_p50: 150,  latency_p95: 400,  fail_rate: 0.01,  quality: { qa: 0.75, code: 0.70, reasoning: 0.60, creative: 0.70 } }
  - { name: premium-model,    cost_in_per_1m: 3.00, cost_out_per_1m: 15.00, latency_p50: 800,  latency_p95: 2000, fail_rate: 0.005, quality: { qa: 0.92, code: 0.88, reasoning: 0.95, creative: 0.85 } }
  - { name: local-llama-sim,  cost_in_per_1m: 0.00, cost_out_per_1m: 0.00,  latency_p50: 1200, latency_p95: 3500, fail_rate: 0.03,  quality: { qa: 0.70, code: 0.55, reasoning: 0.50, creative: 0.60 } }
```

---

## 5. File layout (target)

```
llmrouter/
├── pyproject.toml
├── README.md
├── config/default.yaml
├── llmrouter/
│   ├── api/{main.py, routes.py, schemas.py}        # FastAPI: /route /benchmark /metrics /health
│   ├── core/{registry.py, types.py, pipeline.py, config.py}
│   ├── features/{length.py, task_type.py, embedding.py, cacheability.py}
│   ├── predictors/{oracle.py, rule_based.py, cascade.py, matrix_factorization.py}
│   ├── rerankers/{cost_first.py, latency_first.py, quality_first.py, balanced.py, pareto.py}
│   ├── providers/{mock.py}                          # one parametrized class, 4 instances via config
│   ├── cache/{exact.py, semantic.py}
│   ├── policies/{circuit_breaker.py, retry.py}
│   ├── pricing.py
│   └── storage/{db.py, schema.sql}                  # SQLite: prompts, decisions, metrics
├── benchmark/
│   ├── prompts/labeled_v1.jsonl                     # ~150 prompts: {prompt, task_type, difficulty, expected_tokens}
│   ├── ground_truth.py                              # samples per-model quality from Beta(α,β) keyed on (model, task_type, difficulty)
│   ├── runner.py                                    # replay all (predictor × reranker) combos
│   ├── metrics.py                                   # cost, p50/p95/p99 latency, mean quality, SLO viol, $/quality
│   └── report.py                                    # markdown table + matplotlib Pareto scatter
├── tests/{test_pipeline.py, test_providers.py, test_rerankers.py, test_benchmark.py}
└── scripts/make_prompt_set.py
```

Deps: `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `numpy`, `tiktoken`, `matplotlib`, `pytest`, `pytest-asyncio`, `httpx`. Phase 2 adds `sentence-transformers`, `streamlit`.

---

## 6. Phased delivery

### Phase 1 — Plugin scaffold + cost/latency/quality routing + benchmark  *(~2 days)*
- Protocols + registry + YAML loader.
- Mock provider with `asyncio.sleep(numpy.random.lognormal(...))` latency and Bernoulli failure.
- Features: `length`, `task_type` (keyword/regex), `cacheability`.
- Predictors: **oracle** (uses ground-truth — upper bound), **rule_based** (task→model lookup from YAML), **random** (lower bound).
- Rerankers: all 5 (`cost_first`, `latency_first`, `quality_first`, `balanced`, `pareto`).
- Cache: exact SHA-256 → SQLite.
- Fallback: retry-on-fail with next-best candidate.
- Benchmark: 150 prompts × {3 predictors × 5 rerankers} = **15 policies + 4 baselines** (random, always-cheapest, always-best, oracle).
- Output: markdown table + Pareto plot.

### Phase 2 — Quality predictors + semantic cache + dashboard  *(~2 days)*
- `embedding` feature via `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~10 ms CPU).
- `semantic_cosine` cache, threshold ≥ 0.85; sweep threshold in benchmark.
- `cascade` predictor (FrugalGPT) — try cheapest → synthetic scorer → escalate.
- `difficulty_classifier` predictor (Hybrid LLM) — pseudo-labeled from oracle quality gaps; logistic regression on embedding.
- `circuit_breaker` policy — 3 consecutive failures → 30s cooldown.
- Streamlit dashboard: live req/s, hit-rate, cost-burn, fallback rate; benchmark report viewer.

### Phase 3 — Adaptive + load  *(~3 days)*
- `matrix_factorization` predictor (RouteLLM-style) trained on Phase-1/2 logs.
- `linucb_bandit` reranker (BaRP-style) with online updates.
- Locust load test, 1→500 RPS, SLO violation curve.
- RouterBench subset replay as external sanity check.

---

## 7. Benchmark design (the proof-of-router)

**Prompt set:** 150 prompts × `{qa, code, reasoning, creative, summary}` × `{easy, medium, hard}`. JSONL with `expected_tokens`, `task_type`, `difficulty`.

**Ground-truth quality:** for each `(model, task_type, difficulty)`, define a `Beta(α, β)` distribution. At replay, sample once per `(prompt, model)`. The simulator's whole point is that we *know* the right answer.

**Metrics per policy:**
- Total cost ($), p50 / p95 / p99 latency (ms), mean quality (0–1), cost-per-unit-quality, SLO violation rate (% over `sla_ms`), fallback invocation rate, cache hit rate.

**Baselines (reference bounds):**
- `random` — quality lower bound, mid cost
- `always_cheapest` — cost floor, quality floor
- `always_best` — quality ceiling, cost ceiling
- `oracle` — knows ground-truth quality, picks per-prompt optimum given objective. Upper bound for any router.

**Headline output:** Pareto plot, cost-per-1k-req on x-axis, mean-quality on y-axis. Each policy is a point. A working router sits on the upper-left frontier above `random`/`always_cheapest` and approaches `oracle`.

---

## 8. Build order (Phase 1 priority)

1. `llmrouter/core/types.py` — Protocols, `ModelMeta`, `Objective`, `Response`, `Features`.
2. `llmrouter/core/registry.py` — decorator registry.
3. `llmrouter/providers/mock.py` — async mock with latency + failure simulation.
4. `llmrouter/core/pipeline.py` — orchestrates extract → cache → predict → rerank → call → store → log.
5. `llmrouter/api/{main,routes,schemas}.py` — FastAPI surface.
6. `benchmark/{ground_truth,runner}.py` + `benchmark/prompts/labeled_v1.jsonl`.
7. `tests/test_pipeline.py` — end-to-end test, deterministic seed.

---

## 9. Verification

```bash
uv pip install -e .
pytest -q

uvicorn llmrouter.api.main:app --reload
curl -X POST localhost:8000/route -H content-type:application/json \
  -d '{"prompt":"What is 2+2?","objective":"cost"}'

# Phase 1 success criterion
python -m benchmark.runner --config config/default.yaml --prompts benchmark/prompts/labeled_v1.jsonl
# → benchmark/results/run_<ts>.md  + pareto.png
#
# pass conditions:
#   oracle ≥ rule_based ≥ random            on mean_quality
#   cheapest ≤ rule_based(cost) ≤ best      on total_cost
#   pareto/balanced sit on the frontier
```

---

## 10. Open questions / iterate-on-these

1. **Difficulty signal source** — keyword-rules (Phase 1) vs. small classifier (Phase 2)? Rules are fast to ship but biased; classifier is more honest but adds a model dependency.
2. **Embedding cost accounting** — count embedding latency in measured request latency? (Argument: yes — be honest. A real deployment pays it.)
3. **Bandit feedback signal** — quality is observable in the simulator (we sampled it), but in the real world it's not. For Phase 3, simulate the realistic case where the bandit only sees a delayed/noisy quality proxy?
4. **Tenant / budget scope** — single global budget vs. per-tenant in YAML? Tier-3 territory but worth deciding before we lock the schema.
5. **Streaming** — do we simulate token streaming (TTFT vs total)? Affects latency-policy semantics. Probably Phase 2/3.
6. **Tokenizer** — `tiktoken` cl100k_base for cost accounting (matches OpenAI tables, free, no model load). Confirm.
7. **Async everywhere** — assumed yes (latency simulation needs it). Confirm.
8. **Where does ground-truth live?** — `benchmark/ground_truth.py` only. The runtime pipeline never reads it (except via the explicit `oracle` predictor used as a baseline). Keeps the simulator honest.
