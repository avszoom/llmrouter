"""Streamlit demo UI for the LLM router.

Run from the repo root:
    streamlit run ui/app.py

Layout:
    - Top: prompt input + Route button
    - Sidebar: routing policy + optional latency SLA
    - 3 columns: classifier output | routing decision | cost estimate
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable when streamlit is launched from anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability.loader import load_capability_table
from classifier import Classifier
from router.engine import ALL_POLICIES, Objective, count_input_tokens, route


# ─── caching ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading classifier (one-time, ~5s)…")
def get_classifier():
    return Classifier(artifacts_dir=ROOT / "classifier" / "artifacts")


@st.cache_resource
def get_models():
    return load_capability_table(ROOT / "llm_router_poc_spec_and_dataset" / "model_capability_table_seed.csv")


# ─── page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Inference Router POC",
    layout="wide",
    page_icon="⚡",
)

st.title("⚡ LLM Inference Router")
st.caption(
    "Prompt-aware routing: classifier predicts what the prompt needs → rule-based router "
    "picks the cheapest/fastest model that meets the bar."
)

# ─── sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Routing objective")
    policy = st.selectbox(
        "Policy",
        ALL_POLICIES,
        index=ALL_POLICIES.index("balanced"),
        help="cost_first / latency_first / quality_first pick a single dimension. "
             "balanced uses 0.55·Q − 0.25·Cnorm − 0.20·Lnorm.",
    )
    enforce_sla = st.checkbox("Enforce latency SLA", value=False)
    sla_ms = st.number_input(
        "Latency SLA (ms)",
        min_value=100,
        max_value=10000,
        value=2500,
        step=100,
        disabled=not enforce_sla,
    )
    objective = Objective(
        policy=policy,
        latency_sla_ms=int(sla_ms) if enforce_sla else None,
    )

    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        "1. **Embed** — MiniLM (384-dim)\n"
        "2. **Classify** — 6 sklearn heads predict prompt needs\n"
        "3. **Filter** — capability_quality ≥ required_quality, latency ≤ SLA\n"
        "4. **Rank** — by selected policy"
    )

    st.divider()
    st.markdown("### Try")
    examples = {
        "Easy / writing": "Write a short birthday message for a coworker.",
        "Medium / coding": "Write a Python function to deduplicate a list while preserving order.",
        "Hard / system_design": "Design a distributed cache with strong consistency and multi-region failover.",
        "Easy / factual": "What is the capital of Australia?",
        "Hard / math": "Prove that the sum of two odd integers is even.",
    }
    chosen_example = st.selectbox("Example prompt", ["(none)"] + list(examples.keys()))


# ─── prompt input ───────────────────────────────────────────────────────

default_prompt = examples[chosen_example] if chosen_example != "(none)" else ""
prompt = st.text_area(
    "Prompt",
    value=default_prompt,
    height=120,
    placeholder="e.g., Design a real-time chat system with multi-region failover…",
)
go = st.button("Route", type="primary", disabled=not prompt.strip())


# ─── results ────────────────────────────────────────────────────────────

if go:
    with st.spinner("Classifying and routing…"):
        clf = get_classifier()
        models = get_models()
        classifier_output = clf.predict(prompt)
        input_tokens = count_input_tokens(prompt)
        decision = route(classifier_output, models, objective, input_tokens)

    col_clf, col_rt, col_cost = st.columns(3, gap="large")

    # ── Classifier panel ────────────────────────────────────────────
    with col_clf:
        st.subheader("🔍 Classifier")
        st.metric("task_type", classifier_output["task_type"])
        diff = float(classifier_output["difficulty"])
        st.metric(
            "difficulty",
            f"{diff:.2f}",
            help=f"bucket = {classifier_output['difficulty_bucket']}",
        )
        st.progress(min(1.0, max(0.0, diff)))
        rq = float(classifier_output["required_quality"])
        st.metric("required_quality", f"{rq:.2f}")
        st.progress(min(1.0, max(0.0, rq)))
        st.metric("risk_level", classifier_output["risk_level"])
        st.metric("latency_sensitivity", classifier_output["latency_sensitivity"])
        st.metric("expected_output_tokens", classifier_output["expected_output_tokens"])

        with st.expander("Class probabilities"):
            for key in ("task_type_proba", "risk_level_proba", "latency_sensitivity_proba"):
                if key in classifier_output:
                    st.markdown(f"**{key.replace('_proba','')}**")
                    sorted_probs = sorted(
                        classifier_output[key].items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    for label, p in sorted_probs:
                        st.write(f"- {label}: {p:.3f}")

    # ── Routing decision panel ──────────────────────────────────────
    with col_rt:
        st.subheader("🎯 Routing decision")
        if decision.fallback:
            st.error(f"**{decision.selected_model}** (fallback)")
        else:
            st.success(f"**{decision.selected_model}**")
        st.write(f"Provider: `{decision.provider}`")
        st.write(f"Policy: `{decision.policy}`")
        st.info(decision.routing_reason)

        st.markdown(f"**Eligible** ({len(decision.eligible)})")
        if decision.eligible:
            for name in decision.eligible:
                marker = " ← selected" if name == decision.selected_model else ""
                st.write(f"- `{name}`{marker}")
        else:
            st.write("_none_")

        if decision.rejected:
            with st.expander(f"Rejected ({len(decision.rejected)})"):
                for name, reason in decision.rejected.items():
                    st.write(f"- `{name}` — {reason}")

    # ── Cost panel ──────────────────────────────────────────────────
    with col_cost:
        st.subheader("💰 Cost & latency")
        st.metric("estimated_cost_usd", f"${decision.estimated_cost_usd:.6f}")
        st.metric("estimated_latency_ms", f"{decision.estimated_latency_ms} ms")
        st.metric("estimated_quality", f"{decision.estimated_quality:.2f}")

        st.divider()
        st.markdown("**Token accounting**")
        st.write(f"Input tokens: `{decision.estimated_input_tokens}`")
        st.write(f"Expected output tokens: `{decision.estimated_output_tokens}`")

        # Cost breakdown using the selected model's pricing
        chosen = next((m for m in get_models() if m.name == decision.selected_model), None)
        if chosen is not None:
            in_cost = decision.estimated_input_tokens * chosen.input_usd_per_1m / 1_000_000
            out_cost = decision.estimated_output_tokens * chosen.output_usd_per_1m / 1_000_000
            st.markdown("**Pricing (USD per 1M tokens)**")
            st.write(f"- input: ${chosen.input_usd_per_1m:.3f}/1M → **${in_cost:.6f}**")
            st.write(f"- output: ${chosen.output_usd_per_1m:.3f}/1M → **${out_cost:.6f}**")
            st.caption(f"Total = input + output = ${in_cost + out_cost:.6f}")

else:
    st.info(
        "Enter a prompt and click **Route**. The first run loads the embedding model "
        "(~5s); after that, each request is ~15–25 ms."
    )
