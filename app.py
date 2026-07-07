"""Streamlit dashboard for the Project Sentinel MVP and captured logs evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from experiments.analyze_captured import analyze_captured_rows
from experiments.run_mvp import SCENARIOS, compute_metrics, run_all_experiments
from sentinel.data import load_prompt_cases
from sentinel.responses import read_captured_response_rows

CAPTURED_PATH = Path("outputs/captured_responses.csv")
REAL_MODEL_PATH = Path("outputs/real_model_responses.csv")

st.set_page_config(page_title="Project Sentinel Dashboard", layout="wide")

# Dashboard Header
st.markdown("""
# 🛡️ Project Sentinel: Semantic Reliability Dashboard
*Continuous, unsupervised behavioral monitoring for Production Large Language Models.*

Sentinel extracts raw behavioral signals directly from LLM outputs to detect quality degradation, hallucinations, and instruction-following failures **without requiring ground-truth labels or secondary LLM judges.**
""")

# Glossary of the 5 Behavioral Signals
with st.expander("📖 Glossary: What are the 5 Behavioral Signals?"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        *   **Semantic Consistency:** Measures if different runs of the same prompt yield similar meaning. If the model becomes chaotic or repeats template errors, consistency crashes.
        *   **Response Stability:** Measures length variance. If the model's output lengths fluctuate wildly for similar queries, it indicates instability.
        *   **Embedding-Space Drift:** Coordinates of the model's outputs mapped on a semantic meaning map. If the topics drift away from the baseline, it indicates a semantic shift.
        """)
    with col2:
        st.markdown("""
        *   **Confidence Proxy:** Scans the text for logical uncertainty and hedging terms (like *maybe*, *contradiction*, *paradox*, *impossible*).
        *   **Task Compliance:** Verifies structural and logical instructions (like word counts and letter constraints) using dynamic pattern matching.
        """)

st.sidebar.header("Experiment Configurations")
mode = st.sidebar.radio("Evaluation Mode", ["Synthetic MVP", "Captured outputs"])
scenario = st.sidebar.selectbox("Filter Scenario", ["all", *SCENARIOS])

# Selectbox to toggle embedding backend
embedding_backend = st.sidebar.selectbox(
    "Embedding Backend",
    ["Hashed Bag-of-Words (Fast)", "Sentence-Transformers (Premium)"]
)

if "Hashed" in embedding_backend:
    os.environ["SENTINEL_EMBEDDINGS"] = "hashed"
else:
    os.environ["SENTINEL_EMBEDDINGS"] = "sentence-transformers"

# Loading data
if mode == "Synthetic MVP":
    dataset = st.sidebar.selectbox("Dataset", ["sample", "truthfulqa", "gsm8k"])
    limit = st.sidebar.slider("Prompt limit", min_value=4, max_value=64, value=16, step=4)
    cases = load_prompt_cases(source=dataset, limit=limit)
    source_labels = sorted({case.source for case in cases})
    dataset_label = dataset if source_labels == [dataset] else f"{dataset}({'+'.join(source_labels)})"
    rows = run_all_experiments(cases=cases, dataset_label=dataset_label)
    prompt_preview = pd.DataFrame([case.__dict__ for case in cases]).head(10)
    st.caption(f"Synthetic mode. Dataset source: {dataset_label}. Prompt cases loaded: {len(cases)}.")
else:
    outputs_dir = Path("outputs")
    csv_files = []
    if outputs_dir.exists():
        csv_files = sorted(list(outputs_dir.glob("*responses.csv")))
    
    if not csv_files:
        csv_files = [REAL_MODEL_PATH if REAL_MODEL_PATH.exists() else CAPTURED_PATH]
        
    csv_options = [str(p) for p in csv_files]
    selected_option = st.sidebar.selectbox("Select Captured Log", csv_options)
    capture_path = Path(selected_option)
    
    if not capture_path.exists():
        st.warning(f"Captured response file not found: {capture_path}")
        st.stop()
        
    captured_rows = read_captured_response_rows(capture_path)
    rows = analyze_captured_rows(captured_rows)
    prompt_preview = pd.DataFrame(captured_rows)[["case_id", "question", "provider", "model"]].drop_duplicates().head(10)
    st.caption(f"Captured-output mode. Loaded {len(captured_rows)} raw monitored-model responses from {capture_path}.")

metrics = compute_metrics(rows)
df = pd.DataFrame(rows)
metrics_df = pd.DataFrame(metrics)

visible_df = df if scenario == "all" else df[df["scenario"] == scenario]

# Active Alert Warning Banner
if not visible_df.empty:
    alert_windows = visible_df[visible_df["degradation_alert"] == True]["window"].unique().tolist()
    if alert_windows:
        st.error(f"🚨 **ALERT ACTIVE:** Sentinel detected semantic degradation in Window(s): {', '.join(map(str, alert_windows))}!")
    else:
        st.success("✅ **SYSTEM HEALTHY:** No semantic degradation alerts detected.")

# Statistics KPIs Card
st.subheader("Classification Performance Metrics")
if not visible_df.empty:
    matching_metrics = metrics_df[metrics_df["scenario"] == scenario]
    if not matching_metrics.empty:
        visible_metrics = matching_metrics.iloc[0]
        
        # Check if actual degradation exists in the selected slice to handle 0/NaN statistics gracefully
        has_positives = visible_df["actual_degraded"].any()
        
        left, middle, right, fourth = st.columns(4)
        left.metric("Accuracy", f"{visible_metrics['accuracy']:.2f}")
        
        # Gracefully handle Precision showing 0.0 on healthy-only benchmarks
        if has_positives or visible_metrics['true_positive'] > 0 or visible_metrics['false_positive'] > 0:
            precision_str = f"{visible_metrics['precision']:.2f}"
        else:
            precision_str = "N/A (All Healthy)"
        middle.metric("Precision", precision_str, help="Precision is not applicable when there are no degradation alerts.")
        
        # Gracefully handle Recall showing 0.0 on healthy-only benchmarks
        if has_positives:
            recall_str = f"{visible_metrics['recall']:.2f}"
        else:
            recall_str = "N/A (All Healthy)"
        right.metric("Recall", recall_str, help="Recall is not applicable when there are no actual degradation windows.")
        
        fourth.metric("False Positive Rate (FPR)", f"{visible_metrics['false_positive_rate']:.2f}")
    else:
        st.warning(f"No metrics available for scenario: {scenario}")
else:
    st.info("No data available to display metrics.")

# Time-series charts tabs
st.subheader("Reliability Trends Over Time")
if not visible_df.empty:
    tab1, tab2 = st.tabs(["Fused Reliability Score (SRS)", "Individual Signals Trend"])
    
    with tab1:
        st.markdown("Shows the overall **Fused Semantic Reliability Score (SRS)**. Safe baseline is `>0.85`.")
        chart_df = visible_df.pivot(index="window", columns="scenario", values="semantic_reliability_score")
        st.line_chart(chart_df)
        
    with tab2:
        st.markdown("Shows the five behavioral signals plotted together. Use this correlation to see exactly *which* signal triggered a crash.")
        signal_columns = [
            "semantic_consistency",
            "response_stability",
            "embedding_drift",
            "confidence_proxy",
            "task_compliance",
        ]
        if scenario == "all":
            st.info("Please select a specific scenario in the sidebar to view individual signal correlation trends.")
        else:
            chart_signals = visible_df.set_index("window")[signal_columns]
            st.line_chart(chart_signals)
else:
    st.info("No data to plot.")

# Dataframes breaking down signals
st.subheader("Logged Signal Details")
if not visible_df.empty:
    st.dataframe(
        visible_df[
            [
                "dataset",
                "scenario",
                "window",
                "actual_degradation",
                "semantic_consistency",
                "response_stability",
                "embedding_drift",
                "confidence_proxy",
                "task_compliance",
                "semantic_reliability_score",
                "degradation_alert",
            ]
        ],
        use_container_width=True,
    )

st.subheader("Loaded Prompt Preview")
st.dataframe(prompt_preview, use_container_width=True)

# Comparisons tab if benchmark results are cached
benchmark_csv = Path("outputs/embedding_benchmark_results.csv")
if benchmark_csv.exists():
    st.subheader("Embedding Backend Comparison")
    st.markdown(
        "This table compares the detection performance of **Hashed Bag-of-Words** and **Sentence-Transformers** "
        "across different degradation scenarios."
    )
    comparison_df = pd.read_csv(benchmark_csv)
    st.dataframe(comparison_df, use_container_width=True)

    plot_df = comparison_df[comparison_df["scenario"] != "all"]
    if not plot_df.empty:
        chart_data = plot_df.pivot(index="scenario", columns="embedding_backend", values="accuracy")
        st.caption("Accuracy Comparison by Scenario")
        st.bar_chart(chart_data)
else:
    st.subheader("Embedding Backend Comparison")
    st.info(
        "No embedding benchmark data found. To generate comparison data, run the benchmark script in your terminal:\n\n"
        "`python experiments/run_embedding_benchmark.py`"
    )

st.subheader("System Boundaries")
st.write(
    "Sentinel does not use an LLM judge or answer verifier. In captured-output mode, "
    "the selected CSV contains raw outputs from the model being monitored; Sentinel then "
    "computes reliability signals algorithmically."
)
