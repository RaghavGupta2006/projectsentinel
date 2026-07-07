"""Small Streamlit dashboard for the Project Sentinel MVP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.analyze_captured import analyze_captured_rows
from experiments.run_mvp import SCENARIOS, compute_metrics, run_all_experiments
from sentinel.data import load_prompt_cases
from sentinel.responses import read_captured_response_rows


CAPTURED_PATH = Path("outputs/captured_responses.csv")
REAL_MODEL_PATH = Path("outputs/real_model_responses.csv")

st.set_page_config(page_title="Project Sentinel", layout="wide")

st.title("Project Sentinel")
st.caption("Unlabeled semantic degradation detection for monitored LLM outputs")

st.sidebar.header("Experiment")
mode = st.sidebar.radio("Mode", ["Synthetic MVP", "Captured outputs"])
scenario = st.sidebar.selectbox("Scenario", ["all", *SCENARIOS])

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
    default_capture_path = REAL_MODEL_PATH if REAL_MODEL_PATH.exists() else CAPTURED_PATH
    capture_path_text = st.sidebar.text_input("Captured CSV", str(default_capture_path))
    capture_path = Path(capture_path_text)
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

if visible_df.empty:
    st.warning(f"No data available for scenario: {scenario}")
else:
    matching_metrics = metrics_df[metrics_df["scenario"] == scenario]
    if not matching_metrics.empty:
        visible_metrics = matching_metrics.iloc[0]
        left, middle, right, fourth = st.columns(4)
        left.metric("Accuracy", f"{visible_metrics['accuracy']:.2f}")
        middle.metric("Precision", f"{visible_metrics['precision']:.2f}")
        right.metric("Recall", f"{visible_metrics['recall']:.2f}")
        fourth.metric("False Positive Rate", f"{visible_metrics['false_positive_rate']:.2f}")
    else:
        st.warning(f"No metrics available for scenario: {scenario}")

st.subheader("Reliability Over Time")
if not visible_df.empty:
    chart_df = visible_df.pivot(index="window", columns="scenario", values="semantic_reliability_score")
    st.line_chart(chart_df)
else:
    st.info("No data to plot.")

st.subheader("Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Signal Breakdown")
signal_columns = [
    "semantic_consistency",
    "response_stability",
    "embedding_drift",
    "confidence_proxy",
    "task_compliance",
]
st.dataframe(
    visible_df[
        [
            "dataset",
            "scenario",
            "window",
            "actual_degradation",
            *signal_columns,
            "semantic_reliability_score",
            "degradation_alert",
        ]
    ],
    use_container_width=True,
)

st.subheader("Prompt Sample")
st.dataframe(prompt_preview, use_container_width=True)

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

st.subheader("Boundary")
st.write(
    "Sentinel does not use an LLM judge or answer verifier. In captured-output mode, "
    "the selected CSV contains raw outputs from the model being monitored; Sentinel then "
    "computes reliability signals algorithmically."
)


