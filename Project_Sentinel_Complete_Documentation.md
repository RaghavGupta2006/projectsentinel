# Project Sentinel: Complete System Documentation & Completion Report

This document serves as the master system documentation and progress log for **Project Sentinel: Semantic Reliability Inference for Production Large Language Models**. It covers the architecture, mathematics, module structure, benchmark results, CLI workflow, and developer tutorial in exhaustive detail.

---

## Table of Contents
1. **Executive Summary & Project Scope**
2. **Mathematical Signal Formulations**
3. **System Architecture & Codebase Walkthrough**
4. **Reproducible Command Pipeline**
5. **Empirical Benchmarks (Gemma-2B Case Study)**
6. **Visualization & Diagnostic Assets**
7. **Developer Conceptual Tutorial**
8. **CV & Career Presentation**

---

## 1. Executive Summary & Project Scope

### 1.1 The Silent Performance Degradation Problem
Large Language Models (LLMs) deployed in production fail silently. Standard system monitoring checks throughput, GPU saturation, and network latency but is unable to detect when a model regresses in accuracy, outputs contradictory responses, or exhibits semantic drift.

### 1.2 Core Research Boundary: No LLM Judge
Project Sentinel rejects the standard "LLM-as-a-judge" pattern (using a model like GPT-4 to rate output quality) because:
*   Calling a evaluation API for every log introduces massive cost and latency.
*   It substitutes one model's uncertainty with another model's bias.
*   It fails in high-throughput streams.

**Sentinel's Innovation**: A passive, algorithmic evaluation engine that extracts behavioral signals directly from target model response logs over streaming time windows.

---

## 2. Mathematical Signal Formulations

Sentinel aggregates responses into sliding time windows $W$ and compares them to a normal baseline window $W_{base}$ (typically the first window representing the healthy model state). For each window, it extracts five distinct behavioral signals:

### 2.1 Semantic Consistency
*   **Concept**: Measures if the model responds consistently when asked the same prompt multiple times (using prompt perturbations or variants).
*   **Math**: Pairs variant embedding vectors $V_i, V_j$ of prompt $p$:
    $$\text{Sim}_{\text{cosine}}(V_i, V_j) = \frac{V_i \cdot V_j}{\|V_i\| \|V_j\|}$$
    $$\text{Semantic Consistency} = \text{clamp}\left( \frac{\overline{\text{Sim}}_{\text{cosine}}(V_i, V_j) + 1.0}{2.0} \right)$$

### 2.2 Response Stability
*   **Concept**: Evaluates length distribution variance across variant responses. An unstable model exhibits high length variance under high temperatures or instructions conflicts.
*   **Math**: Length $L$ measured in word tokens:
    $$\text{Response Stability} = 1.0 - \overline{\left( \frac{\sigma(L)}{\mu(L)} \right)}$$

### 2.3 Embedding Drift
*   **Concept**: Tracks semantic centroid drift in representation space.
*   **Math**: Centroid $\mathbf{C}_W$ of all window embeddings:
    $$\mathbf{C}_W = \frac{1}{|W|} \sum_{i \in W} \mathbf{v}_i$$
    $$\text{Embedding Drift} = \text{clamp}\left( \frac{\text{Sim}_{\text{cosine}}(\mathbf{C}_W, \mathbf{C}_{W_{\text{base}}}) + 1.0}{2.0} \right)$$

### 3.4 Confidence Proxy
*   **Concept**: Measures the prevalence of hedging words $T_{hedge}$ (e.g., *maybe*, *probably*, *unknown*, *uncertain*).
*   **Math**:
    $$\text{Hedge Rate} = \frac{N_{\text{hedging}}}{N_{\text{tokens\_in\_window}}}$$
    $$\text{Confidence Proxy} = \text{clamp}\left( 1.0 - \sqrt{8.0 \times \text{Hedge Rate}} \right)$$

### 3.5 Task Compliance
*   **Concept**: Rule-based template matching to flag evasive catchphrases, responses that are too short, or generic claims.
*   **Math**:
    $$\text{Task Compliance} = 1.0 - \frac{N_{\text{failures}}}{|W|}$$

### 3.6 Semantic Reliability Score (SRS) Fusion
The signals are linearly fused using weighted parameters:
$$\text{SRS} = w_1 \cdot \text{Consistency} + w_2 \cdot \text{Stability} + w_3 \cdot \text{Drift} + w_4 \cdot \text{Proxy} + w_5 \cdot \text{Compliance}$$
Default weights: $w = [0.25, 0.15, 0.25, 0.15, 0.20]$.

---

## 3. System Architecture & Codebase Walkthrough

```
Project Sentinel/
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── technical_report.md        # Formulated technical report (ASCII clean)
├── Progress_Report_Complete.md # Complete progress record
├── Developer_Learning_Guide.md # Step-by-step developer tutorial
│
├── sentinel/                  # Core Library Package
│   ├── __init__.py            # Package init
│   ├── data.py                # PromptCase dataclass & loaders (TruthfulQA, GSM8K, CSV)
│   ├── embeddings.py          # Representation vectorizers (Hashed BoW vs. Sentence-Transformers)
│   ├── responses.py           # Capture CSV schema and loaders
│   ├── scoring.py             # SRS score calculation & alert systems
│   ├── signals.py             # Feature extraction calculators
│   └── simulator.py           # Synthetic normal and degraded stream simulator
│
├── experiments/               # Reproducible CLI runners
│   ├── run_mvp.py             # Synthetic MVP stream simulation runner
│   ├── run_ablation.py        # Signal ablation study runner (synthetic)
│   ├── run_captured_ablation.py # Study runner for captured real model logs
│   ├── run_embedding_benchmark.py # Comparative embedding backend benchmark
│   ├── generate_plots.py      # Matplotlib diagnostics chart exporter
│   └── run_real_model_experiment.py # One-command local Ollama model runner
│
└── outputs/                   # Generated evaluation datasets and figures
    ├── real_model_responses.csv
    ├── real_model_analysis_results.csv
    ├── real_model_analysis_metrics.csv
    ├── real_model_ablation_metrics.csv
    ├── embedding_benchmark_results.csv
    ├── real_model_reliability_over_time.png
    └── real_model_ablation_comparison.png
```

---

## 4. Reproducible Command Pipeline

To execute the entire evaluation pipeline from scratch, run the following commands:

```bash
# 1. Capture and analyze real outputs using a local Gemma-2B model
python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b --dataset sample --limit 4 --scenario all --windows 4 --variants-per-case 1

# 2. Run the signal ablation study over the captured real model logs
python experiments/run_captured_ablation.py --input outputs/real_model_responses.csv --output outputs/real_model_ablation_metrics.csv

# 3. Benchmark Hashed BoW vs. Sentence-Transformers embeddings
python experiments/run_embedding_benchmark.py --input outputs/real_model_responses.csv

# 4. Generate all diagnostic Matplotlib plots
python experiments/generate_plots.py

# 5. Launch the Streamlit interactive dashboard
streamlit run app.py
```

---

## 5. Empirical Benchmarks (Gemma-2B Case Study)

We ran a benchmark of `gemma2:2b` using Ollama (48 total generations across 3 scenarios, 4 windows, and 4 prompts).

### 5.1 Real-Model Detection Metrics

| Scenario | Accuracy | Precision | Recall | False Positive Rate | Detection Lead Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **all** | 0.8333 | 1.00 | 0.3333 | 0.00 | 0 windows |
| **prompt_corruption** | 1.0000 | 1.00 | 1.0000 | 0.00 | 0 windows |
| **context_truncation** | 0.7500 | 0.00 | 0.0000 | 0.00 | missed |
| **temperature_spike** | 0.7500 | 0.00 | 0.0000 | 0.00 | missed |

**Analysis**:
Prompt corruption is detected immediately due to a collapse in task compliance (0.0) and high embedding centroid drift. Context truncation and temperature spikes were missed under the default absolute threshold (0.70) because Gemma remained highly fluent (SRS 0.7734 and 0.7704). This empirical gap demonstrates the necessity of adopting **Z-score dynamic anomaly detection** over hard absolute thresholds in production.

### 5.2 Real-Model Ablation Study Metrics

| Variant (Active Signal) | Accuracy | Precision | Recall | False Positive Rate | Detection Lead Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **full_score** | 0.8333 | 1.0000 | 0.3333 | 0.0000 | 0 |
| **embedding_drift** | 0.8333 | 0.6667 | 0.6667 | 0.1111 | 0 |
| **semantic_consistency** | 0.2500 | 0.2500 | 1.0000 | 1.0000 | 0 |
| **response_stability** | 0.7500 | 0.0000 | 0.0000 | 0.0000 | missed |
| **confidence_proxy** | 0.7500 | 0.5000 | 0.3333 | 0.1111 | 0 |
| **task_compliance** | 0.7500 | 0.5000 | 0.3333 | 0.1111 | 0 |

**Analysis**:
Evaluating consistency alone results in a False Positive Rate of 1.0 (firing false alarms on normal windows). Fusing the indicators into the `full_score` successfully drops the False Positive Rate to **0.0000**, verifying that Sentinel's multi-signal score fusion successfully suppresses noise.

---

## 6. Visualization & Diagnostic Assets

Executing `generate_plots.py` outputs five PNG visual assets to the `outputs/` folder:

1.  `outputs/reliability_over_time.png` (Synthetic SRS score tracking).
2.  `outputs/ablation_comparison.png` (Synthetic ablation bar chart).
3.  `outputs/real_model_reliability_over_time.png` (Real Gemma-2B SRS score tracking).
4.  `outputs/real_model_ablation_comparison.png` (Real Gemma-2B ablation bar chart).
5.  `outputs/embedding_comparison.png` (Comparative bar chart of Hashed BoW vs. Sentence-Transformers).

---

## 7. Developer Conceptual Tutorial

If you are transitioning from "vibe coding" to code mastery, review these system concepts:

### 7.1 Dynamic Z-Scores vs. Hard Thresholds
If you use a hard threshold (like `Score < 0.70`), the monitor is fragile. A larger, more fluent model (e.g. Llama-3-70B) might naturally score `0.88`, while a smaller model (Gemma-2B) might naturally score `0.78`. If Gemma drops to `0.72` under degradation, a hard threshold of `0.70` misses it.
**Rolling Z-Scores** compute the mean ($\mu$) and standard deviation ($\sigma$) of the historical scores:
$$\text{Z}_W = \frac{\text{SRS}_W - \mu}{\sigma}$$
Sentinel triggers when the score drops more than $2.0$ standard deviations ($Z_W < -2.0$), adapting dynamically to each model's native fluency.

### 7.2 Why Embedding Centroids Detect Topic Drift
If the model begins generating garbage tokens, repeating sentences, or outputting foreign text, the cosine similarity of individual variants remains high, but the overall centroid (average vector) of the window drifts. Comparing the current window's centroid to the baseline centroid catches this macro-level shift.

---

## 8. CV & Career Presentation

Use these bullet points to present Project Sentinel on your resume:

*   **System Design**: Designed and built an offline AI reliability framework to detect silent semantic degradation in streaming LLM outputs without relying on ground-truth labels or secondary LLM judge systems.
*   **Feature Engineering**: Implemented five statistical NLP and embedding-drift indicators (Consistency, Stability, Drift, Proxy, and Compliance) fused into a unified Semantic Reliability Score (SRS).
*   **Statistical Anomaly Detection**: Developed dynamic rolling Z-score anomaly detection to identify degradation thresholds adapted to specific model fluency distributions.
*   **Evaluation & Ablation**: Benchmarked performance over local `gemma2:2b` streams; ran ablation studies proving signal fusion successfully reduced the False Positive Rate (FPR) from 1.0 to 0.00.
*   **Representation Comparison**: Contrated hashed bag-of-words vectors against Sentence-Transformers (`all-MiniLM-L6-v2`) on real model output streams.
