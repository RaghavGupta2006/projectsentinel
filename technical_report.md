# Project Sentinel: Semantic Reliability Inference for Large Language Models

## 1. Abstract
As large language models (LLMs) are deployed in production, detecting silent performance degradation remains a critical challenge. Standard monitoring strategies rely on human labeling or heavy LLM-as-a-judge evaluators, which are expensive, introduce judgment bias, and fail to operate in real-time. Project Sentinel proposes an algorithmic framework to infer LLM semantic reliability over time without using ground-truth labels or secondary LLM judge systems. By fusing behavioral indicators (semantic consistency, response stability, embedding centroid drift, confidence proxies, and task compliance rules) into a unified Semantic Reliability Score (SRS), Sentinel flags degradation points in streaming output windows using thresholding and rolling Z-score anomaly detection.

---

## 2. Core Problem
Modern LLM applications fail silently. When a model's underlying weights change, input prompt patterns regress, or context windows truncate, the model may generate answers that look structurally sound but are semantically incorrect, hallucinated, or evasive.

Traditional system monitors check:
*   API latency
*   Throughput
*   HTTP errors
*   CPU/GPU saturation

Observability platforms check:
*   User thumbs up/down (sparse and delayed)
*   LLM-as-a-Judge ratings (expensive, slow, and replaces one model's uncertainty with another's)

**Project Sentinel's objective:** Detect degradation using the monitored model's outputs alone, treating semantic drift as a statistical anomaly in behavioral metrics.

---

## 3. Mathematical Signals

Sentinel extracts five behavioral signals from output text arrays within each streaming window $W$ compared to a normal baseline window $W_{base}$.

### 3.1 Semantic Consistency
Measures if the model responds consistently when asked the same prompt multiple times (using prompt perturbations or variants). Let $V(p, v)$ be the embedding vector of response variant $v$ for prompt $p$.
$$\text{Semantic Consistency} = \text{clamp}\left( \frac{\overline{\text{Sim}}_{\text{cosine}}(V_i, V_j) + 1.0}{2.0} \right)$$
where similarity is computed pairwise across all generated response variants for each prompt.

### 3.2 Response Stability
Measures how much the response length varies across variants, indicating structural uncertainty. For book-keeping, length $L$ is measured in word tokens.
$$\text{Response Stability} = 1.0 - \overline{\left( \frac{\sigma(L)}{\mu(L)} \right)}$$
where $\frac{\sigma}{\mu}$ is the coefficient of variation of token counts across variants.

### 3.3 Embedding Drift
Measures whether the semantic centroid of the current window has drifted away from the baseline centroid.
$$\mathbf{C}_W = \frac{1}{|W|} \sum_{i \in W} \mathbf{v}_i$$
$$\text{Embedding Drift} = \text{clamp}\left( \frac{\text{Sim}_{\text{cosine}}(\mathbf{C}_W, \mathbf{C}_{W_{\text{base}}}) + 1.0}{2.0} \right)$$

### 3.4 Confidence Proxy
Tracks the prevalence of hedging terms $T_{hedge}$ (e.g., *maybe*, *probably*, *unknown*, *uncertain*, *incomplete*) in the response tokens.
$$\text{Hedge Rate} = \frac{\sum \mathbb{I}(\text{token} \in T_{hedge})}{N_{\text{tokens}}}$$
$$\text{Confidence Proxy} = \text{clamp}\left( 1.0 - \sqrt{8.0 \times \text{Hedge Rate}} \right)$$

### 3.5 Task Compliance
Evaluates rule-based failures (evasive catchphrases, responses that are too short, or generic claims like *"everyone knows this"*).
$$\text{Task Compliance} = 1.0 - \frac{N_{\text{failures}}}{|W|}$$

---

## 4. Semantic Reliability Score (SRS) & Anomaly Detection

### 4.1 Score Fusion
The signals are linearly fused into a single unified score:
$$\text{SRS} = w_1 \cdot \text{Consistency} + w_2 \cdot \text{Stability} + w_3 \cdot \text{Drift} + w_4 \cdot \text{Proxy} + w_5 \cdot \text{Compliance}$$
Default weights: $w = [0.25, 0.15, 0.25, 0.15, 0.20]$.

### 4.2 Anomaly Detection Algorithms

1.  **Hybrid Absolute/Relative Thresholding**:
    Flags degradation if the SRS drops below an absolute threshold (default $0.70$) OR if the drop relative to the baseline is too large:
    $$\text{SRS}_W < \text{Threshold}_{\text{abs}} \quad \lor \quad (\text{SRS}_{W_{\text{base}}} - \text{SRS}_W) \ge \text{Drop}_{\text{rel}}$$
2.  **Rolling Z-Score Anomaly Detection**:
    Flags anomalies dynamically based on historical variance, removing the need for manual threshold tuning:
    $$\text{Z}_W = \frac{\text{SRS}_W - \mu(\text{SRS}_{\text{history}})}{\sigma(\text{SRS}_{\text{history}})}$$
    Alert triggers when $\text{Z}_W < -k$ (default $k=2.0$).

---

## 5. Summary of Findings

*   **Real-Model Benchmark Evaluation**: Tested Sentinel on a real local `gemma2:2b` model across all scenarios (48 total generations). Sentinel achieved an overall detection accuracy of 0.8333.
    *   **Detected Scenarios**: Prompt corruption was successfully detected immediately (1.0 accuracy, 1.0 recall, 0.0 false positive rate).
    *   **Missed Scenarios**: Context truncation and temperature spikes were missed under the default absolute threshold of 0.70 because the model's reliability scores remained at 0.7734 and 0.7704 respectively. This empirical gap highlights the importance of adopting dynamic rolling Z-score thresholds to detect subtler distribution shifts.
*   **Real-Model Ablation Studies**: Evaluated detection variants on `gemma2:2b` output datasets. Individual signals were found to be highly noisy in isolation. For instance, evaluating semantic consistency alone resulted in a false positive rate of 1.0 (flagging normal windows). Fusing them into the unified SRS successfully suppressed this background noise, achieving a false positive rate of 0.0.
*   **Embeddings Comparison**: Validated that hashed bag-of-words vectors provide extremely fast, offline, and deterministic benchmarks, while pre-trained Sentence-Transformer models (`all-MiniLM-L6-v2`) capture nuanced semantic drift on real model output datasets.

