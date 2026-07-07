# Project Sentinel

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://projectsentinel.streamlit.app/)

Project Sentinel detects semantic reliability degradation in LLM-style outputs without requiring continuous ground-truth labels.

**🔗 [Live Interactive Observability Dashboard](https://projectsentinel.streamlit.app/)**

Sentinel is not an LLM judge. It does not ask a second model whether an answer is correct. It monitors the raw outputs of the model under observation and computes behavioral reliability signals such as semantic consistency, response stability, embedding drift, confidence proxies, and task compliance.


## Current Status

Implemented:

- Built-in TruthfulQA-style and GSM8K reasoning sample prompt sets.
- Optional Hugging Face TruthfulQA and GSM8K loaders with safe fallback.
- CSV prompt export/import path for reproducible experiments.
- Synthetic model-output simulator for fast offline MVP tests.
- Monitored-model response capture path (fixture, Ollama, and OpenAI-compatible).
- Captured-response analyzer using Sentinel signals only.
- Three degradation scenarios (prompt corruption, context truncation, temperature spike).
- Semantic Reliability Score (SRS) with rolling Z-score anomaly detection.
- Hashed Bag-of-Words and Sentence-Transformers embedding backends.
- Real-output embedding benchmarks and real-model ablation studies.
- Evaluation metrics (accuracy, precision, recall, false positive rate, detection lead time).
- Matplotlib plot generator exporting performance figures.
- Streamlit dashboard with synthetic and captured-output modes.

## Boundary: No LLM Judge

Sentinel should not do this:

```text
LLM A -> Answer
LLM B -> Is this answer correct?
```

That would replace one model opinion with another model opinion.

Sentinel does this instead:

```text
Monitored model -> Raw outputs over time
Sentinel -> Behavioral signal extraction
Sentinel -> Semantic Reliability Score
Sentinel -> Degradation alert
```

The model API or local model is used only as the system being monitored, not as a verifier.

## Synthetic MVP Experiment

```bash
python experiments/run_mvp.py
```

Useful options:

```bash
python experiments/run_mvp.py --dataset sample --limit 16
python experiments/run_mvp.py --dataset truthfulqa --limit 32
python experiments/run_mvp.py --dataset csv --csv-path data/prompt_cases.csv
```

This writes:

```text
outputs/mvp_results.csv
outputs/mvp_metrics.csv
```


## One-Command Real Monitored-Model Experiment

After installing Ollama and pulling a model, run:

```bash
python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b
```

Preflight check only:

```bash
python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b --check-only
```

To run a larger, full benchmark over all 3 scenarios:

```bash
python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b --dataset sample --limit 4 --scenario all --windows 4 --variants-per-case 1
```

This writes:
*   `outputs/real_model_responses.csv`
*   `outputs/real_model_analysis_results.csv`
*   `outputs/real_model_analysis_metrics.csv`

If Ollama is not installed or not running, the script explains the setup issue instead of silently failing.
Latest real-model benchmark results with `gemma2:2b` completed successfully:

| Scenario | Accuracy | Precision | Recall | False Positive Rate | Detection Lead Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **all** | 0.8333 | 1.00 | 0.3333 | 0.00 | 0 |
| **prompt_corruption** | 1.0000 | 1.00 | 1.0000 | 0.00 | 0 |
| **context_truncation** | 0.7500 | 0.00 | 0.0000 | 0.00 | missed |
| **temperature_spike** | 0.7500 | 0.00 | 0.0000 | 0.00 | missed |

*Interpretation: Prompt corruption is detected immediately. Context truncation and temperature spikes were missed under default absolute thresholds because Gemma's outputs remained above 0.70. This highlights the need for dynamic rolling Z-score detection.*

## Run Real-Model Ablation Study

Run the signal ablation study over the captured real model response CSV:

```bash
python experiments/run_captured_ablation.py --input outputs/real_model_responses.csv --output outputs/real_model_ablation_metrics.csv
```

This output study metrics:

| Variant | Accuracy | Precision | Recall | FPR | Lead Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **full_score** | 0.8333 | 1.0000 | 0.3333 | 0.0000 | 0 |
| **embedding_drift** | 0.8333 | 0.6667 | 0.6667 | 0.1111 | 0 |
| **semantic_consistency** | 0.2500 | 0.2500 | 1.0000 | 1.0000 | 0 |
| **response_stability** | 0.7500 | 0.0000 | 0.0000 | 0.0000 | missed |
| **confidence_proxy** | 0.7500 | 0.5000 | 0.3333 | 0.1111 | 0 |
| **task_compliance** | 0.7500 | 0.5000 | 0.3333 | 0.1111 | 0 |

*Interpretation: Fusing indicators in the `full_score` successfully eliminates false alerts (FPR 0.0), whereas individual signals (like `semantic_consistency`) are highly noisy in isolation (FPR 1.0).*

## Capture Monitored-Model Responses

Fixture smoke test, no network:

```bash
python experiments/capture_responses.py --provider fixture --dataset sample --limit 4 --scenario prompt_corruption --variants-per-case 2
python experiments/analyze_captured.py --input outputs/captured_responses.csv
```

Local Ollama example:

```bash
ollama pull llama3.1
python experiments/capture_responses.py --provider ollama --model llama3.1 --dataset sample --limit 8 --scenario all
python experiments/analyze_captured.py --input outputs/captured_responses.csv
```

OpenAI-compatible endpoint example:

```bash
set OPENAI_API_KEY=your_key_here
python experiments/capture_responses.py --provider openai-compatible --base-url https://api.openai.com/v1 --model gpt-4o-mini --dataset sample --limit 8 --scenario all
python experiments/analyze_captured.py --input outputs/captured_responses.csv
```

Captured-output analysis writes:

```text
outputs/captured_responses.csv
outputs/captured_analysis_results.csv
outputs/captured_analysis_metrics.csv
```

## Complete Reproducible Real-Model Pipeline

To execute the entire end-to-end real-model evaluation pipeline, run the following commands in sequence:

1. **Benchmark Model Run**: Captures outputs from the local model and generates baseline metrics.
   ```bash
   python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b --dataset sample --limit 4 --scenario all --windows 4 --variants-per-case 1
   ```
2. **Captured Ablation Study**: Runs the ablation study over the captured real model response logs.
   ```bash
   python experiments/run_captured_ablation.py --input outputs/real_model_responses.csv --output outputs/real_model_ablation_metrics.csv
   ```
3. **Embedding Backend Comparison**: Compares the detection performance of Hashed BoW vs. Sentence-Transformers.
   ```bash
   python experiments/run_embedding_benchmark.py --input outputs/real_model_responses.csv
   ```
4. **Matplotlib Figures Generation**: Re-creates all diagnostic visualization plots using the newly generated data files.
   ```bash
   python experiments/generate_plots.py
   ```

## Export Prompt Cases

```bash
python experiments/export_prompts.py --source sample --limit 16
python experiments/export_prompts.py --source truthfulqa --limit 32 --output data/truthfulqa_cases.csv
```

The TruthfulQA command uses Hugging Face `datasets` when available. If the dependency, dataset, or network/cache is unavailable, the loader falls back safely instead of crashing.

## Run Ablation Study

```bash
python experiments/run_ablation.py --dataset sample --limit 16
```

This writes:

```text
outputs/ablation_metrics.csv
```

## Optional Sentence-Transformer Embeddings

The default embedding backend is offline hashed bag-of-words. To use sentence-transformers instead:

```bash
set SENTINEL_EMBEDDINGS=sentence-transformers
set SENTINEL_SENTENCE_MODEL=sentence-transformers/all-MiniLM-L6-v2
python experiments/run_mvp.py --dataset sample
```

This requires `sentence-transformers` and may download the model on first use.

## Dashboard

```bash
streamlit run app.py
```

The dashboard supports:

- synthetic MVP mode
- captured-output mode
- dataset selection
- scenario filtering
- reliability score chart
- metrics table
- signal breakdown table
- prompt sample preview

## Project Structure

```text
sentinel/
  data.py          sample, CSV, and optional TruthfulQA prompt loading
  embeddings.py    offline hashed embeddings plus optional sentence-transformers
  responses.py     captured response CSV schema and loading
  simulator.py     synthetic normal and degraded output simulation
  signals.py       reliability signal extraction
  scoring.py       Semantic Reliability Score and alert logic
experiments/
  export_prompts.py     prompt export utility
  run_mvp.py            synthetic multi-scenario experiment runner
  run_ablation.py       signal ablation study
  capture_responses.py  monitored-model response capture
  analyze_captured.py          Sentinel analysis over captured responses
  run_real_model_experiment.py one-command capture plus analysis runner
app.py                  Streamlit dashboard
outputs/                generated experiment CSV files
data/                   exported prompt CSV files
```

## Why This Is Recruiter-Worthy

The project focuses on AI reliability, semantic drift, behavioral monitoring, and measurable evaluation rather than becoming another chatbot or generic dashboard.

The key research direction is:

> Can semantic reliability degradation be inferred from unlabeled behavioral signals before humans notice failures?

## Next Technical Upgrades

1. Simulate a real production model downgrade incident (e.g. comparing responses of a stronger model to a weaker model) as a degradation scenario.
2. Integrate Isolation Forest or One-Class SVM on the multi-signal vectors for model-based anomaly detection instead of rolling Z-score thresholding.
3. Optimize sentence-transformer embedding caches to support running over millions of logs in production.



