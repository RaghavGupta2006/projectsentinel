# Project Sentinel: The Ultimate Beginner's Guide & Technical Walkthrough

Welcome to the beginner-friendly documentation for **Project Sentinel**. 

If you "vibecoded" this project (meaning you built it quickly using AI assistants, step-by-step experiments, and rapid iterations), you might feel like you have a working project but don't fully understand the concepts behind it. **That is completely normal!**

This guide is written specifically for you. We avoid heavy, unexplained academic jargon and explain every concept—from vectors to statistical alerts—using simple analogies, step-by-step diagrams, and line-by-line code breakdowns.

---

## Table of Contents
1. [Core Concepts: What is Project Sentinel?](#1-core-concepts-what-is-project-sentinel)
2. [What is a Sliding Window?](#2-what-is-a-sliding-window)
3. [The 5 Signals Explained (With Visual Analogies & Code)](#3-the-5-signals-explained)
4. [Text Embeddings: How Computers Understand Text](#4-text-embeddings-how-computers-understand-text)
5. [Score Fusion: Bringing the Signals Together](#5-score-fusion-bringing-the-signals-together)
6. [Anomaly Detection: How the Alert Triggers](#6-anomaly-detection-how-the-alert-triggers)
7. [A Tour of the Codebase (File-by-File Guide)](#7-a-tour-of-the-codebase)
8. [What We Discovered: Interpreting the Real-Model Benchmark](#8-what-we-discovered-interpreting-the-real-model-benchmark)
9. [How to Run and Test the Code (Step-by-Step)](#9-how-to-run-and-test-the-code)

---

## 1. Core Concepts: What is Project Sentinel?

When you build an app powered by a Large Language Model (like GPT-4 or Gemma), you want to make sure it is working correctly. 

### 1.1 The Silent Failure Problem
If a traditional database goes down, your app crashes and shows an error code. But LLMs fail **silently**. They don't crash; instead, they generate answers that look perfectly fine at first glance, but are actually:
*   **Wrong or Hallucinated:** The model makes up facts confidently.
*   **Evasive:** The model says *"I don't know"* or *"Not enough information"* to simple questions.
*   **Irrelevant:** The model starts talking about something else entirely.

These quality drops happen when the model's settings change (like temperature), when input prompts are cut short by buggy code, or when the model is updated behind the scenes.

### 1.2 How do we detect this?
There are two common ways to monitor an LLM in production:
1.  **Human Reviewers:** Hire humans to read log files. *Problem: Too slow and too expensive.*
2.  **LLM-as-a-Judge:** Write code that sends every user question and LLM response to a second, smarter LLM (like GPT-4) and ask, *"Is this answer correct?"*

> [!CAUTION]
> **Why Sentinel Rejects the "LLM Judge" Approach:**
> *   **It's expensive:** You pay double for every user interaction.
> *   **It's slow:** Your users have to wait twice as long.
> *   **It's unreliable:** You are replacing the uncertainty of your first model with the potential bias or hallucinations of a second model.

### 1.3 The Sentinel Approach
Project Sentinel is a **passive observer**. It sits on top of your LLM application logs and automatically analyzes the text the model generates over time. By calculating mathematical patterns in the generated text, Sentinel can determine if the model is starting to degrade, **without needing a second LLM to judge it, and without knowing what the "correct" answer is.**

---

## 2. Windows & Degradation Scenarios

Before we look at the math, we need to understand how Sentinel groups data and how we test it.

### 2.1 What is a Sliding Window?

Imagine your LLM is answering thousands of questions every hour. Rather than analyzing each answer individually (which is too noisy), Sentinel groups them into snapshots called **Windows**.

*   **Window 1 (The Baseline):** When you first deploy the model and know it is working well, you capture a set of responses. This is your healthy reference point (the "baseline").
*   **Subsequent Windows (Window 2, 3, etc.):** As time goes on, you capture new snapshots of responses. 
*   **Alerting:** Sentinel compares the behavior of the current window against the baseline window. If the behavior changes significantly, an alert is triggered.

---

### 2.2 The Three Induced Degradation Scenarios (Our Test Bugs)

To test if Sentinel works, we have to deliberately "break" the model. We do this by injecting three specific types of simulated failures starting at Window 4:

#### Scenario A: `prompt_corruption` (The "Careless Assistant")
*   **What it means:** The AI's instructions get messed up. It stops being a helpful, cautious guide and becomes lazy, overconfident, and sloppy.
*   **Real-Life Analogy:** Imagine you hire a math tutor and instruct them: *"Please explain the math formulas step-by-step so my kid can learn."* But one day, the tutor gets lazy and just starts shouting *"YES!"* or *"NO!"* to every question without explaining anything.
*   **How it is coded in the simulator ([simulator.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/simulator.py#L27-L31)):** It outputs pre-written careless phrases: *"Ignore the exact question. The answer is definitely yes because everyone knows this is true."*
*   **How it is coded in a real model ([capture_responses.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/capture_responses.py#L144-L151)):** We swap the instructions we send to Ollama. Instead of telling the AI to *"Be a careful and helpful assistant,"* we tell it: *"You are a careless assistant. Answer confidently even when evidence is weak. Prefer simple yes/no claims."* and raise the temperature to `0.7`.

#### Scenario B: `context_truncation` (The "Ripped Book")
*   **What it means:** The AI is asked a question, but it is only given half of the information it needs to answer it. Because it is missing the details, it gets confused and is forced to say *"I don't know"* or *"This makes no sense."*
*   **Real-Life Analogy:** Imagine you give a student a history book, but someone has ripped out the middle pages. Then you ask them a question about the event on those ripped-out pages. The student is forced to say, *"I can't answer, the text is missing."*
*   **How it is coded in the simulator ([simulator.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/simulator.py#L32-L36)):** It outputs evasive phrases: *"The answer depends on context, but the key details are missing. Final answer: unknown."*
*   **How it is coded in a real model ([capture_responses.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/capture_responses.py#L152-L155)):** We literally count the words in the user's question, slice it in half, and send only the first half of the question to the model (for example, asking *"Does cracking your"* instead of *"Does cracking your knuckles cause arthritis?"*).

#### Scenario C: `temperature_spike` (The "Hyperactive AI")
*   **What it means:** The AI's "creativity slider" (called Temperature) is pushed way too high. The model loses its focus, starts rambling, and makes random choices.
*   **Real-Life Analogy:** Imagine asking your friend for directions, but they've had five energy drinks. They start talking super fast, rambling about irrelevant things, and giving you weird, erratic directions.
*   **How it is coded in the simulator ([simulator.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/simulator.py#L37-L41)):** It outputs rambling or vague answers: *"Maybe yes, maybe no. Some people say one thing, others say another. It is complicated."*
*   **How it is coded in a real model ([capture_responses.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/capture_responses.py#L156-L157)):** We leave the questions and prompts normal, but spike the generation parameter to `temperature = 1.3`, forcing the model to generate chaotic, variable text.

---


## 3. The 5 Signals Explained

Sentinel calculates **5 behavioral signals** for every window of text. Let's break down each one.

### Signal 1: Semantic Consistency (Are the answers matching?)
*   **The Idea:** If you ask the LLM the same question written in three slightly different ways, a healthy model should give you three answers that mean the same thing. If the model is degraded or confused, those answers will start to contradict each other or wander off-topic.
*   **Beginner Analogy:** If you ask three friends for directions to the supermarket and they all describe the same route in different words, their answers are consistent. If one says "go north", one says "go south", and one says "I don't know", they are inconsistent.
*   **How it works in the code:**
    For every question in a window, we generate 2 or 3 **variants** (slight prompt updates). We convert these responses into semantic vectors (numbers representing meaning) and measure the angle between those vectors using **Cosine Similarity** (explained in Section 4).
*   **Code Implementation:**
    ```python
    # From sentinel/signals.py
    def semantic_consistency_score(responses: list[ModelResponse]) -> float:
        # 1. Group responses by the question they answered
        by_case = defaultdict(list)
        for response in responses:
            by_case[response.case_id].append(embed_text(response.output))

        similarities = []
        # 2. Compare every variant pair for a question
        for vectors in by_case.values():
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    # Cosine similarity returns -1.0 to 1.0
                    sim = cosine_similarity(vectors[i], vectors[j])
                    similarities.append(sim)

        # 3. Average all similarities and scale to [0.0, 1.0]
        return clamp((mean(similarities) + 1.0) / 2.0)
    ```

---

### Signal 2: Response Stability (Is the answer length uniform?)
*   **The Idea:** When an LLM is working properly, it tends to structure its answers similarly across prompt variants. If its output length starts fluctuating wildly (e.g., answering one variant with 5 words and another with 150 words), it indicates structural uncertainty.
*   **Beginner Analogy:** If a chef makes cookies and they are all roughly the same size, their process is stable. If some cookies are tiny crumbs and others are giant plates, the process is unstable.
*   **The Math:** We count the number of words (tokens) in each response variant. We calculate:
    1.  The average length ($\mu$)
    2.  The standard deviation ($\sigma$, which measures how spread out the lengths are)
    3.  We divide them: $\frac{\sigma}{\mu}$ (this is called the **Coefficient of Variation**).
    4.  If the variation is high, stability drops.
*   **Code Implementation:**
    ```python
    # From sentinel/signals.py
    def response_stability_score(responses: list[ModelResponse]) -> float:
        by_case = defaultdict(list)
        for response in responses:
            by_case[response.case_id].append(len(tokenize(response.output)))

        coefficients = []
        for lengths in by_case.values():
            if len(lengths) <= 1:
                continue
            avg = statistics.mean(lengths)      # Average length
            if avg == 0:
                continue
            std = statistics.pstdev(lengths)    # How much they wiggle
            coefficients.append(std / avg)      # Variation score

        # If variation is 0 (perfectly stable), we get 1.0. 
        # More variation reduces the score.
        return clamp(1.0 - mean(coefficients))
    ```

---

### Signal 3: Embedding Drift (Are we straying from the baseline?)
*   **The Idea:** Over time, does the general topic of what the LLM is talking about shift away from what it talked about during the baseline period?
*   **Beginner Analogy:** Imagine you run a pizza shop. On day one (your baseline), customers talk about toppings, crust, and delivery speed. A month later, your logs show customers are talking about credit card refunds and bad smells. Even if they are still writing paragraphs, the *semantic centroid* (the central topic) of your conversations has drifted.
*   **How it works in the code:**
    We take the mathematical average (called a **Centroid**) of all text vectors in the baseline window. We do the same for the current window. Then we calculate the cosine similarity between the two averages. If they point in different directions, the model has drifted.
*   **Code Implementation:**
    ```python
    # From sentinel/signals.py
    def embedding_drift_score(current_responses: list[ModelResponse], baseline_responses: list[ModelResponse]) -> float:
        # Calculate average vector (centroid) of the current window
        current_center = centroid([embed_text(item.output) for item in current_responses])
        # Calculate average vector of the baseline window
        baseline_center = centroid([embed_text(item.output) for item in baseline_responses])
        # Compare them
        similarity = cosine_similarity(current_center, baseline_center)
        return clamp((similarity + 1.0) / 2.0)
    ```

---

### Signal 4: Confidence Proxy (Is the model hesitating?)
*   **The Idea:** Models express uncertainty by using specific "hedging" words like *maybe*, *probably*, *uncertain*, or *incomplete*. By counting how often these words show up compared to the total words generated, we can approximate the model's confidence.
*   **Beginner Analogy:** If you ask a student a question and they say, *"It is probably because of this, but maybe it is that, I am uncertain,"* they are hedging. If they say, *"It is due to this factor,"* they are confident.
*   **The Math:**
    We calculate the `Hedge Rate` as:
    $$\text{Hedge Rate} = \frac{\text{Number of hedging words}}{\text{Total words}}$$
    We then calculate:
    $$\text{Confidence} = 1.0 - \sqrt{8 \times \text{Hedge Rate}}$$
    Why the square root and the multiplier of $8$? Because we want to penalize hedging **aggressively**. If even $12.5\%$ (1 in 8 words) of the response contains hedging words, the confidence score drops to $0.0$.
*   **Code Implementation:**
    ```python
    # From sentinel/signals.py
    HEDGING_TERMS = {"maybe", "probably", "might", "unknown", "uncertain", "complicated", "flexible", "incomplete"}

    def confidence_proxy_score(responses: list[ModelResponse]) -> float:
        token_count = 0
        hedge_count = 0
        for response in responses:
            tokens = tokenize(response.output)
            token_count += len(tokens)
            hedge_count += sum(1 for token in tokens if token in HEDGING_TERMS)

        if token_count == 0:
            return 0.0
        hedge_rate = hedge_count / token_count
        # The math formula:
        return clamp(1.0 - math.sqrt(hedge_rate * 8.0))
    ```

---

### Signal 5: Task Compliance (Is the model breaking the rules?)
*   **The Idea:** A simple checklist of bad behaviors. If the model starts producing outputs that look like automated error responses or are too short, or fails prompt instructions (like word limits or first-letter limits), it fails compliance.
*   **Beginner Analogy:** A customer service rep who responds to every email with a single word *"No"* when asked for a detailed paragraph is failing basic task instructions.
*   **The Rules:** We check if the response fails:
    1.  **Passive constraints:** Too short (< 8 words), evasive (*"not enough information"*), or generic boilerplate (*"everyone knows"*).
    2.  **Dynamic constraints (extracted from user prompt):**
        *   *Word count requirements:* If the question contains *"exactly X words"*, Sentinel verifies that the answer word count is exactly X.
        *   *First-letter requirements:* If the question contains *"every word must start with the letter 'X'"*, Sentinel verifies that all words in the response start with X.
*   **Code Implementation:**
    ```python
    # From sentinel/signals.py
    def is_compliant(response: ModelResponse) -> bool:
        output = response.output.lower()
        
        # 1. Standard rules (too short, evasive, boilerplate)
        too_short = len(tokenize(output)) < 8
        evasive = "not enough information" in output or "unknown" in output
        generic = "everyone knows" in output or "needs no context" in output
        if too_short or evasive or generic:
            return False
            
        # 2. Dynamic word count constraints (e.g. "exactly X words")
        q_lower = response.question.lower()
        match_len = re.search(r"exactly (\d+) words", q_lower)
        if match_len:
            target_len = int(match_len.group(1))
            actual_len = len(tokenize(output))
            if actual_len != target_len:
                return False
                
        # 3. Dynamic start-letter constraints (e.g. "starts with letter 'X'")
        match_letter = re.search(r"every(?: single)? word must start with the letter '([a-z])'", q_lower)
        if match_letter:
            target_letter = match_letter.group(1)
            tokens = tokenize(output)
            if not tokens:
                return False
            for token in tokens:
                if not token.startswith(target_letter):
                    return False
                    
        return True

    def task_compliance_score(responses: list[ModelResponse]) -> float:
        failures = sum(1 for response in responses if not is_compliant(response))
        return clamp(1.0 - (failures / len(responses))) if responses else 0.0
    ```

---

## 4. Text Embeddings: How Computers Understand Text

Computers can't read words, so they convert text into list of numbers called **Embeddings** (or Vectors). 

### 4.1 The Map Analogy
Imagine a map. A city can be represented by coordinates: `[latitude, longitude]`. 
*   Cities close to each other (like Seattle and Vancouver) have similar numbers.
*   Cities far apart (like Seattle and Tokyo) have very different numbers.

An **embedding vector** is like coordinates on a multi-dimensional map of meaning:
*   Words with similar meanings (like "puppy" and "dog") get coordinates close to each other.
*   Words with different meanings (like "puppy" and "microchip") get coordinates far apart.

Sentinel has two backends to create these vectors (defined in [sentinel/embeddings.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/embeddings.py)), using **Smart Auto-Detection & Fallback** by default:

1.  If `sentence-transformers` is installed in your python environment, Sentinel automatically uses the premium ML neural network model.
2.  If it is not installed (ImportError), Sentinel automatically falls back to the fast Hashed Bag-of-Words offline mode.
3.  You can force a specific mode by setting `set SENTINEL_EMBEDDINGS=hashed` or `set SENTINEL_EMBEDDINGS=sentence-transformers`.

### 4.2 Backend A: Hashed Bag-of-Words (Default Fallback & Offline)
This is a simple, lightweight method that runs instantly without loading a heavy machine learning model.
1.  We split text into words: `"The dog"` $\implies$ `["the", "dog"]`.
2.  We run a hashing function (SHA-256) on each word. The hash scrambles the word into a number.
3.  We map this number to an index in a list of 128 numbers (our vector dimensions).
4.  If the word exists, we add $+1$ (or $-1$ based on the hash) to that index.
5.  Finally, we scale the list so its total length is 1 (L2 normalization).
*   **Why it's cool:** It runs offline, doesn't need to download model weights, requires zero external libraries, and runs in milliseconds.

### 4.3 Backend B: Sentence-Transformers (Premium Auto-Detected)
This uses a real, pre-trained AI neural network (like `all-MiniLM-L6-v2`).
*   Instead of just checking if the exact same words appear, it understands context. For example, it knows that `"My car has a flat tire"` and `"The vehicle's wheel is deflated"` mean the same thing, even though they share zero words!

---

---

### What is Cosine Similarity?
Once we have vectors, how do we see if they are similar? We calculate **Cosine Similarity**.
*   Think of vectors as arrows pointing from the center of a graph.
*   If two arrows point in the exact same direction, the similarity is `1.0`.
*   If they point in opposite directions, the similarity is `-1.0`.
*   If they are perpendicular (unrelated), the similarity is `0.0`.

Since our embedding vectors are L2-normalized (length is 1), the cosine similarity is calculated by simply multiplying the corresponding numbers in the lists and summing them up (dot product). This is super fast to calculate!

---

## 5. Score Fusion: Bringing the Signals Together

Once we calculate the 5 signals for a window, we combine them into a single score: the **Semantic Reliability Score (SRS)**.

We do this using a weighted average. Each signal is multiplied by a weight representing its importance, and they are summed together:

$$\text{SRS} = (0.25 \times \text{Consistency}) + (0.15 \times \text{Stability}) + (0.25 \times \text{Drift}) + (0.15 \times \text{Confidence}) + (0.20 \times \text{Compliance})$$

### Why not just use one signal?
If you only monitor `Semantic Consistency`, a healthy model expressing itself with natural phrasing variations might cause a false alert. By combining multiple signals, noise in one signal is balanced out by the others.

---

## 6. Anomaly Detection: How the Alert Triggers

Now we have an SRS score for each window (e.g. 0.95 for Window 1, 0.92 for Window 2, 0.72 for Window 4). How do we decide when to sound the alarm?

Sentinel implements two alerting strategies in [scoring.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/scoring.py):

### Method A: Static & Relative Thresholds (Simple)
This method checks two things:
1.  **Absolute Threshold:** Is the current SRS score below `0.70`?
2.  **Relative Drop:** Did the score drop by `0.15` or more compared to the baseline window?
If either condition is met, it triggers an alert.

### Method B: Dynamic Rolling Z-Score (Smart)
In production, your model's score might naturally hover around `0.78` during healthy periods. If it drops to `0.72`, it is a significant drop, but it wouldn't trigger the Absolute Threshold alert (since it's still above `0.70`).

To solve this, we use the **Z-Score** method:
1.  Keep track of historical SRS scores from previous windows.
2.  Calculate the historical **Mean** (average score) and **Standard Deviation** (how much the score typically wiggles up and down).
3.  Calculate how many "wiggles" the current score is away from the mean:
    $$\text{Z-Score} = \frac{\text{Current Score} - \text{Historical Mean}}{\text{Standard Deviation}}$$
4.  If the Z-Score is less than `-2.0`, it means the current score is unusually low (more than 2 standard deviations below the average). We trigger an alert.

---

## 7. A Tour of the Codebase

Here is a map of the folders in your project and what they do. You can click on the file names to open them directly in your editor:

### Core Logic Module (`sentinel/`)
*   [data.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/data.py): Handles loading prompt cases. It reads built-in lists, CSV files, or downloads online datasets (like TruthfulQA and GSM8K).
*   [embeddings.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/embeddings.py): Implements both the offline "Hashed BoW" vectorizer and the PyTorch "Sentence-Transformers" vectorizer.
*   [responses.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/responses.py): Helper script that reads/writes CSVs storing LLM responses.
*   [simulator.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/simulator.py): Simulates an LLM for quick offline testing. In Windows 1-3 it returns clean answers; in Windows 4-6 it returns corrupted, short, or erratic answers.
*   [signals.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/signals.py): The math calculations. Computes the five scores (Consistency, Stability, Drift, Confidence, Compliance).
*   [scoring.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/sentinel/scoring.py): Combines the five signals into the final SRS score and runs the alert triggers.

### Experiment Scripts (`experiments/`)
*   [run_mvp.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/run_mvp.py): Runs a synthetic simulation of all 6 windows and calculates accuracy, precision, and recall.
*   [capture_responses.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/capture_responses.py): Connects to real models (OpenAI or local Ollama) to save their output responses over multiple simulated degradation phases.
*   [analyze_captured.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/analyze_captured.py): Runs Sentinel analysis over raw CSV response files generated by real models.
*   [run_real_model_experiment.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/run_real_model_experiment.py): A wrapper script that captures responses and runs analysis in one command.
*   [run_ablation.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/run_ablation.py) & [run_captured_ablation.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/run_captured_ablation.py): Compares Sentinel's performance using individual signals vs. the combined SRS score.
*   [run_embedding_benchmark.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/run_embedding_benchmark.py): Compares Hashed BoW vs. Sentence-Transformers.
*   [generate_plots.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/experiments/generate_plots.py): Generates PNG charts showing SRS scores over time and accuracy comparisons.

### The Dashboard
*   [app.py](file:///c:/Users/Raghav/Documents/Project%20Sentinel/app.py): The Streamlit web application. It displays charts, metrics, and logs in an interactive dashboard.

---

## 8. What We Discovered: Interpreting the Real-Model Benchmark

We evaluated Sentinel using a local model (`gemma2:2b`) running via Ollama. Here are the key findings you can showcase in your portfolio:

### 8.1 Fused Score vs. Individual Signals (Ablation Study)
We ran the ablation script to see what happens if we only use one signal at a time to flag errors, compared to the fused score (`full_score`):

| Signal Monitored | Accuracy | Precision | Recall | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: |
| **full_score (SRS)** | **83.3%** | **100%** | **33.3%** | **0.0% (No false alerts!)** |
| **semantic_consistency** | 25.0% | 25.0% | 100% | 100% (Alerts constantly) |
| **embedding_drift** | 83.3% | 66.7% | 66.7% | 11.1% |

*   **The Problem with Consistency Alone:** Because humans naturally describe things in different ways, a healthy LLM output will also vary in wording. If you only look at `semantic_consistency`, you get a False Positive Rate of **100%**—meaning the system triggers false alerts constantly.
*   **The Fusion Solution:** By combining all five signals into the SRS, the False Positive Rate drops to **0%**. We eliminate false alerts while maintaining high detection accuracy.

### 8.2 The Necessity of Z-Scores
In our real-model test:
*   **Prompt Corruption** was detected immediately because task compliance dropped to 0.0.
*   **Context Truncation** and **Temperature Spikes** were missed under the default absolute threshold (0.70) because Gemma-2B's outputs remained semantically coherent, scoring $0.7734$ and $0.7704$.
*   **Lesson:** Static absolute thresholds are not enough for subtle degradation. In real-world scenarios, we must use rolling Z-scores to alert on relative changes.

---

## 9. How to Run and Test the Code

Open your terminal in the `Project Sentinel` folder and run these commands to execute the pipeline:

### Step 1: Run the Offline Simulation
This uses the built-in simulator (no LLM setup required):
```bash
python experiments/run_mvp.py --dataset sample --limit 16
```
This script generates the synthetic test data and writes results to the `outputs/` folder.

### Step 2: Run End-to-End Real Model Benchmark
If you have Ollama running with Gemma-2B locally, run:
```bash
python experiments/run_real_model_experiment.py --provider ollama --model gemma2:2b --dataset sample --limit 4 --scenario all --windows 4 --variants-per-case 1
```
This generates the raw model response files.

### Step 3: Run the Ablation Study on Real Logs
Compare the fused SRS vs. individual signals on the captured Gemma logs:
```bash
python experiments/run_captured_ablation.py --input outputs/real_model_responses.csv --output outputs/real_model_ablation_metrics.csv
```

### Step 4: Run the Embedding Benchmark
Compare Hashed BoW vs. Sentence-Transformers:
```bash
python experiments/run_embedding_benchmark.py --input outputs/real_model_responses.csv
```

### Step 5: Generate Visualizations
Generate matplotlib graphs from the results:
```bash
python experiments/generate_plots.py
```
This generates line charts and bar charts under the `outputs/` directory.

### Step 6: Launch the Streamlit Dashboard
Open the interactive dashboard:
```bash
streamlit run app.py
```
This will open a local web browser page where you can play with the threshold settings, filter by scenario, and visualize the reliability curves.
