# SILVERK (Self-Supervised Cognitive Memory Core)

SILVERK is a cognitive memory architecture designed for Large Language Models (LLMs) to achieve **continual learning with zero context-window cost**. By directly modifying the parametric knowledge of an LLM through orthogonal adapter injection and dynamic Per-token Dense Cross-entropy (PDC) routing, SILVERK enables models to autonomously learn and recall facts without relying on fragile prompt engineering or slow Retrieval-Augmented Generation (RAG).

## 🚀 The Tri-Pillar Validation Paradigm

To rigorously prove the stability and integration depth of the SILVERK architecture, we evaluate it across three core pillars. Each script in this repository evaluates a specific dimension of parametric learning:

### 1. Zero Degradation (MMLU Benchmark)
**File: `benchmark_silverk_mmlu.py`**
Proof that the **Residual Lock** physically isolates injected knowledge adapters from the base model's manifold. This benchmark runs the Massive Multitask Language Understanding (MMLU) dataset to guarantee that new memory injections cause **0.00% degradation** to the LLM's pre-trained knowledge base.

### 2. Multi-Hop Reasoning Integration (MQuAKE Benchmark)
**File: `benchmark_silverk_mquake.py`**
Evaluates the model's ability to natively reason over injected knowledge using the MQuAKE (Multi-hop Question Answering Knowledge Editing) dataset. Because SILVERK integrates knowledge parametrically, the LLM can seamlessly "hop" between new facts in a single forward pass—bypassing the token-limit constraints of RAG.

### 3. Lateral Logical Consistency (RippleEdits Benchmark)
**File: `benchmark_silverk_ripple.py`**
Evaluates the "ripple effects" of memory injections. When a fact changes (e.g., *X is married to Y*), the model must consistently update downstream logical entailments (e.g., *Y is the spouse of X*). This script validates the robustness of the Manifold Warp Theory used to align adapter subspaces.

## 💾 Supplementary Data
- **`rag_results.txt`**: This contains raw outputs and baseline evaluations comparing traditional RAG retrieval latency and multi-hop failures against SILVERK's zero-context parametric retrieval.

## 🛠️ Usage

To run the benchmarking suite, you will need a Qwen-backed VLLM or AutoAWQ instance and standard scientific computing libraries (PyTorch, Transformers, Datasets).

Run the scripts individually based on the targeted evaluation matrix:

```bash
# Evaluate base knowledge retention (Catastrophic Forgetting check)
python benchmark_silverk_mmlu.py

# Evaluate multi-hop reasoning performance
python benchmark_silverk_mquake.py

# Evaluate lateral knowledge consistency and structural integrity
python benchmark_silverk_ripple.py
```

## 🧠 Core Mechanics
- **Orthogonal Initialization ($N_{max} = d_{model} / r$)**: Guarantees zero interference between adapters by allocating strictly orthogonal subspaces.
- **PDC Surprise Routing**: Uses localized entropy spikes (surprise) to automatically gate memory retrieval, meaning adapters only interfere when the base model lacks the required information.
- **Synthetic Multi-View Amplification**: Translates a single raw fact into a holographic batch of QA variants to protect against sequence overfitting.
