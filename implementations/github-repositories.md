# GitHub Implementations

This section contains open-source GitHub repositories related to LLM factuality, hallucination detection, scientific claim verification, retrieval-augmented generation (RAG), and LLM evaluation.

---

## 1. RAGTruth

- **Repository:** https://github.com/ParticleMedia/RAGTruth
- **Purpose:** Hallucination detection and evaluation in Retrieval-Augmented Generation (RAG) systems.
- **Research Area:** RAG, hallucination detection, factuality
- **Description:** RAGTruth provides a word-level hallucination corpus for training and evaluating systems that detect hallucinations in RAG-generated responses. The repository includes datasets, baseline implementations, and evaluation code.
- **Why Useful:** It is directly related to factuality and hallucination evaluation and provides both data and reproducible code.
- **License:** MIT
- **Research Paper:** https://arxiv.org/abs/2401.00396

---

## 2. OpenAI Evals

- **Repository:** https://github.com/openai/evals
- **Purpose:** Evaluation of large language models and LLM-based systems.
- **Research Area:** LLM evaluation, benchmarks, factuality
- **Description:** OpenAI Evals is an open-source framework for evaluating LLMs and systems built using LLMs. It includes existing evaluations and supports creating custom evaluations for specific use cases.
- **Why Useful:** Researchers can use the framework to create domain-specific evaluations for factuality, correctness, and reliability.
- **License:** MIT
- **Documentation:** https://github.com/openai/evals

---

## 3. RAGAS

- **Repository:** https://github.com/vibrantlabsai/ragas
- **Purpose:** Evaluation of Retrieval-Augmented Generation systems.
- **Research Area:** RAG, faithfulness, context relevance, answer quality
- **Description:** RAGAS provides evaluation metrics for RAG applications. Its faithfulness metric measures whether the generated response is factually consistent with the retrieved context.
- **Why Useful:** Faithfulness is directly connected to factuality because it evaluates whether generated claims are supported by retrieved evidence.
- **License:** Apache-2.0
- **Documentation:** https://docs.ragas.io/

---

## 4. DeepEval

- **Repository:** https://github.com/confident-ai/deepeval
- **Purpose:** Evaluation and testing of LLM applications.
- **Research Area:** Hallucination, factuality, RAG evaluation
- **Description:** DeepEval is an open-source framework for evaluating LLM applications. It provides metrics including hallucination, answer relevancy, faithfulness, and several RAG-related evaluation metrics.
- **Why Useful:** It can be used to test whether LLM responses are factually aligned with retrieved information and to compare different LLM application configurations.
- **License:** Apache-2.0
- **Documentation:** https://deepeval.com/

---

## 5. TruLens

- **Repository:** https://github.com/truera/trulens
- **Purpose:** Evaluation and observability of LLM applications and AI agents.
- **Research Area:** LLM evaluation, RAG, groundedness
- **Description:** TruLens provides tools for instrumenting LLM applications, evaluating individual steps, and comparing different versions of applications. It includes evaluation concepts such as groundedness and the RAG Triad.
- **Why Useful:** It can help identify whether an LLM application's output is grounded in the information used by the system.
- **License:** MIT
- **Documentation:** https://www.trulens.org/

---

## 6. HCMBench

- **Repository:** https://github.com/vectara/HCMBench
- **Purpose:** Benchmarking hallucination correction methods.
- **Research Area:** Hallucination mitigation, factuality evaluation
- **Description:** HCMBench provides tools and datasets for evaluating how effectively models can correct factual inaccuracies in generated text. The repository works with datasets such as RAGTruth and other factuality resources.
- **Why Useful:** It is useful for studying both hallucination detection and correction.
- **License:** See repository license and individual dataset licenses.
- **Documentation:** https://github.com/vectara/HCMBench

---

# Comparison of GitHub Implementations

| Repository | Main Focus | Factuality Relevance | Code/Data |
|---|---|---|---|
| RAGTruth | RAG hallucination detection | Very High | Code + Dataset |
| OpenAI Evals | LLM evaluation | High | Evaluation Framework |
| RAGAS | RAG evaluation | Very High | Evaluation Framework |
| DeepEval | LLM evaluation | Very High | Evaluation Framework |
| TruLens | LLM/RAG evaluation | High | Evaluation Framework |
| HCMBench | Hallucination correction | Very High | Code + Benchmarks |

---

# Selection Criteria

The repositories were selected using the following criteria:

1. **Research relevance** — The repository should be related to LLM factuality, hallucination, RAG, or evaluation.

2. **Documentation** — The project should provide useful documentation explaining installation and usage.

3. **Reproducibility** — The repository should provide code, datasets, benchmarks, or instructions that allow researchers to reproduce or extend experiments.

4. **Source-code availability** — The implementation should be publicly accessible.

5. **Maintenance** — Repository activity and available updates were considered when selecting projects.

6. **License** — The repository's license and the licenses of included datasets should be checked before redistribution.

7. **Practical usefulness** — The project should be useful for researchers studying factuality or reliability of LLM systems.
