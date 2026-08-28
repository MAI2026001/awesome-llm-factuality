# Tools and Libraries

This section contains tools and libraries that are useful for evaluating LLM factuality, hallucination, retrieval-augmented generation (RAG), evidence grounding, and overall LLM application quality.

---

## 1. RAGAS

- **Purpose:** Evaluation framework for Retrieval-Augmented Generation (RAG) systems.
- **Main Use:** Measures aspects such as faithfulness, answer relevance, and context relevance.
- **Why Relevant:** RAGAS is directly useful for evaluating whether generated answers are supported by retrieved information.
- **Official Repository:** https://github.com/explodinggradients/ragas
- **Documentation:** https://docs.ragas.io/

---

## 2. DeepEval

- **Purpose:** Open-source evaluation framework for LLM applications.
- **Main Use:** Provides metrics for evaluating hallucination, factuality, relevance, answer quality, and RAG systems.
- **Why Relevant:** It can be used to test whether an LLM produces factually reliable responses.
- **Official Repository:** https://github.com/confident-ai/deepeval
- **Documentation:** https://deepeval.com/

---

## 3. TruLens

- **Purpose:** Evaluation and observability framework for LLM applications.
- **Main Use:** Evaluates LLM applications, RAG pipelines, and agentic systems using feedback functions and evaluation metrics.
- **Why Relevant:** It helps measure the quality and reliability of LLM-generated answers and RAG applications.
- **Official Repository:** https://github.com/truera/trulens
- **Documentation:** https://www.trulens.org/

---

## 4. LlamaIndex

- **Purpose:** Framework for building applications that connect LLMs with external data.
- **Main Use:** Supports document indexing, retrieval, RAG pipelines, and evaluation of retrieval-based applications.
- **Why Relevant:** Evidence retrieval and grounding are important for reducing unsupported factual claims in LLM responses.
- **Official Repository:** https://github.com/run-llama/llama_index
- **Documentation:** https://docs.llamaindex.ai/

---

## 5. LangChain

- **Purpose:** Framework for developing applications powered by large language models.
- **Main Use:** Provides components for retrieval, document processing, RAG, agents, evaluation, and application development.
- **Why Relevant:** LangChain can be used to build evidence-grounded LLM systems and evaluate their outputs.
- **Official Repository:** https://github.com/langchain-ai/langchain
- **Documentation:** https://python.langchain.com/

---

## 6. OpenAI Evals

- **Purpose:** Open-source framework for evaluating LLM and AI system performance.
- **Main Use:** Supports custom evaluations and benchmark creation for language model applications.
- **Why Relevant:** Researchers can create evaluations for factuality, correctness, and domain-specific LLM behavior.
- **Official Repository:** https://github.com/openai/evals
- **Documentation:** https://github.com/openai/evals

---

## Tool Comparison

| Tool | Main Purpose | Factuality / RAG Relevance |
|---|---|---|
| RAGAS | RAG evaluation | High |
| DeepEval | LLM evaluation | High |
| TruLens | LLM evaluation and observability | High |
| LlamaIndex | Data connection and RAG | High |
| LangChain | LLM application development | Medium–High |
| OpenAI Evals | Custom LLM evaluation | High |

---

## Selection Criteria

The tools in this collection were selected based on:

- Relevance to LLM evaluation or factuality
- Support for RAG or evidence-grounded systems
- Availability of public documentation
- Availability of source code or project documentation
- Usefulness for research and experimentation
