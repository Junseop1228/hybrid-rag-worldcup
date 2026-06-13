# From B+ Tree to Hybrid RAG
## DB Indexing Principles Behind Modern AI Search — with 2026 FIFA World Cup Data

> ITM 411 Database Management — Individual Project (2nd Presentation)  
> Seoul National University of Science and Technology  
> Prof. Hyuk-Yoon Kwon

---

## Overview

This project bridges a core concept from the ITM 411 lecture — **B+ Tree indexing** — with one of the most actively researched areas in modern AI: **Retrieval-Augmented Generation (RAG)**.

The central argument is simple:

> The essence of an index has not changed. What changed is the search key — from a value to a meaning.

We demonstrate this by building three retrieval systems over a 2026 FIFA World Cup dataset and comparing their performance:

| System | Mechanism | Strength |
|--------|-----------|----------|
| BM25 | Keyword frequency (sparse) | Exact term matching |
| Vector Search | Semantic embeddings (dense) | Meaning-based matching |
| **Hybrid RAG** | BM25 + Vector + RRF fusion | Both |

---

## Motivation

In lecture, we learned that a B+ Tree requires a **total order** on search keys — it can answer "find all players with goals > 5", but it cannot answer "find a player known for creative dribbling."

This limitation motivates the shift from keyword-based to semantic indexing, and ultimately to hybrid systems that combine both. This is exactly the direction that 2024–2025 RAG research has converged on.

---

## Project Structure

```
hybrid-rag-worldcup/
│
├── README.md               ← You are here
├── CONCEPT.md              ← Part 1: Theory (B+ Tree → Hybrid RAG)
├── EXPERIMENT.md           ← Part 2: Experimental design & results
├── RESULTS.md              ← Numerical results & analysis
├── REFERENCES.md           ← All cited papers and sources
├── SETUP.md                ← Environment setup & how to run
│
├── data/
│   ├── raw/                ← Wikipedia source documents
│   └── chunks.json         ← Preprocessed text chunks (1,918 chunks)
│
├── notebooks/
│   ├── 01_data_collection.ipynb    ← Scrape & chunk Wikipedia docs
│   ├── 02_retrieval_engines.ipynb  ← Build BM25, Vector, Hybrid
│   └── 03_experiment.ipynb         ← Run experiments & visualize
│
└── src/
    ├── chunker.py          ← Text chunking utilities
    ├── bm25_retriever.py   ← BM25 search engine
    ├── vector_retriever.py ← FAISS vector search engine
    ├── hybrid_retriever.py ← RRF fusion engine
    └── evaluate.py         ← Evaluation & visualization
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Junseop1228/hybrid-rag-worldcup.git
cd hybrid-rag-worldcup
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

```
notebooks/01_data_collection.ipynb   → builds data/chunks.json
notebooks/02_retrieval_engines.ipynb → sanity check all three engines
notebooks/03_experiment.ipynb        → full experiment + visualization
```

See [SETUP.md](SETUP.md) for detailed local environment instructions.

---

## Key Results

See [RESULTS.md](RESULTS.md) for full per-query analysis.

| System | nDCG@3 Avg | Latency |
|--------|-----------|---------|
| BM25 | 0.526 | 4.65ms |
| Vector | **0.666** | 17.73ms |
| Hybrid | 0.657 | 32.08ms |

| Query Type | BM25 | Vector | Hybrid |
|------------|------|--------|--------|
| Exact keyword (Q1, Q3) | ✅ Strong | ❌ Weak | ✅ Strong |
| Semantic meaning (Q2, Q4) | ❌ Weak | ✅ Strong | ⚠️ Moderate |
| Mixed keyword+context (Q3) | ⚠️ | ⚠️ | ✅ Perfect (1.000) |

---

## References

See [REFERENCES.md](REFERENCES.md) for full citation list.

Key papers:
- Lewis et al. (2020) — RAG original paper, NeurIPS 2020
- Karpukhin et al. (2020) — Dense Passage Retrieval, EMNLP 2020
- Sultania et al. (2024) — Hybrid BM25 + Dense retrieval benchmarks
- Cormack et al. (2009) — Reciprocal Rank Fusion, SIGIR 2009

---

## AI Assistance Disclosure

This project was developed with assistance from AI tools as follows:

| Task | Tool | Extent |
|------|------|--------|
| Project structure & MD documentation | Claude (Anthropic) | Drafted, reviewed and edited by student |
| Code scaffolding & implementation | Claude (Anthropic) + Antigravity | Specifications written by student, all code reviewed by student |
| Research & paper finding | Claude (Anthropic) web search | Student verified all cited papers independently |
| Conceptual explanation (CONCEPT.md) | Claude (Anthropic) | Written with student, based on lecture content |

**What the student did independently:**
- Chose the research topic and direction
- Defined all experiment queries and evaluation criteria
- Reviewed and approved all code before committing
- Made relevance judgments for nDCG evaluation (Cell 5 of 03_experiment.ipynb)
- Drew conclusions connecting results to lecture content
- Prepared and delivered the video presentation

All source code has been reviewed and understood by the student. No code was copied from other students or existing RAG repositories. AI-generated code was used as a scaffold and modified as needed.

---

## Course Information

- Course: ITM 411 Database Management (146058-21001)
- Professor: Hyuk-Yoon Kwon
- Presentation slot: 2nd (Week 15, deadline June 15, 2026)
- Topic scope: Lectures 10–13 (Indexing & Transactions)
