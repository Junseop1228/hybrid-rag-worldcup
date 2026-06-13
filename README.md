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
│   └── chunks.json         ← Preprocessed text chunks
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
git clone https://github.com/<your-username>/hybrid-rag-worldcup.git
cd hybrid-rag-worldcup
```

### 2. Open in Google Colab (recommended)

Run notebooks in order:

1. `notebooks/01_data_collection.ipynb` — builds `data/chunks.json`
2. `notebooks/02_retrieval_engines.ipynb` — builds all three retrievers
3. `notebooks/03_experiment.ipynb` — runs experiments and plots results

### 3. Local setup (optional)

See [SETUP.md](SETUP.md) for local environment instructions.

---

## Key Results (Summary)

See [RESULTS.md](RESULTS.md) for full analysis.

| Query Type | BM25 | Vector | Hybrid |
|------------|------|--------|--------|
| Exact keyword | ✅ Strong | ⚠️ Moderate | ✅ Strong |
| Semantic meaning | ❌ Weak | ✅ Strong | ✅ Strong |
| Mixed | ⚠️ Moderate | ⚠️ Moderate | ✅ Strongest |

---

## References

See [REFERENCES.md](REFERENCES.md) for full citation list.

Key papers:
- Lewis et al. (2020) — RAG original paper, NeurIPS
- Karpukhin et al. (2020) — Dense Passage Retrieval, EMNLP
- Sultania et al. (2024) — Hybrid BM25 + Dense retrieval benchmarks

---

## Course Information

- Course: ITM 411 Database Management (146058-21001)
- Professor: Hyuk-Yoon Kwon
- Presentation slot: 2nd (Week 15, deadline June 15, 2026)
- Topic scope: Lectures 10–13 (Indexing & Transactions)
