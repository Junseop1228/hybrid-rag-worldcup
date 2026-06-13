# RESULTS.md
## Experimental Results & Analysis

**Experiment date:** June 13, 2026
**Dataset:** 1,918 chunks from 15 Wikipedia articles (2026 FIFA World Cup related)
**Evaluation metric:** nDCG@3 (manual relevance judgments)
**Embedding model:** sentence-transformers/all-MiniLM-L6-v2 (dim=384)

---

## Summary Table

| Query | BM25 | Vector | Hybrid | Winner |
|-------|------|--------|--------|--------|
| Q1: Messi 2022 goals | 0.760 | 0.000 | 0.760 | BM25 = Hybrid |
| Q2: Creative dribbler | 0.532 | **0.699** | 0.532 | **Vector** |
| Q3: 2026 host country | 0.950 | 0.760 | **1.000** | **Hybrid** |
| Q4: Penalty-saving GK | 0.387 | **0.920** | 0.613 | **Vector** |
| Q5: Youngest + appearances | 0.000 | **0.950** | 0.380 | **Vector** |
| **Average** | 0.526 | **0.666** | 0.657 | **Vector** (marginal) |

---

## Retrieval Latency

| System | Avg Latency (ms) | Relative |
|--------|-----------------|----------|
| BM25 | 4.65 | 1x (baseline) |
| Vector | 17.73 | 3.8x |
| Hybrid | 32.08 | 6.9x |

---

## Per-Query Analysis

### Q1 — "How many goals did Messi score in the 2022 World Cup?"
**Expected winner:** BM25 (exact keyword query)
**Actual:** BM25 = Hybrid (0.760), Vector 0.000

Vector failed entirely — it retrieved a chunk about Messi's 2026 career goal instead of his 2022 World Cup stats, misled by semantic proximity to "Messi" + "goal" without temporal precision.
BM25 correctly ranked chunk 577 (Messi 2022 WC Argentina) via keyword matching.
**→ Exact stat queries with specific years/numbers favor BM25.**

---

### Q2 — "Which player is known for creative dribbling and exceptional vision?"
**Expected winner:** Vector (semantic query)
**Actual:** Vector 0.699 > BM25 0.532 > Hybrid 0.532

"Creative dribbling" does not appear verbatim in Wikipedia. Vector mapped the query to Vinicius Jr's description ("dynamic, intelligent, nimble winger") via semantic similarity.
Hybrid did not improve over BM25 — RRF diluted Vector's advantage by mixing in BM25's lower-quality results.
**→ Semantic queries without exact keyword matches strongly favor Vector.**

---

### Q3 — "What country hosted the 2026 FIFA World Cup?"
**Expected winner:** BM25 (keyword query)
**Actual:** Hybrid 1.000 > BM25 0.950 > Vector 0.760

Hybrid achieved perfect nDCG@3 = 1.000 — the only query where Hybrid outperformed both.
RRF placed both chunk 0 (2026 WC intro, directly naming USA/Canada/Mexico) and chunk 23 (bidding context) in the top-2 positions, combining BM25's exact match with Vector's contextual ranking.
**→ Mixed keyword + context queries are where Hybrid shines.**

---

### Q4 — "Tell me about a goalkeeper famous for saving penalty kicks"
**Expected winner:** Vector (semantic query)
**Actual:** Vector 0.920 > Hybrid 0.613 > BM25 0.387

BM25 returned chunk 1630 (general GK area description) as rank 1 — completely irrelevant.
Vector correctly identified chunk 1899 ("Goalkeepers are crucial in penalty shoot-outs...") via semantic match with "saving penalty kicks."
Hybrid improved over BM25 but still worse than Vector alone.
**→ Conceptual queries without exact phrases strongly favor Vector.**

---

### Q5 — "Who is the youngest player with the most World Cup appearances?"
**Expected winner:** Hybrid (complex multi-concept)
**Actual:** Vector 0.950 > Hybrid 0.380 > BM25 0.000

BM25 scored 0.000 — returned chunk 1906 (GK caps record, completely irrelevant) at rank 1.
Vector found chunk 236 ("Youngest player: 17 years, 41 days – Norman Whiteside...") directly.
Hybrid scored lower than Vector — BM25's poor ranking contaminated the RRF fusion.
**→ When BM25 fails badly, Hybrid inherits the damage. Vector dominates complex semantic queries.**

---

## Hypothesis Evaluation

| Hypothesis | Predicted | Actual | Confirmed? |
|-----------|-----------|--------|------------|
| H1: BM25 >= Vector on Q1, Q3 | BM25 wins keyword queries | Q1: tied; Q3: Hybrid wins | Partial |
| H2: Vector >= BM25 on Q2, Q4 | Vector wins semantic queries | Q2: 0.699>0.532 ✅; Q4: 0.920>0.387 ✅ | Confirmed |
| H3: Hybrid >= max(BM25,Vector) on all | Hybrid always best | Only Q3 confirmed | Not confirmed |
| H4: Hybrid strictly > both on Q5 | Hybrid wins mixed query | Vector wins Q5 (0.950>0.380) | Not confirmed |

---

## Key Findings

**Finding 1 — Vector dominates this dataset (avg 0.666)**
3 out of 5 queries are semantic in nature. When queries use descriptive language not present verbatim
in documents, Vector's embedding-based retrieval consistently outperforms BM25 by large margins.

**Finding 2 — Hybrid excels specifically at mixed keyword+context queries**
Q3 is the only query combining an exact named entity ("2026 FIFA World Cup") with contextual lookup
("hosted"). RRF fusion achieved perfect 1.000 — the only perfect score in the experiment.

**Finding 3 — BM25 is fastest and competitive on exact stat queries**
At 4.65ms (vs 32ms for Hybrid), BM25 is 6.9x faster. For exact keyword queries like Q1, it matches
Hybrid performance at a fraction of the cost.

**Finding 4 — When BM25 fails, Hybrid inherits the damage**
Q5 shows that RRF is only as good as its inputs. If BM25 retrieves irrelevant documents (score=0.000),
mixing them into Vector's results degrades overall performance.

**Finding 5 — No single index dominates all query types**
This validates the core argument from CONCEPT.md: different index types have different strengths.
Optimal retrieval requires understanding the query type and choosing (or combining) indexes accordingly.

---

## Connection to B+ Tree Indexing

| Era | Index | Search Key | Strength | Demonstrated |
|-----|-------|------------|----------|-------------|
| 1970s | B+ Tree | Ordered scalar | Exact + range | Q1 exact stat (year, goals) |
| 1990s | BM25 / Inverted Index | Term (word) | Keyword matching | Q1, Q3 BM25 |
| 2020s | Vector Index (FAISS) | Embedding (meaning) | Semantic similarity | Q2, Q4, Q5 |
| 2024+ | Hybrid RAG (BM25+Vector+RRF) | Both | All query types | Q3 perfect 1.000 |

> The B+ Tree requires a total order on search keys — it answers "goals > 5" but
> cannot answer "find a creative dribbler." Vector indexes replace the order relation
> with distance in semantic space. Hybrid RAG combines both — and this experiment
> shows that the optimal strategy depends on the query type, exactly as index
> selection in database design depends on the access pattern.
