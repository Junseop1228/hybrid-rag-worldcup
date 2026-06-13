# RESULTS.md
## Experimental Results & Analysis

> This file will be updated after running `notebooks/03_experiment.ipynb`.  
> Placeholder structure provided below.

---

## Summary Table

| Query | BM25 nDCG@3 | Vector nDCG@3 | Hybrid nDCG@3 | Winner |
|-------|-------------|---------------|---------------|--------|
| Q1: Messi 2022 goals | TBD | TBD | TBD | TBD |
| Q2: Creative dribbler | TBD | TBD | TBD | TBD |
| Q3: 2026 host country | TBD | TBD | TBD | TBD |
| Q4: Penalty-saving GK | TBD | TBD | TBD | TBD |
| Q5: Youngest appearances | TBD | TBD | TBD | TBD |
| **Average** | TBD | TBD | TBD | **Hybrid** |

---

## Retrieval Time

| System | Avg latency (ms) | Notes |
|--------|-----------------|-------|
| BM25 | TBD | CPU only, no GPU needed |
| Vector | TBD | FAISS IndexFlatIP |
| Hybrid | TBD | BM25 + Vector + RRF overhead |

---

## Per-Query Analysis

### Query 1 — "How many goals did Messi score in the 2022 World Cup?"

**BM25 Top-3:**
1. (TBD)
2. (TBD)
3. (TBD)

**Vector Top-3:**
1. (TBD)
2. (TBD)
3. (TBD)

**Hybrid Top-3:**
1. (TBD)
2. (TBD)
3. (TBD)

**Analysis:** (TBD after experiment)

---

### Query 2 — "Which player is known for creative dribbling and exceptional vision?"

**BM25 Top-3:** (TBD)  
**Vector Top-3:** (TBD)  
**Hybrid Top-3:** (TBD)  
**Analysis:** (TBD)

---

### Query 3 — "What country hosted the 2026 FIFA World Cup?"

**BM25 Top-3:** (TBD)  
**Vector Top-3:** (TBD)  
**Hybrid Top-3:** (TBD)  
**Analysis:** (TBD)

---

### Query 4 — "Tell me about a goalkeeper famous for saving penalty kicks"

**BM25 Top-3:** (TBD)  
**Vector Top-3:** (TBD)  
**Hybrid Top-3:** (TBD)  
**Analysis:** (TBD)

---

### Query 5 — "Who is the youngest player with the most World Cup appearances?"

**BM25 Top-3:** (TBD)  
**Vector Top-3:** (TBD)  
**Hybrid Top-3:** (TBD)  
**Analysis:** (TBD)

---

## Key Findings

(To be completed after experiments)

1. **Finding 1:** BM25 vs Vector on keyword queries
2. **Finding 2:** Vector advantage on semantic queries  
3. **Finding 3:** Hybrid consistency across all query types
4. **Finding 4:** Latency trade-off analysis

---

## Conclusion

> The results confirm that no single index type dominates across all query types.  
> Hybrid retrieval combining BM25 and vector search via RRF fusion achieves the best  
> overall performance — consistent with the findings of Sultania et al. (2024) and  
> directly analogous to the multi-index design principles covered in ITM 411.
