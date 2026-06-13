# EXPERIMENT.md
## Part 2: Experimental Design & Setup

---

## Research Question

> Given the same document corpus, does Hybrid RAG (BM25 + Vector + RRF) outperform  
> single-method retrieval (BM25-only or Vector-only) across different query types?

---

## Dataset

### Source
2026 FIFA World Cup — Wikipedia articles  
Collected via `wikipedia-api` Python library (June 2026)

### Pages Collected

| # | Article | Reason |
|---|---------|--------|
| 1 | 2026 FIFA World Cup | Core tournament info |
| 2 | 2026 FIFA World Cup squads | Player roster data |
| 3 | FIFA World Cup records and statistics | Historical facts & numbers |
| 4 | 2022 FIFA World Cup | Recent tournament reference |
| 5 | Lionel Messi | Star player profile |
| 6 | Kylian Mbappé | Star player profile |
| 7 | Cristiano Ronaldo | Star player profile |
| 8 | Erling Haaland | Top scorer in qualification |
| 9 | Lamine Yamal | Young star player |
| 10 | Vinicius Jr | Star player profile |
| 11 | Jude Bellingham | Star player profile |
| 12 | History of the FIFA World Cup | Background context |
| 13 | Association football | Sport definition & rules |
| 14 | FIFA | Governing body info |
| 15 | Goalkeeper (association football) | Positional context |

### Chunking Strategy

- Split articles by paragraph (double newline `\n\n`)
- Discard chunks shorter than 50 characters (headers, captions)
- Split chunks longer than 400 characters at sentence boundaries
- Target: ~80–120 chunks total

### Output format (`data/chunks.json`)

```json
[
  {
    "id": 0,
    "source": "2026 FIFA World Cup",
    "url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    "text": "The 2026 FIFA World Cup is the 23rd FIFA World Cup..."
  },
  ...
]
```

---

## Retrieval Systems

### System A — BM25 (Sparse)

Library: `rank_bm25` (BM25Okapi implementation)  
Preprocessing: lowercase, tokenize by whitespace, remove punctuation  
Parameters: default (k1=1.5, b=0.75)  
Output: ranked list of chunk IDs by BM25 score

### System B — Vector Search (Dense)

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`  
- Dimension: 384
- Language: English
- License: Apache 2.0

Index: FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors)  
Output: ranked list of chunk IDs by cosine similarity score

### System C — Hybrid (BM25 + Vector + RRF)

Fusion: Reciprocal Rank Fusion (RRF) with k=60  
Formula: `score(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_vector(d))`  
If document not in one ranking: assign rank = N+1 (worst possible)  
Output: unified re-ranked list of chunk IDs by RRF score

---

## Experiment Queries

Five queries designed to stress-test different retrieval capabilities:

### Query 1 — Exact Keyword (BM25 favored)
```
"How many goals did Messi score in the 2022 World Cup?"
```
Expected: BM25 retrieves exact statistics; Vector may return contextually similar but less precise chunks.

### Query 2 — Semantic / Descriptive (Vector favored)
```
"Which player is known for creative dribbling and exceptional vision?"
```
Expected: Vector retrieves Messi/Neymar profiles; BM25 fails if exact words don't appear.

### Query 3 — Proper Noun Lookup (BM25 favored)
```
"What country hosted the 2026 FIFA World Cup?"
```
Expected: BM25 finds exact mentions of "2026 FIFA World Cup" and "host"; Vector may over-generalize.

### Query 4 — Conceptual / Role-Based (Vector favored)
```
"Tell me about a goalkeeper famous for saving penalty kicks"
```
Expected: Vector matches semantics of "penalty saving"; BM25 requires the exact phrase.

### Query 5 — Mixed (Hybrid favored)
```
"Who is the youngest player with the most World Cup appearances?"
```
Expected: Requires both exact stat retrieval (BM25) and contextual understanding (Vector); Hybrid wins.

---

## Evaluation Method

### Primary: Manual Relevance Judgment

For each query, inspect Top-3 retrieved chunks and assign relevance:
- 2 = Directly answers the query
- 1 = Related but not directly answering
- 0 = Irrelevant

Compute **nDCG@3** (Normalized Discounted Cumulative Gain):
```
DCG@3   = rel_1 + rel_2/log2(3) + rel_3/log2(4)
IDCG@3  = best possible DCG@3
nDCG@3  = DCG@3 / IDCG@3
```

### Secondary: Retrieval Time

Measure wall-clock time (ms) for each retrieval system per query.

---

## Expected Hypotheses

H1: BM25 scores higher on Queries 1 and 3 (keyword-heavy)  
H2: Vector scores higher on Queries 2 and 4 (semantic)  
H3: Hybrid scores ≥ max(BM25, Vector) on all queries  
H4: Hybrid scores strictly higher on Query 5 (mixed)
