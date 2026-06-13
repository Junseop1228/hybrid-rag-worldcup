# Coding Task Instructions for hybrid-rag-worldcup

## Agent Instructions
- Read ALL markdown files in the repository root before starting any work
- Files to read first: `README.md`, `CONCEPT.md`, `EXPERIMENT.md`, `REFERENCES.md`, `SETUP.md`
- These documents define the project scope, theory, and expected behavior of every module
- Do NOT deviate from the specifications in these documents

---

## Repository
https://github.com/Junseop1228/hybrid-rag-worldcup

## Branch Strategy
- Base branch: `main`
- Create one branch per issue: `feat/issue-{N}-{short-name}`
- Each branch must be pushed and a PR opened against `main`
- PR title format: `feat: {description} - Closes #{N}`
- Do NOT merge PRs — human will review and merge

---

## Work Order
Issues MUST be completed in the following order due to dependencies:

```
#1 src/chunker.py
    ↓
#2 notebooks/01_data_collection.ipynb   ←─┐
#3 src/bm25_retriever.py                  │  (can start after #1 merges)
#4 src/vector_retriever.py              ←─┘
    ↓
#5 src/hybrid_retriever.py              (after #3 and #4 merge)
    ↓
#6 notebooks/02_retrieval_engines.ipynb (after #2 and #5 merge)
#7 src/evaluate.py                      (after #3, #4, #5 merge)
    ↓
#8 notebooks/03_experiment.ipynb        (after #6 and #7 merge)
```

---

## Global Code Standards
- Language: English only (comments, docstrings, variable names)
- Every function must have a docstring with: description, Args, Returns, Example
- Type hints required on all function signatures
- Error handling required (FileNotFoundError, ImportError, etc.)
- Each file must have a module-level docstring referencing relevant paper from REFERENCES.md

---

## Issue #1 — `src/chunker.py`
**Branch:** `feat/issue-1-chunker`

### Specification
Wikipedia article fetcher and text chunker utility module.

### Functions Required

```python
def fetch_wikipedia_page(title: str) -> Optional[dict]:
    """
    Fetch raw text and URL from Wikipedia.
    Returns dict with keys: 'title', 'url', 'text'
    Returns None if page does not exist.
    Uses: wikipediaapi library, language='en'
    User-Agent: 'hybrid-rag-worldcup/1.0 (ITM411 course project)'
    """

def chunk_text(
    text: str,
    source: str,
    url: str,
    max_chars: int = 400,
    min_chars: int = 50
) -> list[dict]:
    """
    Split article text into chunks.
    Strategy:
      1. Split by paragraph (double newline)
      2. Discard paragraphs shorter than min_chars
      3. If paragraph <= max_chars: keep as-is
      4. If paragraph > max_chars: split at sentence boundaries ('. ', '? ', '! ')
    Returns list of dicts: {'source', 'url', 'text'} — no ID yet
    """

def build_chunk_list(pages: list[dict]) -> list[dict]:
    """
    Flatten list of fetched pages into single chunk list with sequential IDs.
    Calls chunk_text() for each page.
    Returns list of dicts: {'id', 'source', 'url', 'text'}
    """

def save_chunks(chunks: list[dict], path: str) -> None:
    """
    Save chunks to JSON file.
    Creates parent directories if needed (os.makedirs).
    Prints: "Saved {N} chunks to {path}"
    """

def load_chunks(path: str) -> list[dict]:
    """
    Load chunks from JSON file.
    Raises FileNotFoundError with helpful message if path missing.
    Prints: "Loaded {N} chunks from {path}"
    """
```

### Module footer
```python
if __name__ == "__main__":
    # Self-test: fetch "Association football", chunk it, print stats
```

---

## Issue #2 — `notebooks/01_data_collection.ipynb`
**Branch:** `feat/issue-2-data-collection`
**Depends on:** Issue #1 merged

### Specification
Jupyter notebook that collects 15 Wikipedia pages and produces `data/chunks.json`.

### Notebook Cell Structure

**Cell 1 — Install dependencies**
```python
# install if not already present
!pip install wikipedia-api==0.6.0 tqdm
```

**Cell 2 — Imports**
```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from src.chunker import fetch_wikipedia_page, build_chunk_list, save_chunks
from tqdm import tqdm
```

**Cell 3 — Define target pages**
```python
TARGET_PAGES = [
    "2026 FIFA World Cup",
    "2026 FIFA World Cup squads",
    "FIFA World Cup records and statistics",
    "2022 FIFA World Cup",
    "Lionel Messi",
    "Kylian Mbappé",
    "Cristiano Ronaldo",
    "Erling Haaland",
    "Lamine Yamal",
    "Vinicius Jr.",
    "Jude Bellingham",
    "History of the FIFA World Cup",
    "Association football",
    "FIFA",
    "Goalkeeper (association football)"
]
```

**Cell 4 — Fetch all pages with tqdm progress bar**
```python
# fetch each page, print success/failure per page
# store results in `pages` list
```

**Cell 5 — Build chunk list and show stats**
```python
# call build_chunk_list(pages)
# print: total chunks, chunks per source (table)
# print sample chunk (id=0)
```

**Cell 6 — Save**
```python
save_chunks(chunks, "../data/chunks.json")
```

---

## Issue #3 — `src/bm25_retriever.py`
**Branch:** `feat/issue-3-bm25`
**Depends on:** Issue #1 merged

### Specification
BM25 sparse keyword retrieval engine.

Reference: Robertson & Zaragoza (2009) — REFERENCES.md [7]

### Class Required

```python
class BM25Retriever:
    """
    Sparse keyword retrieval using BM25Okapi algorithm.
    Reference: Robertson & Zaragoza (2009) — REFERENCES.md [7]
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.index = None  # BM25Okapi instance

    def _preprocess(self, text: str) -> list[str]:
        """
        Lowercase, remove punctuation, split by whitespace.
        Returns list of tokens.
        """

    def build_index(self, chunks: list[dict]) -> None:
        """
        Tokenize all chunk texts and build BM25Okapi index.
        Stores chunks internally for result lookup.
        Prints: "BM25 index built: {N} chunks"
        """

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Score all chunks against query using BM25.
        Returns top_k results sorted by score descending.
        Each result dict: {'rank', 'score', 'chunk_id', 'source', 'text'}
        rank is 1-indexed.
        """
```

### Module footer
```python
if __name__ == "__main__":
    # Self-test: load data/chunks.json, build index, run one query
    # Query: "How many goals did Messi score?"
    # Print top 3 results
```

---

## Issue #4 — `src/vector_retriever.py`
**Branch:** `feat/issue-4-vector`
**Depends on:** Issue #1 merged

### Specification
Dense vector retrieval engine using sentence-transformers + FAISS.

Reference: Karpukhin et al. (2020) DPR — REFERENCES.md [2]

### Class Required

```python
class VectorRetriever:
    """
    Dense semantic retrieval using sentence-transformers embeddings and FAISS index.
    Reference: Karpukhin et al. (2020) Dense Passage Retrieval — REFERENCES.md [2]
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # dimension: 384, license: Apache 2.0

    def __init__(self):
        self.chunks: list[dict] = []
        self.model = None       # SentenceTransformer instance
        self.index = None       # faiss.IndexFlatIP instance
        self.embeddings = None  # np.ndarray shape (N, 384)

    def _load_model(self) -> None:
        """
        Lazy-load sentence-transformer model on first use.
        Prints: "Loading embedding model: {MODEL_NAME}"
        """

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed list of texts using sentence-transformers.
        Normalize vectors (L2) so inner product = cosine similarity.
        Returns np.ndarray of shape (len(texts), 384), dtype float32.
        """

    def build_index(self, chunks: list[dict]) -> None:
        """
        Embed all chunk texts and build FAISS IndexFlatIP.
        Uses inner product on L2-normalized vectors = cosine similarity.
        Prints: "Vector index built: {N} chunks, dim=384"
        """

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed query and retrieve top_k nearest chunks.
        Returns top_k results sorted by cosine similarity descending.
        Each result dict: {'rank', 'score', 'chunk_id', 'source', 'text'}
        rank is 1-indexed.
        """
```

### Module footer
```python
if __name__ == "__main__":
    # Self-test: load data/chunks.json, build index, run one query
    # Query: "creative dribbling and exceptional vision"
    # Print top 3 results
```

---

## Issue #5 — `src/hybrid_retriever.py`
**Branch:** `feat/issue-5-hybrid`
**Depends on:** Issues #3 and #4 merged

### Specification
Hybrid retrieval engine combining BM25 + Vector via Reciprocal Rank Fusion (RRF).

Reference: Cormack et al. (2009) SIGIR — REFERENCES.md [4]
Reference: Sultania et al. (2024) — REFERENCES.md [3]

### Class Required

```python
class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and Vector (dense) search
    via Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1 / (k + rank_i(d)), k=60
    Reference: Cormack et al. (2009) SIGIR — REFERENCES.md [4]
    """

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        """
        Args:
            bm25:   Pre-built BM25Retriever instance
            vector: Pre-built VectorRetriever instance
        Both indexes must be built before passing to HybridRetriever.
        """

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: list[list[dict]],
        k: int = 60
    ) -> list[dict]:
        """
        Merge multiple ranked result lists using RRF.

        Args:
            rankings: List of ranked result lists (each from one retriever).
                      Each result must have 'chunk_id' field.
            k:        RRF smoothing constant. Default 60 per Cormack et al. (2009).

        Logic:
            - For each result list, assign rank 1..N
            - Documents not appearing in a list get rank = len(list) + 1
            - Final score = sum of 1/(k + rank) across all lists
            - Sort by final score descending

        Returns:
            List of dicts: {'chunk_id', 'rrf_score', 'bm25_rank', 'vector_rank'}
        """

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Run BM25 and Vector search, fuse results with RRF.
        Returns top_k results.
        Each result dict: {'rank', 'rrf_score', 'bm25_rank', 'vector_rank',
                           'chunk_id', 'source', 'text'}
        rank is 1-indexed.
        """
```

### Module footer
```python
if __name__ == "__main__":
    # Self-test: load chunks, build both indexes, run hybrid search
    # Query: "youngest player with most World Cup appearances"
    # Print top 3 results with bm25_rank and vector_rank shown
```

---

## Issue #6 — `notebooks/02_retrieval_engines.ipynb`
**Branch:** `feat/issue-6-engine-notebook`
**Depends on:** Issues #2 and #5 merged

### Specification
Sanity-check notebook verifying all three retrieval engines work correctly.

### Notebook Cell Structure

**Cell 1 — Install + imports**
```python
!pip install rank-bm25 sentence-transformers faiss-cpu
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from src.chunker import load_chunks
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.hybrid_retriever import HybridRetriever
```

**Cell 2 — Load data**
```python
chunks = load_chunks("../data/chunks.json")
print(f"Loaded {len(chunks)} chunks")
```

**Cell 3 — Build all indexes**
```python
bm25 = BM25Retriever()
bm25.build_index(chunks)

vector = VectorRetriever()
vector.build_index(chunks)

hybrid = HybridRetriever(bm25, vector)
```

**Cell 4 — Run sanity-check query and display side-by-side**
```python
TEST_QUERY = "Who is the best player in the 2026 FIFA World Cup?"
# Run all three retrievers with top_k=3
# Print results in a clear formatted table:
# Rank | BM25 result | Vector result | Hybrid result
# Each cell shows: source + first 120 chars of text
```

**Cell 5 — Verify output schema**
```python
# Assert all required keys present in each result dict
# Print: "All schema checks passed"
```

---

## Issue #7 — `src/evaluate.py`
**Branch:** `feat/issue-7-evaluate`
**Depends on:** Issues #3, #4, #5 merged

### Specification
Evaluation metrics and visualization utilities.

### Functions Required

```python
def compute_ndcg(
    retrieved: list[dict],
    relevant_ids: list[int],
    k: int = 3
) -> float:
    """
    Compute nDCG@k.

    Relevance scoring:
      - chunk_id in relevant_ids with score 2: directly answers query
      - chunk_id in relevant_ids with score 1: related but indirect
      - otherwise: 0

    Args:
        retrieved:    Ordered list of result dicts (must have 'chunk_id').
        relevant_ids: Dict mapping chunk_id -> relevance score {0,1,2}.
        k:            Cutoff rank. Default 3.

    Formula:
        DCG@k  = rel_1 + rel_2/log2(3) + rel_3/log2(4)
        IDCG@k = best possible DCG given relevant_ids scores
        nDCG@k = DCG@k / IDCG@k  (0.0 if IDCG=0)

    Returns: float in [0.0, 1.0]
    """

def measure_latency(
    retriever,
    queries: list[str],
    top_k: int = 5,
    runs: int = 10
) -> float:
    """
    Measure average retrieval latency in milliseconds.

    Runs each query `runs` times and averages wall-clock time.
    Uses time.perf_counter() for precision.

    Returns: average latency in ms (float)
    """

def plot_comparison(
    results: dict,
    save_path: str = None
) -> None:
    """
    Generate grouped bar chart comparing BM25, Vector, Hybrid per query.

    Args:
        results: Dict structure:
            {
              'Q1': {'bm25': 0.8, 'vector': 0.4, 'hybrid': 0.9},
              'Q2': {'bm25': 0.2, 'vector': 0.9, 'hybrid': 0.95},
              ...
            }
        save_path: If provided, save figure to this path (e.g. 'results.png')
                   If None, call plt.show()

    Chart specs:
        - x-axis: query labels (Q1–Q5)
        - y-axis: nDCG@3 score, range [0, 1]
        - bars: BM25=steelblue, Vector=darkorange, Hybrid=mediumseagreen
        - legend, grid (y-axis only), title: "Retrieval System Comparison (nDCG@3)"
        - figure size: (10, 6)
    """

def print_results_table(results: dict) -> None:
    """
    Print a formatted ASCII table of nDCG@3 results.

    Output format:
        Query  | BM25   | Vector | Hybrid
        -------|--------|--------|-------
        Q1     | 0.800  | 0.400  | 0.900
        ...
        Average| x.xxx  | x.xxx  | x.xxx
    """
```

---

## Issue #8 — `notebooks/03_experiment.ipynb`
**Branch:** `feat/issue-8-experiment`
**Depends on:** Issues #6 and #7 merged

### Specification
Full experiment notebook. Runs all 5 queries, computes nDCG@3 and latency, visualizes results.

### Notebook Cell Structure

**Cell 1 — Install + imports**
```python
!pip install rank-bm25 sentence-transformers faiss-cpu matplotlib pandas
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from src.chunker import load_chunks
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluate import compute_ndcg, measure_latency, plot_comparison, print_results_table
```

**Cell 2 — Load data and build indexes**
```python
chunks = load_chunks("../data/chunks.json")
bm25 = BM25Retriever(); bm25.build_index(chunks)
vector = VectorRetriever(); vector.build_index(chunks)
hybrid = HybridRetriever(bm25, vector)
```

**Cell 3 — Define experiment queries**
```python
QUERIES = {
    "Q1": "How many goals did Messi score in the 2022 World Cup?",
    "Q2": "Which player is known for creative dribbling and exceptional vision?",
    "Q3": "What country hosted the 2026 FIFA World Cup?",
    "Q4": "Tell me about a goalkeeper famous for saving penalty kicks",
    "Q5": "Who is the youngest player with the most World Cup appearances?"
}
```

**Cell 4 — Run all retrievers, print Top-3 per query**
```python
# For each query: run bm25.search(), vector.search(), hybrid.search() with top_k=3
# Print formatted table: Rank | BM25 | Vector | Hybrid
# Show source + first 120 chars of text per result
```

**Cell 5 — Manual relevance annotation**
```python
# Define RELEVANCE dict:
# RELEVANCE[query_label][chunk_id] = score (0, 1, or 2)
# Fill in after inspecting Cell 4 results
# This cell requires human judgment — leave template with TODOs

RELEVANCE = {
    "Q1": {},  # TODO: fill after running Cell 4
    "Q2": {},
    "Q3": {},
    "Q4": {},
    "Q5": {},
}
```

**Cell 6 — Compute nDCG@3 for all systems**
```python
# Compute nDCG@3 for each (query, system) combination
# Store in results dict matching plot_comparison() input format
# Print results table via print_results_table()
```

**Cell 7 — Latency measurement**
```python
# Measure avg latency for BM25, Vector, Hybrid across all 5 queries
# Print: System | Avg Latency (ms)
```

**Cell 8 — Visualization**
```python
plot_comparison(results, save_path="../data/comparison_chart.png")
```

**Cell 9 — Summary and conclusions**
```python
# Markdown cell (not code)
# Written conclusions:
# - Which system won on keyword queries (Q1, Q3)?
# - Which system won on semantic queries (Q2, Q4)?
# - Did Hybrid win overall? Compare to hypotheses in EXPERIMENT.md
# - Connection back to B+ Tree indexing principles from lecture
```

---

## Final Checklist Before Each PR

- [ ] All functions have docstrings with Args, Returns, Example
- [ ] All function signatures have type hints
- [ ] Error cases handled (FileNotFoundError, ImportError, empty results)
- [ ] Module-level docstring references relevant paper from REFERENCES.md
- [ ] Self-test passes (for .py files)
- [ ] Notebook runs top-to-bottom without errors (for .ipynb files)
- [ ] PR title format: `feat: {description} - Closes #{N}`
- [ ] PR targets `main` branch
- [ ] Do NOT merge — human will review
