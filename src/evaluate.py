"""
evaluate.py — Evaluation metrics and visualisation utilities.

Provides:
- nDCG@k computation for information-retrieval quality assessment
- Latency measurement for retrieval speed benchmarking
- Grouped bar chart plotting for system comparison
- Formatted ASCII table printing for quick result inspection

Evaluation methodology (EXPERIMENT.md §Evaluation Method):
    Primary metric: nDCG@3 (Normalised Discounted Cumulative Gain at rank 3)
    Secondary metric: average wall-clock retrieval latency in milliseconds

References:
    [3] Sultania, A. et al. (2024). Hybrid BM25 and Dense Retrieval for RAG
        Pipelines. https://www.emergentmind.com/topics/hybrid-bm25-retrieval
        — nDCG and MAP are the standard metrics for retrieval evaluation.

    [4] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
        Learning Methods. SIGIR 2009.
"""

from __future__ import annotations

import math
import time
from typing import Any


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------

def compute_ndcg(
    retrieved: list[dict],
    relevant_ids: dict,
    k: int = 3,
) -> float:
    """Compute nDCG@k (Normalised Discounted Cumulative Gain at rank k).

    Relevance scoring:
        - ``chunk_id`` in *relevant_ids* maps to its relevance score (0, 1, 2).
        - A score of 2 means the chunk directly answers the query.
        - A score of 1 means related but not directly answering.
        - Chunk IDs not in *relevant_ids* receive score 0.

    Args:
        retrieved:    Ordered list of result dicts from a retrieval system.
                      Each dict **must** contain a ``'chunk_id'`` field.
                      The first element is rank 1.
        relevant_ids: Dict mapping ``chunk_id`` (int) → relevance score
                      (int ∈ {0, 1, 2}).
        k:            Rank cutoff.  Default 3.

    Formula::

        DCG@k  = rel_1 + rel_2 / log2(3) + rel_3 / log2(4) + ...
                 (position i uses denominator log2(i + 1))
        IDCG@k = best possible DCG@k given relevance scores
        nDCG@k = DCG@k / IDCG@k   (returns 0.0 if IDCG@k == 0)

    Returns:
        float in [0.0, 1.0].  Returns 0.0 when *relevant_ids* has no
        positive-score entries (IDCG = 0 — cannot normalise).

    Example:
        >>> retrieved = [{'chunk_id': 1}, {'chunk_id': 2}, {'chunk_id': 3}]
        >>> relevant  = {1: 2, 3: 1}
        >>> score = compute_ndcg(retrieved, relevant, k=3)
        >>> round(score, 4)
        0.9614
    """
    # DCG@k
    dcg = 0.0
    for i, result in enumerate(retrieved[:k]):
        cid = result["chunk_id"]
        rel = relevant_ids.get(cid, 0)
        if i == 0:
            dcg += rel
        else:
            dcg += rel / math.log2(i + 2)  # denominator: log2(position + 1), position 1-indexed

    # IDCG@k — sort available relevance scores descending, take top-k
    sorted_rels = sorted(relevant_ids.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_rels):
        if i == 0:
            idcg += rel
        else:
            idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    retriever: Any,
    queries: list[str],
    top_k: int = 5,
    runs: int = 10,
) -> float:
    """Measure average retrieval latency in milliseconds.

    Runs each query in *queries* exactly *runs* times and averages the
    wall-clock time per query call.  Uses ``time.perf_counter()`` for
    sub-millisecond precision.

    Args:
        retriever: Any retrieval object with a ``search(query, top_k)``
                   method (``BM25Retriever``, ``VectorRetriever``, or
                   ``HybridRetriever``).
        queries:   List of query strings to benchmark.
        top_k:     Number of results requested per query call. Default 5.
        runs:      Number of repetitions per query. Default 10.

    Returns:
        Average latency per query call in milliseconds (float).

    Example:
        >>> import time
        >>> class FakeRetriever:
        ...     def search(self, q, top_k=5): time.sleep(0.001); return []
        >>> latency = measure_latency(FakeRetriever(), ["hello"], runs=3)
        >>> latency > 0
        True
    """
    total_time = 0.0
    total_calls = 0

    for query in queries:
        for _ in range(runs):
            start = time.perf_counter()
            retriever.search(query, top_k=top_k)
            end = time.perf_counter()
            total_time += end - start
            total_calls += 1

    avg_ms = (total_time / total_calls) * 1000.0
    return avg_ms


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, save_path: str = None) -> None:
    """Generate a grouped bar chart comparing BM25, Vector, and Hybrid per query.

    Args:
        results:   Nested dict mapping query label → system name → nDCG score::

                       {
                           'Q1': {'bm25': 0.8, 'vector': 0.4, 'hybrid': 0.9},
                           'Q2': {'bm25': 0.2, 'vector': 0.9, 'hybrid': 0.95},
                           ...
                       }

        save_path: If provided, the figure is saved to this file path
                   (e.g. ``'../data/comparison_chart.png'``).  When ``None``,
                   ``plt.show()`` is called instead.

    Chart specifications:
        - Figure size: (10, 6)
        - x-axis: query labels (Q1–Q5)
        - y-axis: nDCG@3 score, range [0, 1]
        - Bar colours: BM25 = steelblue, Vector = darkorange, Hybrid = mediumseagreen
        - Legend shown; grid on y-axis only
        - Title: ``"Retrieval System Comparison (nDCG@3)"``

    Returns:
        None

    Example:
        >>> results = {'Q1': {'bm25': 0.8, 'vector': 0.5, 'hybrid': 0.9}}
        >>> plot_comparison(results, save_path=None)  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for plotting. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    query_labels = list(results.keys())
    systems = ["bm25", "vector", "hybrid"]
    colours = {"bm25": "steelblue", "vector": "darkorange", "hybrid": "mediumseagreen"}
    display_names = {"bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid"}

    n_queries = len(query_labels)
    n_systems = len(systems)
    bar_width = 0.25
    x = np.arange(n_queries)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, system in enumerate(systems):
        scores = [results[q].get(system, 0.0) for q in query_labels]
        offsets = x + (i - n_systems / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            scores,
            bar_width,
            label=display_names[system],
            color=colours[system],
            alpha=0.85,
        )
        # Annotate bars with score values
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Query", fontsize=12)
    ax.set_ylabel("nDCG@3", fontsize=12)
    ax.set_title("Retrieval System Comparison (nDCG@3)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# ASCII results table
# ---------------------------------------------------------------------------

def print_results_table(results: dict) -> None:
    """Print a formatted ASCII table of nDCG@3 results.

    Args:
        results: Nested dict as accepted by ``plot_comparison``::

                     {
                         'Q1': {'bm25': 0.800, 'vector': 0.400, 'hybrid': 0.900},
                         ...
                     }

    Output format::

        Query  | BM25   | Vector | Hybrid
        -------|--------|--------|-------
        Q1     | 0.800  | 0.400  | 0.900
        ...
        Average| x.xxx  | x.xxx  | x.xxx

    Returns:
        None

    Example:
        >>> print_results_table({'Q1': {'bm25': 0.8, 'vector': 0.4, 'hybrid': 0.9}})
        Query  | BM25   | Vector | Hybrid
        -------|--------|--------|-------
        Q1     | 0.800  | 0.400  | 0.900
        -------|--------|--------|-------
        Average| 0.800  | 0.400  | 0.900
    """
    header = f"{'Query':<7}| {'BM25':<7}| {'Vector':<7}| {'Hybrid'}"
    sep    = "-" * 7 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 7
    print(header)
    print(sep)

    bm25_scores, vector_scores, hybrid_scores = [], [], []

    for query_label, scores in results.items():
        bm25   = scores.get("bm25", 0.0)
        vector = scores.get("vector", 0.0)
        hybrid = scores.get("hybrid", 0.0)
        bm25_scores.append(bm25)
        vector_scores.append(vector)
        hybrid_scores.append(hybrid)
        print(f"{query_label:<7}| {bm25:<7.3f}| {vector:<7.3f}| {hybrid:.3f}")

    print(sep)
    avg_bm25   = sum(bm25_scores)   / len(bm25_scores)   if bm25_scores   else 0.0
    avg_vector = sum(vector_scores) / len(vector_scores) if vector_scores else 0.0
    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0.0
    print(f"{'Average':<7}| {avg_bm25:<7.3f}| {avg_vector:<7.3f}| {avg_hybrid:.3f}")
