"""
hybrid_retriever.py — Hybrid retrieval engine combining BM25 + Vector via RRF.

Implements Reciprocal Rank Fusion (RRF) to merge the ranked result lists
from a ``BM25Retriever`` and a ``VectorRetriever`` into a single, unified
ranking that outperforms either system alone.

Conceptual connection (CONCEPT.md §5):
    No single index type dominates across all query types.  The 2024–2025
    research consensus is that hybrid retrieval (sparse + dense + fusion)
    consistently outperforms either alone.  RRF avoids the score-scale
    incompatibility problem by operating only on rank positions.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))  where k = 60

References:
    [3] Sultania, A. et al. (2024). Hybrid BM25 and Dense Retrieval for RAG
        Pipelines. Rayo, J. et al. (2025). Domain-specific Hybrid Retrieval.
        https://www.emergentmind.com/topics/hybrid-bm25-retrieval

    [4] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
        Learning Methods. SIGIR 2009, pp. 758–759.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.bm25_retriever import BM25Retriever
    from src.vector_retriever import VectorRetriever


class HybridRetriever:
    """Hybrid retrieval combining BM25 (sparse) and Vector (dense) search
    via Reciprocal Rank Fusion (RRF).

    Both component retrievers must have their indexes built before being
    passed to ``HybridRetriever``.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d)),  k = 60
    Reference: Cormack et al. (2009) SIGIR — REFERENCES.md [4]

    Attributes:
        bm25:   Pre-built ``BM25Retriever`` instance.
        vector: Pre-built ``VectorRetriever`` instance.

    Example:
        >>> # (assumes bm25 and vector already have build_index() called)
        >>> hybrid = HybridRetriever(bm25, vector)  # doctest: +SKIP
        >>> results = hybrid.search("youngest player World Cup")  # doctest: +SKIP
        >>> results[0]['rank']  # doctest: +SKIP
        1
    """

    def __init__(self, bm25: "BM25Retriever", vector: "VectorRetriever") -> None:
        """Initialise the hybrid retriever with two pre-built sub-retrievers.

        Args:
            bm25:   A ``BM25Retriever`` instance whose ``build_index`` has
                    already been called.
            vector: A ``VectorRetriever`` instance whose ``build_index`` has
                    already been called.

        Returns:
            None

        Raises:
            ValueError: If either retriever's index is ``None``.

        Example:
            >>> hr = HybridRetriever(bm25_instance, vector_instance)  # doctest: +SKIP
        """
        if bm25.index is None:
            raise ValueError("BM25Retriever.index is None — call build_index() first.")
        if vector.index is None:
            raise ValueError("VectorRetriever.index is None — call build_index() first.")

        self.bm25 = bm25
        self.vector = vector

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: list[list[dict]],
        k: int = 60,
    ) -> list[dict]:
        """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

        RRF is scale-invariant: it doesn't matter that BM25 scores are in the
        thousands while cosine scores are in [−1, 1].  Only rank order matters.

        Per Cormack et al. (2009) the optimal smoothing constant is k=60.

        Args:
            rankings: A list of ranked result lists, one per retrieval system.
                      Each result dict **must** contain a ``'chunk_id'`` field.
                      Rank within each list is determined by list position
                      (index 0 = rank 1).
            k:        RRF smoothing constant.  Default 60 per Cormack et al.

        Logic:
            - For each result list, assign ranks 1 … N (1-indexed).
            - Documents not appearing in a list receive rank = len(list) + 1.
            - Final RRF score = Σ  1 / (k + rank_i(d))  across all lists.
            - Results are sorted by RRF score descending.

        Returns:
            A list of dicts::

                {
                    'chunk_id':    int,   # original chunk id
                    'rrf_score':   float, # sum of reciprocal ranks
                    'bm25_rank':   int,   # rank in the first (BM25) list
                    'vector_rank': int,   # rank in the second (vector) list
                }

            The list is ordered from highest to lowest RRF score.

        Example:
            >>> ranking_a = [{'chunk_id': 1}, {'chunk_id': 2}]
            >>> ranking_b = [{'chunk_id': 2}, {'chunk_id': 3}]
            >>> fused = HybridRetriever.reciprocal_rank_fusion([ranking_a, ranking_b])
            >>> fused[0]['chunk_id']  # chunk 2 appears in both lists
            2
        """
        if not rankings:
            return []

        # Collect all unique chunk IDs across all ranking lists
        all_chunk_ids: set[int] = set()
        for ranking in rankings:
            for result in ranking:
                all_chunk_ids.add(result["chunk_id"])

        # Build rank lookup: {chunk_id: rank} for each ranking list
        # Documents missing from a list get penalty rank = len(list) + 1
        rank_maps: list[dict[int, int]] = []
        for ranking in rankings:
            penalty = len(ranking) + 1
            rank_map: dict[int, int] = {result["chunk_id"]: (i + 1) for i, result in enumerate(ranking)}
            # Fill missing chunk_ids with penalty rank
            for cid in all_chunk_ids:
                if cid not in rank_map:
                    rank_map[cid] = penalty
            rank_maps.append(rank_map)

        # Compute RRF scores
        fused: list[dict] = []
        for cid in all_chunk_ids:
            rrf_score = sum(1.0 / (k + rank_map[cid]) for rank_map in rank_maps)
            entry: dict = {
                "chunk_id": cid,
                "rrf_score": rrf_score,
            }
            # Store individual ranks (supports exactly 2 lists: bm25 + vector)
            if len(rank_maps) >= 1:
                entry["bm25_rank"] = rank_maps[0][cid]
            if len(rank_maps) >= 2:
                entry["vector_rank"] = rank_maps[1][cid]
            fused.append(entry)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Unified search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run BM25 and Vector search then fuse results with RRF.

        Internally calls ``bm25.search()`` and ``vector.search()`` with a
        large candidate pool (``top_k * 10``) to ensure good coverage before
        fusion, then trims to *top_k* after RRF re-ranking.

        Args:
            query:  Natural-language query string.
            top_k:  Number of results to return. Default 5.

        Returns:
            A list of up to *top_k* result dicts sorted by RRF score
            descending.  Each dict contains::

                {
                    'rank':        int,   # 1-indexed final position
                    'rrf_score':   float, # RRF fusion score
                    'bm25_rank':   int,   # rank in the BM25 result list
                    'vector_rank': int,   # rank in the vector result list
                    'chunk_id':    int,   # original chunk id
                    'source':      str,   # article title
                    'text':        str,   # chunk text
                }

        Example:
            >>> results = hybrid.search("youngest World Cup player", top_k=3)  # doctest: +SKIP
            >>> 'rrf_score' in results[0]
            True
        """
        # Use a larger pool for fusion to avoid cutting off good candidates
        fetch_k = max(top_k * 10, len(self.bm25.chunks))

        bm25_results = self.bm25.search(query, top_k=fetch_k)
        vector_results = self.vector.search(query, top_k=fetch_k)

        fused = self.reciprocal_rank_fusion([bm25_results, vector_results])
        top_fused = fused[:top_k]

        # Build a chunk lookup for text/source enrichment
        chunk_lookup: dict[int, dict] = {
            c.get("id", i): c for i, c in enumerate(self.bm25.chunks)
        }

        results = []
        for rank, item in enumerate(top_fused, start=1):
            cid = item["chunk_id"]
            chunk = chunk_lookup.get(cid, {})
            results.append(
                {
                    "rank": rank,
                    "rrf_score": item["rrf_score"],
                    "bm25_rank": item.get("bm25_rank", -1),
                    "vector_rank": item.get("vector_rank", -1),
                    "chunk_id": cid,
                    "source": chunk.get("source", ""),
                    "text": chunk.get("text", ""),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, ".."))

    from src.chunker import load_chunks  # type: ignore
    from src.bm25_retriever import BM25Retriever  # type: ignore
    from src.vector_retriever import VectorRetriever  # type: ignore

    data_path = os.path.join(script_dir, "..", "data", "chunks.json")
    print("=== hybrid_retriever.py self-test ===")

    chunks = load_chunks(data_path)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    vector = VectorRetriever()
    vector.build_index(chunks)

    hybrid = HybridRetriever(bm25, vector)

    QUERY = "youngest player with most World Cup appearances"
    print(f"\nQuery: '{QUERY}'")
    print("Top 3 hybrid results:")
    for result in hybrid.search(QUERY, top_k=3):
        print(
            f"  Rank {result['rank']} | RRF={result['rrf_score']:.5f}"
            f" | BM25_rank={result['bm25_rank']} | Vec_rank={result['vector_rank']}"
            f" | {result['source']}"
        )
        print(f"    {result['text'][:120]} ...")

    print("\n=== self-test complete ===")
