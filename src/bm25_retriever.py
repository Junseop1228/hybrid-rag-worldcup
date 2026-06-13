"""
bm25_retriever.py — BM25 sparse keyword retrieval engine.

Implements the BM25Okapi algorithm via the ``rank_bm25`` library to build
an inverted-index-like retrieval system over a pre-chunked text corpus.

Conceptual connection (CONCEPT.md §3):
    BM25 is conceptually a sparse inverted index: each unique term maps to a
    list of documents containing it — structurally similar to a B+ Tree
    mapping keys to record pointers.

Reference (REFERENCES.md [7]):
    Robertson, S., & Zaragoza, H. (2009).
    The Probabilistic Relevance Framework: BM25 and Beyond.
    Foundations and Trends in Information Retrieval, 3(4), 333–389.
"""

from __future__ import annotations

import re
import string
from typing import Optional


class BM25Retriever:
    """Sparse keyword retrieval using the BM25Okapi algorithm.

    BM25 (Best Match 25) scores documents by term frequency and inverse
    document frequency, penalising very long documents to reduce noise.

    Reference: Robertson & Zaragoza (2009) — REFERENCES.md [7]

    Attributes:
        chunks: Internal copy of the corpus chunk list.
        index:  A ``BM25Okapi`` instance after ``build_index`` is called.

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.build_index([{'id': 0, 'text': 'hello world', 'source': 'test', 'url': ''}])
        BM25 index built: 1 chunks
        >>> results = retriever.search('hello', top_k=1)
        >>> results[0]['rank']
        1
    """

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.index = None  # BM25Okapi instance

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> list[str]:
        """Lowercase, remove punctuation, and split by whitespace.

        Args:
            text: Raw text to tokenise.

        Returns:
            A list of lowercase token strings with punctuation stripped.

        Example:
            >>> r = BM25Retriever()
            >>> r._preprocess("Hello, World! Foo-bar.")
            ['hello', 'world', 'foobar']
        """
        text = text.lower()
        # Remove punctuation characters
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
        tokens = text.split()
        return tokens

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """Tokenise all chunk texts and build a BM25Okapi index.

        The full corpus is stored internally so that ``search`` can
        reconstruct result dicts with source and text metadata.

        Args:
            chunks: List of chunk dicts. Each must have at minimum the key
                    ``'text'``. Keys ``'id'``, ``'source'``, and ``'url'``
                    are preserved in search results.

        Returns:
            None

        Raises:
            ImportError: If ``rank_bm25`` is not installed.

        Example:
            >>> r = BM25Retriever()
            >>> r.build_index([{'id': 0, 'text': 'test', 'source': 's', 'url': ''}])
            BM25 index built: 1 chunks
            >>> r.index is not None
            True
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required. Install with: pip install rank-bm25"
            ) from exc

        self.chunks = chunks
        tokenised_corpus = [self._preprocess(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(tokenised_corpus)
        print(f"BM25 index built: {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Score all chunks against *query* using BM25 and return top results.

        Args:
            query:  Natural-language query string.
            top_k:  Maximum number of results to return. Default 5.

        Returns:
            A list of up to *top_k* result dicts sorted by BM25 score
            descending.  Each dict contains::

                {
                    'rank':     int,   # 1-indexed position in results
                    'score':    float, # BM25 score
                    'chunk_id': int,   # original chunk id
                    'source':   str,   # article title
                    'text':     str,   # chunk text
                }

        Raises:
            RuntimeError: If ``build_index`` has not been called yet.

        Example:
            >>> r = BM25Retriever()
            >>> r.build_index([{'id': 0, 'text': 'Messi scored goals', 'source': 'Messi', 'url': ''}])
            BM25 index built: 1 chunks
            >>> r.search('goals')[0]['chunk_id']
            0
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_tokens = self._preprocess(query)
        scores = self.index.get_scores(query_tokens)

        # Pair scores with chunk indices and sort
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results = []
        for rank, (idx, score) in enumerate(top, start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.get("id", idx),
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

    # Resolve data path relative to this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "chunks.json")

    print("=== bm25_retriever.py self-test ===")

    # Load chunks
    try:
        from src.chunker import load_chunks  # type: ignore
    except ImportError:
        # Allow running as a standalone script from the repo root
        sys.path.insert(0, os.path.join(script_dir, ".."))
        from src.chunker import load_chunks  # type: ignore

    chunks = load_chunks(data_path)

    # Build index
    retriever = BM25Retriever()
    retriever.build_index(chunks)

    # Run test query
    QUERY = "How many goals did Messi score?"
    print(f"\nQuery: '{QUERY}'")
    print("Top 3 results:")
    for result in retriever.search(QUERY, top_k=3):
        print(f"  Rank {result['rank']} | Score {result['score']:.4f} | {result['source']}")
        print(f"    {result['text'][:120]} ...")

    print("\n=== self-test complete ===")
