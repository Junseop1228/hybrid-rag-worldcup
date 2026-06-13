"""
vector_retriever.py — Dense vector retrieval engine using sentence-transformers + FAISS.

Maps chunk texts into a 384-dimensional semantic embedding space using the
``all-MiniLM-L6-v2`` model, builds a FAISS ``IndexFlatIP`` index, and
answers queries via cosine-similarity nearest-neighbour search.

Conceptual connection (CONCEPT.md §4):
    Unlike a B+ Tree (which requires a total order on scalar keys), a vector
    index organises points in high-dimensional space and retrieves by
    proximity.  This enables *semantic* lookup — "creative dribbler" can
    match "exceptional ball control" even with no shared tokens.

References:
    [2] Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain
        Question Answering. EMNLP 2020. https://arxiv.org/abs/2004.04906

    [5] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity
        search with GPUs. IEEE Transactions on Big Data, 7(3), 535–547.
        https://github.com/facebookresearch/faiss

    [6] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
        using Siamese BERT-Networks. EMNLP 2019. https://www.sbert.net
"""

from __future__ import annotations

import numpy as np


class VectorRetriever:
    """Dense semantic retrieval using sentence-transformers embeddings and FAISS index.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` to embed text into
    384-dimensional vectors, then builds a FAISS ``IndexFlatIP`` (inner
    product) index over L2-normalised vectors so that inner product equals
    cosine similarity.

    Reference: Karpukhin et al. (2020) Dense Passage Retrieval — REFERENCES.md [2]

    Attributes:
        MODEL_NAME: The HuggingFace model identifier (class-level constant).
        chunks:     Internal copy of the corpus chunk list.
        model:      Lazy-loaded ``SentenceTransformer`` instance.
        index:      FAISS ``IndexFlatIP`` instance after ``build_index``.
        embeddings: ``np.ndarray`` of shape ``(N, 384)`` storing chunk vectors.

    Example:
        >>> vr = VectorRetriever()
        >>> vr.build_index([{'id': 0, 'text': 'Messi is a creative dribbler', 'source': 'test', 'url': ''}])
        Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
        Vector index built: 1 chunks, dim=384
        >>> results = vr.search('creative dribbling', top_k=1)
        >>> results[0]['rank']
        1
    """

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # dimension: 384, license: Apache 2.0

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.model = None          # SentenceTransformer instance
        self.index = None          # faiss.IndexFlatIP instance
        self.embeddings: np.ndarray | None = None  # shape (N, 384)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model on first use.

        Prints a status message before downloading/loading so the user
        knows the first call may be slow (~90 MB download).

        Args:
            None

        Returns:
            None

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.

        Example:
            >>> vr = VectorRetriever()
            >>> vr.model is None
            True
            >>> vr._load_model()  # doctest: +SKIP
            Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
        """
        if self.model is not None:
            return  # already loaded

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        print(f"Loading embedding model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalised float32 vectors.

        Calls ``_load_model()`` if needed, then encodes *texts* using the
        sentence-transformer.  Vectors are L2-normalised so that FAISS
        inner-product search is equivalent to cosine similarity.

        Args:
            texts: List of text strings to embed.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), 384)``, dtype ``float32``,
            with each row normalised to unit length.

        Example:
            >>> vr = VectorRetriever()
            >>> vr._load_model()  # doctest: +SKIP
            Loading embedding model: ...
            >>> vecs = vr._embed(["hello world"])  # doctest: +SKIP
            >>> vecs.shape[1]
            384
        """
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        # L2-normalise: after normalisation, dot(a, b) == cosine_similarity(a, b)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
        embeddings = embeddings / norms
        return embeddings

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """Embed all chunk texts and build a FAISS ``IndexFlatIP`` index.

        Uses inner product on L2-normalised vectors which equals cosine
        similarity.  ``IndexFlatIP`` performs exact (non-approximate) search,
        which is feasible for our ~100 chunk corpus.

        Args:
            chunks: List of chunk dicts. Each must have ``'text'``. Keys
                    ``'id'``, ``'source'``, and ``'url'`` are used in results.

        Returns:
            None

        Raises:
            ImportError: If ``faiss-cpu`` (or ``faiss-gpu``) is not installed.

        Example:
            >>> vr = VectorRetriever()
            >>> vr.build_index([{'id': 0, 'text': 'test', 'source': 's', 'url': ''}])  # doctest: +SKIP
            Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
            Vector index built: 1 chunks, dim=384
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = self._embed(texts)

        dim = self.embeddings.shape[1]  # 384
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"Vector index built: {len(chunks)} chunks, dim={dim}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed *query* and retrieve the top-k nearest chunks by cosine similarity.

        Args:
            query:  Natural-language query string.
            top_k:  Maximum number of results to return. Default 5.

        Returns:
            A list of up to *top_k* result dicts sorted by cosine similarity
            descending.  Each dict contains::

                {
                    'rank':     int,   # 1-indexed position
                    'score':    float, # cosine similarity in [−1, 1]
                    'chunk_id': int,   # original chunk id field
                    'source':   str,   # article title
                    'text':     str,   # chunk text
                }

        Raises:
            RuntimeError: If ``build_index`` has not been called yet.

        Example:
            >>> vr = VectorRetriever()
            >>> # (after build_index is called)
            >>> results = vr.search("creative dribbling", top_k=3)  # doctest: +SKIP
            >>> results[0]['rank']
            1
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self._embed([query])  # shape (1, 384)
        scores, indices = self.index.search(query_vec, top_k)  # both shape (1, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:  # FAISS returns -1 for padding when corpus < top_k
                break
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.get("id", int(idx)),
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
    data_path = os.path.join(script_dir, "..", "data", "chunks.json")

    print("=== vector_retriever.py self-test ===")

    sys.path.insert(0, os.path.join(script_dir, ".."))
    from src.chunker import load_chunks  # type: ignore

    chunks = load_chunks(data_path)

    retriever = VectorRetriever()
    retriever.build_index(chunks)

    QUERY = "creative dribbling and exceptional vision"
    print(f"\nQuery: '{QUERY}'")
    print("Top 3 results:")
    for result in retriever.search(QUERY, top_k=3):
        print(f"  Rank {result['rank']} | Score {result['score']:.4f} | {result['source']}")
        print(f"    {result['text'][:120]} ...")

    print("\n=== self-test complete ===")
