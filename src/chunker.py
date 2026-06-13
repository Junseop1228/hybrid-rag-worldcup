"""
chunker.py — Wikipedia article fetcher and text chunker utility.

Fetches Wikipedia articles via the wikipedia-api library, splits them into
overlapping text chunks suitable for BM25 and vector-index retrieval, and
serialises/deserialises the chunk list to/from JSON.

References
----------
Dataset source (REFERENCES.md [9]):
    Wikipedia contributors. (2026). 2026 FIFA World Cup and related articles.
    Retrieved June 2026 from https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
    License: Creative Commons Attribution-ShareAlike 4.0

Chunking strategy (EXPERIMENT.md §Dataset):
    - Split by paragraph (double newline)
    - Discard chunks shorter than min_chars (default 50)
    - Keep paragraphs ≤ max_chars as-is
    - Split longer paragraphs at sentence boundaries ('. ', '? ', '! ')
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Wikipedia fetching
# ---------------------------------------------------------------------------

def fetch_wikipedia_page(title: str) -> Optional[dict]:
    """Fetch raw text and URL for a Wikipedia article.

    Uses the ``wikipediaapi`` library with language='en'.  Returns ``None``
    when the requested page does not exist rather than raising an exception.

    Args:
        title: The exact Wikipedia page title (e.g. ``"Lionel Messi"``).

    Returns:
        A dict with keys ``'title'``, ``'url'``, and ``'text'``; or ``None``
        if the page does not exist.

    Example:
        >>> page = fetch_wikipedia_page("Lionel Messi")
        >>> page is not None
        True
        >>> set(page.keys()) == {'title', 'url', 'text'}
        True
    """
    try:
        import wikipediaapi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wikipedia-api is required. Install with: pip install wikipedia-api"
        ) from exc

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="hybrid-rag-worldcup/1.0 (ITM411 course project)",
    )
    page = wiki.page(title)
    if not page.exists():
        return None
    return {
        "title": page.title,
        "url": page.fullurl,
        "text": page.text,
    }


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    url: str,
    max_chars: int = 400,
    min_chars: int = 50,
) -> list[dict]:
    """Split article text into retrievable chunks.

    Strategy (per EXPERIMENT.md §Chunking Strategy):

    1. Split the full text by double-newline (paragraph boundaries).
    2. Discard paragraphs shorter than *min_chars* (headers, captions, etc.).
    3. If the paragraph is ≤ *max_chars*: keep as a single chunk.
    4. If the paragraph is > *max_chars*: split at sentence boundaries
       (``'. '``, ``'? '``, ``'! '``) so that each resulting piece is at most
       *max_chars* characters.  Sentence fragments that are too short after
       splitting are merged with the preceding piece.

    Args:
        text:      Full article text returned by the Wikipedia API.
        source:    Human-readable name of the source article (e.g. page title).
        url:       Canonical URL of the Wikipedia article.
        max_chars: Maximum character length of a single chunk. Default 400.
        min_chars: Minimum character length; shorter paragraphs are discarded.
                   Default 50.

    Returns:
        A list of dicts, each with keys ``'source'``, ``'url'``, ``'text'``.
        IDs are *not* assigned here; ``build_chunk_list`` adds them.

    Example:
        >>> chunks = chunk_text("Hello world.\\n\\nShort.", "Test", "http://x")
        >>> all('source' in c and 'url' in c and 'text' in c for c in chunks)
        True
    """
    paragraphs = text.split("\n\n")
    chunks: list[dict] = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_chars:
            continue  # too short — skip

        if len(para) <= max_chars:
            chunks.append({"source": source, "url": url, "text": para})
        else:
            # Split at sentence boundaries
            sentences = _split_sentences(para)
            current = ""
            for sentence in sentences:
                candidate = (current + " " + sentence).strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        if len(current) >= min_chars:
                            chunks.append({"source": source, "url": url, "text": current})
                        current = sentence
                    else:
                        # single sentence exceeds max_chars — keep anyway
                        chunks.append({"source": source, "url": url, "text": sentence})
                        current = ""
            if current and len(current) >= min_chars:
                chunks.append({"source": source, "url": url, "text": current})

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries preserving the delimiter.

    Splits on ``. ``, ``? ``, and ``! `` while keeping the punctuation
    attached to the preceding sentence.

    Args:
        text: A paragraph of text to split.

    Returns:
        A list of sentence strings with leading/trailing whitespace stripped.

    Example:
        >>> _split_sentences("Hello world. How are you? Fine!")
        ['Hello world.', 'How are you?', 'Fine!']
    """
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Building the corpus
# ---------------------------------------------------------------------------

def build_chunk_list(pages: list[dict]) -> list[dict]:
    """Flatten a list of fetched pages into a single chunk list with sequential IDs.

    Calls ``chunk_text()`` for each page in *pages* and assigns a unique
    ``'id'`` field (0-indexed, sequential across all pages) to every chunk.

    Args:
        pages: A list of page dicts as returned by ``fetch_wikipedia_page``.
               Each dict must have keys ``'title'``, ``'url'``, and ``'text'``.

    Returns:
        A flat list of chunk dicts with keys ``'id'``, ``'source'``, ``'url'``,
        ``'text'``.  The list is ordered by page then by position within the
        page.

    Example:
        >>> page = {'title': 'Foo', 'url': 'http://foo', 'text': 'Para one.\\n\\nPara two and more text here to ensure min length reached.'}
        >>> chunks = build_chunk_list([page])
        >>> chunks[0]['id'] == 0
        True
    """
    flat: list[dict] = []
    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            source=page["title"],
            url=page["url"],
        )
        flat.extend(page_chunks)

    # Assign sequential IDs
    for idx, chunk in enumerate(flat):
        chunk["id"] = idx

    return flat


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    """Serialise a chunk list to a JSON file.

    Creates any missing parent directories automatically via ``os.makedirs``.

    Args:
        chunks: The list of chunk dicts to save.
        path:   Destination file path (e.g. ``'data/chunks.json'``).

    Returns:
        None

    Example:
        >>> import tempfile, os
        >>> with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        ...     tmp = f.name
        >>> save_chunks([{'id': 0, 'text': 'hello'}], tmp)
        Saved 1 chunks to ...
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path: str) -> list[dict]:
    """Load a chunk list from a JSON file.

    Args:
        path: Path to the JSON file previously written by ``save_chunks``.

    Returns:
        A list of chunk dicts.

    Raises:
        FileNotFoundError: If *path* does not exist, with a helpful message
            suggesting the user run ``01_data_collection.ipynb`` first.

    Example:
        >>> import tempfile, json
        >>> with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        ...     json.dump([{'id': 0, 'text': 'hi'}], f)
        ...     tmp = f.name
        >>> chunks = load_chunks(tmp)
        Loaded 1 chunks from ...
        >>> chunks[0]['text'] == 'hi'
        True
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunk file not found: {path}\n"
            "Please run notebooks/01_data_collection.ipynb first to generate the data."
        )
    with open(path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== chunker.py self-test ===")
    print("Fetching 'Association football' from Wikipedia ...")

    page = fetch_wikipedia_page("Association football")
    if page is None:
        print("ERROR: Page not found on Wikipedia.")
    else:
        print(f"  Title : {page['title']}")
        print(f"  URL   : {page['url']}")
        print(f"  Length: {len(page['text'])} characters")

        chunks = chunk_text(page["text"], source=page["title"], url=page["url"])
        print(f"\nchunk_text() produced {len(chunks)} chunks")
        print(f"  Shortest chunk : {min(len(c['text']) for c in chunks)} chars")
        print(f"  Longest chunk  : {max(len(c['text']) for c in chunks)} chars")
        print(f"  Average chunk  : {sum(len(c['text']) for c in chunks) // len(chunks)} chars")

        full_list = build_chunk_list([page])
        print(f"\nbuild_chunk_list() produced {len(full_list)} chunks with IDs")
        print(f"  First chunk id : {full_list[0]['id']}")
        print(f"  Last  chunk id : {full_list[-1]['id']}")
        print(f"\nSample chunk (id=0):")
        print(f"  {full_list[0]['text'][:200]} ...")

    print("\n=== self-test complete ===")
