"""
rag_storage.py
--------------
Retrieval-Augmented Generation (RAG) storage layer.

Ingests .txt and .pdf documents, chunks them, embeds with
sentence-transformers, and indexes with FAISS for fast retrieval.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_cfg():
    from config import get_config
    return get_config()


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.

    Parameters
    ----------
    text:
        Full document text.
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Characters to overlap between consecutive chunks.

    Returns
    -------
    list[str]
        Non-empty text chunks.
    """
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Document readers
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    """Read a plain-text file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""


def _read_pdf(path: Path) -> str:
    """Read a PDF file using pypdf if available, otherwise return empty string."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        logger.warning(
            "pypdf is not installed; skipping PDF file %s. "
            "Run: pip install pypdf",
            path,
        )
        return ""

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.warning("Failed to read PDF %s: %s", path, exc)
        return ""


def _load_documents(docs_dir: str) -> List[dict]:
    """
    Scan *docs_dir* for .txt and .pdf files, returning a list of
    ``{"source": str, "text": str}`` dicts.
    """
    docs: List[dict] = []
    base = Path(docs_dir)

    if not base.exists():
        logger.warning("docs_dir '%s' does not exist.", docs_dir)
        return docs

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = _read_txt(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        text = text.strip()
        if text:
            docs.append({"source": str(path), "text": text})
            logger.debug("Loaded document: %s (%d chars)", path.name, len(text))

    logger.info("Loaded %d document(s) from '%s'.", len(docs), docs_dir)
    return docs


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Manages the FAISS index, embeddings, and retrieval logic.

    Typical workflow
    ----------------
    1. ``engine = RAGEngine()``
    2. ``engine.ingest("./docs")``  — or ``engine.load()`` if already indexed
    3. ``chunks = engine.retrieve("What are your hours?", top_k=3)``
    """

    def __init__(self, cfg=None):
        """
        Parameters
        ----------
        cfg:
            AppConfig instance.  If None, the global singleton is used.
        """
        if cfg is None:
            cfg = _get_cfg()

        self._rag_cfg = cfg.rag
        self._index_path = Path(cfg.rag.index_path)
        self._metadata_path = Path(cfg.rag.metadata_path)
        self._embedding_model_name = cfg.rag.embedding_model
        self._chunk_size = cfg.rag.chunk_size
        self._chunk_overlap = cfg.rag.chunk_overlap

        self._index = None          # faiss.IndexFlatIP
        self._metadata: List[dict] = []   # [{source, chunk_id, text}, …]
        self._embedder = None       # SentenceTransformer

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the SentenceTransformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            logger.info("Loading embedding model: %s", self._embedding_model_name)
            # Use Metal (MPS) on Apple Silicon for ~5x faster encoding
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info("Embedding device: %s", device)
            self._embedder = SentenceTransformer(self._embedding_model_name, device=device)
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns
        -------
        np.ndarray
            shape (N, D), float32, L2-normalised.
        """
        embedder = self._get_embedder()
        vecs = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, docs_dir: Optional[str] = None) -> int:
        """
        Scan *docs_dir*, chunk all documents, build FAISS index, and save to disk.

        Parameters
        ----------
        docs_dir:
            Directory to scan.  Defaults to the value in config.

        Returns
        -------
        int
            Number of text chunks indexed.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        if docs_dir is None:
            docs_dir = self._rag_cfg.docs_dir

        t0 = time.perf_counter()
        documents = _load_documents(docs_dir)

        if not documents:
            logger.warning("No documents found in '%s'. Index will be empty.", docs_dir)
            return 0

        # Chunk all documents
        all_chunks: List[dict] = []
        for doc in documents:
            chunks = _chunk_text(doc["text"], self._chunk_size, self._chunk_overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": doc["source"],
                    "chunk_id": idx,
                    "text": chunk,
                })

        logger.info("Created %d text chunks from %d documents.", len(all_chunks), len(documents))

        # Embed
        texts = [c["text"] for c in all_chunks]
        logger.info("Embedding %d chunks …", len(texts))
        embeddings = self._embed(texts)

        # Build FAISS inner-product index (vectors are already L2-normalised,
        # so inner product == cosine similarity)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._index = index
        self._metadata = all_chunks

        # Persist to disk
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._index_path))
        with self._metadata_path.open("wb") as fh:
            pickle.dump(all_chunks, fh)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "RAG index built: %d chunks, dim=%d (%.0f ms). "
            "Saved to %s.",
            len(all_chunks),
            dim,
            elapsed_ms,
            self._index_path,
        )
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.

        Returns
        -------
        bool
            True if loaded successfully, False if files do not exist.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        if not self._index_path.exists() or not self._metadata_path.exists():
            logger.info(
                "RAG index not found at '%s'. Run ingest() first.", self._index_path
            )
            return False

        try:
            self._index = faiss.read_index(str(self._index_path))
            with self._metadata_path.open("rb") as fh:
                self._metadata = pickle.load(fh)
            logger.info(
                "RAG index loaded: %d chunks, dim=%d.",
                self._index.ntotal,
                self._index.d,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load RAG index: %s", exc)
            self._index = None
            self._metadata = []
            return False

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve the most relevant text chunks for *query*.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Number of chunks to return.  Defaults to the value in config.

        Returns
        -------
        list[str]
            Retrieved text chunks, ordered by relevance (highest first).
            Returns ``["No relevant context found."]`` when the index is
            empty or no results match.
        """
        if top_k is None:
            top_k = self._rag_cfg.top_k

        if self._index is None or self._index.ntotal == 0:
            logger.warning("RAG index not loaded or empty. Returning fallback context.")
            return ["No relevant context found."]

        if not query or not query.strip():
            logger.debug("Empty query; returning fallback context.")
            return ["No relevant context found."]

        t0 = time.perf_counter()
        query_vec = self._embed([query.strip()])  # (1, D)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vec, k)

        results: List[str] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS can return -1 for missing results
                continue
            chunk = self._metadata[idx]
            results.append(chunk["text"])
            logger.debug("  RAG hit [%d] dist=%.3f: %s…", idx, dist, chunk["text"][:60])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("RAG retrieve: %d results in %.0f ms for query='%s'", len(results), elapsed_ms, query[:60])

        return results if results else ["No relevant context found."]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """Return True if the FAISS index is loaded and contains at least one vector."""
        return self._index is not None and self._index.ntotal > 0
