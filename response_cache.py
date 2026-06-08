"""
response_cache.py
-----------------
Semantic response cache for the kiosk pipeline.

Pre-written Q→A pairs are embedded once at startup and stored in a small
FAISS index. At runtime the user's query is embedded and matched against
cached questions. If the cosine similarity exceeds the threshold the cached
answer is returned immediately — no LLM call needed.

Cache hit path:  STT → embed+lookup (~40ms) → TTS   ≈ 270ms E2E
Cache miss path: falls through to full RAG → LLM pipeline
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Q&A pairs — edit or extend for your kiosk content
# Keep questions phrased the way visitors actually speak them.
# ---------------------------------------------------------------------------

DEFAULT_QA: list[dict] = [
    {
        "questions": [
            "what are your hospital timings",
            "when is the hospital open",
            "what time does the hospital open",
            "what time does the hospital close",
            "are you open on weekends",
            "hospital opening hours",
            "what are the visiting hours"
        ],
        "answer": "The hospital operates twenty four hours a day. Visitor hours are from 5 PM to 8 PM daily."
    },

    {
        "questions": [
            "when is the cardiologist available",
            "cardiology timings",
            "what time does the heart doctor come",
            "cardiologist availability",
            "when can i visit the cardiologist",
            "heart specialist timings"
        ],
        "answer": "The cardiology specialist is available Monday to Friday from 9 AM to 1 PM."
    },

    {
        "questions": [
            "what are the orthopedic doctor timings",
            "orthopedic availability",
            "bone specialist timings",
            "when is the orthopedic doctor available",
            "orthopedic department timings"
        ],
        "answer": "The orthopedic specialist is available on weekdays from 2 PM to 6 PM."
    },

    {
        "questions": [
            "what is the opd fee",
            "general consultation charges",
            "how much is the opd consultation",
            "doctor consultation fee",
            "general opd charges"
        ],
        "answer": "The general OPD consultation fee is 1500 Pakistani Rupees."
    },

    {
        "questions": [
            "what are the emergency charges",
            "emergency consultation fee",
            "how much does emergency cost",
            "emergency department charges",
            "emergency fees"
        ],
        "answer": "Emergency consultation charges start from 5000 Pakistani Rupees depending on the case."
    },

    {
        "questions": [
            "is the pharmacy open at night",
            "pharmacy timings",
            "when does the pharmacy close",
            "hospital pharmacy hours",
            "is pharmacy open twenty four hours"
        ],
        "answer": "The hospital pharmacy is open twenty four hours every day."
    },

    {
        "questions": [
            "how can i book an appointment",
            "appointment booking process",
            "how do i schedule a doctor appointment",
            "can i book through the kiosk",
            "ways to book appointment"
        ],
        "answer": "Appointments can be booked through the reception desk, hospital website, or self service kiosk."
    },

    {
        "questions": [
            "what documents are needed for admission",
            "admission requirements",
            "required documents for hospital admission",
            "what should i bring for admission",
            "patient admission process"
        ],
        "answer": "Patients must provide a valid CNIC, previous medical records, and insurance information during admission."
    },

    {
        "questions": [
            "does the hospital accept insurance",
            "insurance providers",
            "which insurance companies are accepted",
            "can i use insurance here",
            "insurance coverage information"
        ],
        "answer": "The hospital accepts major insurance providers including Jubilee Insurance, EFU Health, and State Life."
    },

    {
        "questions": [
            "what are the lab timings",
            "laboratory timings",
            "when is sample collection available",
            "lab collection hours",
            "pathology timings"
        ],
        "answer": "Laboratory sample collection is available daily from 7 AM to 10 PM."
    }
]




# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Embeds all canonical questions at startup and matches incoming queries
    via cosine similarity using FAISS.
    """

    def __init__(self, threshold: float = 0.78, qa_pairs: Optional[list] = None):
        """
        Parameters
        ----------
        threshold:
            Minimum cosine similarity (0–1) to accept a cache hit.
            0.82 gives good precision; lower it (e.g. 0.75) to match more
            aggressively, raise it (e.g. 0.90) to be more conservative.
        qa_pairs:
            List of ``{"questions": [...], "answer": "..."}`` dicts.
            Defaults to DEFAULT_QA.
        """
        self._threshold = threshold
        self._qa_pairs = qa_pairs or DEFAULT_QA
        self._index = None      # faiss.IndexFlatIP
        self._answers: list[str] = []   # parallel to index rows
        self._embedder = None

    # ------------------------------------------------------------------

    def _get_embedder(self):
        from rag_storage import _get_shared_embedder
        return _get_shared_embedder("all-MiniLM-L6-v2")

    def build(self) -> int:
        """Embed all questions and build the FAISS index. No-op if already built."""
        if self._index is not None:
            return len(self._answers)

        import faiss

        embedder = self._get_embedder()
        questions: list[str] = []
        answers: list[str] = []

        for pair in self._qa_pairs:
            for q in pair["questions"]:
                questions.append(q.lower().strip())
                answers.append(pair["answer"])

        t0 = time.perf_counter()
        vecs = embedder.encode(
            questions,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        self._index = index
        self._answers = answers
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Response cache built: %d entries, dim=%d, %.0f ms",
            len(questions), dim, elapsed,
        )
        return len(questions)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string and return a normalized float32 vector (shape 1, D)."""
        embedder = self._get_embedder()
        return embedder.encode(
            [query.lower().strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def lookup_vec(self, query_vec: np.ndarray) -> Optional[str]:
        """Lookup using a pre-computed query embedding. Returns answer or None."""
        if self._index is None:
            return None
        D, I = self._index.search(query_vec, 1)
        similarity = float(D[0][0])
        if similarity >= self._threshold:
            answer = self._answers[int(I[0][0])]
            logger.info("Cache HIT  sim=%.3f", similarity)
            return answer
        logger.info("Cache MISS sim=%.3f → LLM", similarity)
        return None

    def lookup(self, query: str) -> Optional[str]:
        """
        Look up a cached answer for *query*.

        Returns the answer string on a cache hit, or None on a miss.
        """
        if self._index is None:
            return None

        embedder = self._get_embedder()
        t0 = time.perf_counter()

        vec = embedder.encode(
            [query.lower().strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        D, I = self._index.search(vec, 1)
        similarity = float(D[0][0])
        elapsed = (time.perf_counter() - t0) * 1000

        if similarity >= self._threshold:
            answer = self._answers[int(I[0][0])]
            logger.info(
                "Cache HIT  sim=%.3f  (%.0f ms)  query=%r",
                similarity, elapsed, query,
            )
            return answer

        logger.info(
            "Cache MISS sim=%.3f  (%.0f ms)  query=%r → LLM",
            similarity, elapsed, query,
        )
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
