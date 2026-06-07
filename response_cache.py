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
    # Hours
    {
        "questions": [
            "what are your hours",
            "when are you open",
            "what time do you open",
            "what time do you close",
            "are you open on weekends",
            "opening hours",
            "what are the opening times",
        ],
        "answer": "We are open Monday to Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM. We are closed on Sundays.",
    },
    # Location
    {
        "questions": [
            "where are you located",
            "where is this kiosk",
            "what floor are you on",
            "how do I find you",
            "where is wavetec",
            "where is the office",
        ],
        "answer": "The WaveTec InfoPoint kiosk is located in the main lobby of Tower A, near the entrance.",
    },
    # Services
    {
        "questions": [
            "what services do you offer",
            "what can you help me with",
            "what do you do",
            "how can you help me",
            "what is this kiosk for",
            "what can this kiosk do",
        ],
        "answer": "I can help with visitor registration, building wayfinding, event information, and general questions about WaveTec InfoPoint.",
    },
    # WiFi
    {
        "questions": [
            "what is the wifi password",
            "how do I connect to wifi",
            "how do I connect to the internet",
            "do you have wifi",
            "internet access",
            "wireless network",
            "how do I get internet",
            "wifi network",
        ],
        "answer": "The guest Wi-Fi network is WaveTec-Guest. Please ask reception for the current password.",
    },
    # Parking
    {
        "questions": [
            "where can I park",
            "is there parking",
            "parking information",
            "where is the car park",
            "parking area",
        ],
        "answer": "Visitor parking is available in the underground car park accessed from Level B1. The first two hours are complimentary.",
    },
    # Contact
    {
        "questions": [
            "what is your phone number",
            "how do I contact you",
            "who do I speak to",
            "reception number",
            "contact information",
        ],
        "answer": "You can reach our reception desk on the ground floor or call us. Please check the directory board for the current contact numbers.",
    },
    # Restrooms
    {
        "questions": [
            "where are the restrooms",
            "where is the bathroom",
            "where is the toilet",
            "where are the toilets",
        ],
        "answer": "Restrooms are located on every floor, next to the elevator lobby.",
    },
    # Events
    {
        "questions": [
            "what events are on today",
            "are there any events",
            "what is happening today",
            "events schedule",
            "today's events",
        ],
        "answer": "For today's events and bookings, please check the digital notice board in the lobby or speak with reception.",
    },
    # Visitor registration
    {
        "questions": [
            "how do I register as a visitor",
            "visitor registration",
            "I am here to visit someone",
            "I have an appointment",
            "how do I sign in",
        ],
        "answer": "Please tap 'Visitor Check-In' on my screen or proceed to the reception desk. You will need a valid ID and the name of your host.",
    },
    # Accessibility
    {
        "questions": [
            "is there wheelchair access",
            "disabled access",
            "accessibility",
            "lift location",
            "where is the elevator",
            "where is the lift",
        ],
        "answer": "The building is fully accessible. Elevators are located at the centre of each floor and accessible entrances are on the ground level.",
    },
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
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return self._embedder

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
