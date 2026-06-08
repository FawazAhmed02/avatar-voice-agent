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
            "hospital opening hours",
            "are you open twenty four hours",
            "is the hospital open at night"
        ],
        "answer": "The hospital operates twenty four hours a day including weekends and public holidays."
    },

    {
        "questions": [
            "what are the visitor timings",
            "when can visitors come",
            "visitor hours",
            "what time can i visit patients",
            "when are visitors allowed"
        ],
        "answer": "Visitor timings are from 5 PM to 8 PM daily."
    },

    {
        "questions": [
            "when is the cardiologist available",
            "cardiology timings",
            "heart specialist timings",
            "what time does the cardiologist come",
            "cardiology department hours"
        ],
        "answer": "The cardiology specialist is available Monday to Friday from 9 AM to 1 PM."
    },

    {
        "questions": [
            "what are the orthopedic timings",
            "bone specialist timings",
            "orthopedic doctor availability",
            "when is the orthopedic doctor available",
            "orthopedic department timings"
        ],
        "answer": "The orthopedic specialist is available on weekdays from 2 PM to 6 PM."
    },

    {
        "questions": [
            "what are the pediatrician timings",
            "children doctor timings",
            "pediatrics clinic hours",
            "when is the pediatrician available",
            "kids doctor availability"
        ],
        "answer": "The Pediatrics clinic operates daily from 10 AM to 8 PM including Saturdays."
    },

    {
        "questions": [
            "what is the opd fee",
            "general consultation charges",
            "general opd charges",
            "doctor consultation fee",
            "how much is general consultation"
        ],
        "answer": "The general OPD consultation fee is 1500 Pakistani Rupees."
    },

    {
        "questions": [
            "what are the cardiology charges",
            "cardiologist fee",
            "heart specialist consultation fee",
            "how much does cardiology consultation cost",
            "cardiology consultation charges"
        ],
        "answer": "Cardiology consultation costs 3500 Pakistani Rupees per visit."
    },

    {
        "questions": [
            "what are the orthopedic consultation charges",
            "orthopedic fee",
            "bone specialist fee",
            "how much is orthopedic consultation",
            "orthopedic doctor charges"
        ],
        "answer": "Orthopedic consultation costs 3000 Pakistani Rupees per visit."
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
            "is the emergency open twenty four hours",
            "emergency department timings",
            "when is emergency open",
            "is emergency available at night",
            "emergency availability"
        ],
        "answer": "The Emergency Department operates twenty four hours every day."
    },

    {
        "questions": [
            "is the pharmacy open at night",
            "pharmacy timings",
            "hospital pharmacy hours",
            "when does the pharmacy close",
            "is the pharmacy open twenty four hours"
        ],
        "answer": "The hospital pharmacy remains open twenty four hours every day."
    },

    {
        "questions": [
            "what are the laboratory timings",
            "lab timings",
            "when is sample collection available",
            "pathology timings",
            "lab collection hours"
        ],
        "answer": "Laboratory sample collection is available daily from 7 AM to 10 PM."
    },

    {
        "questions": [
            "what are the mri timings",
            "mri availability",
            "when can i get an mri scan",
            "radiology timings",
            "mri department hours"
        ],
        "answer": "MRI services are available daily from 9 AM to 9 PM and require prior appointment."
    },

    {
        "questions": [
            "how can i book an appointment",
            "appointment booking process",
            "how do i schedule an appointment",
            "can i book through the kiosk",
            "ways to book doctor appointment"
        ],
        "answer": "Appointments can be booked through the reception desk, self service kiosk, website, or mobile application."
    },

    {
        "questions": [
            "do you accept walk in patients",
            "can i come without appointment",
            "walk in policy",
            "is appointment necessary",
            "can i visit directly"
        ],
        "answer": "Walk in patients are accepted depending on doctor availability and queue capacity."
    },

    {
        "questions": [
            "what documents are needed for admission",
            "admission requirements",
            "required documents for hospital admission",
            "what should i bring for admission",
            "patient admission process"
        ],
        "answer": "Patients must provide a valid CNIC, previous medical records, insurance information, and emergency contact details during admission."
    },

    {
        "questions": [
            "does the hospital accept insurance",
            "insurance providers",
            "which insurance companies are accepted",
            "can i use insurance here",
            "insurance coverage information"
        ],
        "answer": "The hospital accepts Jubilee Insurance, EFU Health, State Life, Adamjee Insurance, and selected corporate healthcare plans."
    },

    {
        "questions": [
            "what payment methods are accepted",
            "can i pay with card",
            "payment options",
            "do you accept digital wallets",
            "billing payment methods"
        ],
        "answer": "The hospital accepts cash, debit cards, credit cards, bank transfers, and selected digital wallet services."
    },

    {
        "questions": [
            "what are the vaccination center timings",
            "vaccination timings",
            "when is vaccination available",
            "vaccine center hours",
            "immunization timings"
        ],
        "answer": "The vaccination center operates Monday to Saturday from 9 AM to 5 PM."
    },

    {
        "questions": [
            "do you provide ambulance service",
            "ambulance availability",
            "can i request ambulance",
            "emergency transport service",
            "ambulance support"
        ],
        "answer": "The hospital provides ambulance services within city limits with trained emergency staff and medical support equipment."
    },

    {
        "questions": [
            "what are the dental clinic timings",
            "dentist timings",
            "dental department hours",
            "when is dental clinic open",
            "tooth doctor timings"
        ],
        "answer": "The dental clinic operates Monday to Saturday from 10 AM to 6 PM."
    },

    {
        "questions": [
            "what are the cafeteria timings",
            "hospital cafeteria hours",
            "when is cafeteria open",
            "food court timings",
            "canteen timings"
        ],
        "answer": "The hospital cafeteria operates daily from 7 AM until midnight."
    },

    {
        "questions": [
            "is parking available",
            "parking information",
            "where can i park",
            "visitor parking",
            "parking facility availability"
        ],
        "answer": "Parking facilities are available in Basement Levels 1 and 2 with hourly parking charges."
    },

    {
        "questions": [
            "do you provide wheelchair assistance",
            "wheelchair availability",
            "mobility support",
            "can i get a wheelchair",
            "patient wheelchair service"
        ],
        "answer": "Wheelchair assistance is available at all major entrances and reception areas free of charge."
    },

    {
        "questions": [
            "what are the dialysis timings",
            "dialysis availability",
            "kidney treatment timings",
            "dialysis department hours",
            "when is dialysis available"
        ],
        "answer": "The dialysis unit operates in multiple scheduled shifts throughout the week."
    },

    {
        "questions": [
            "do you offer telemedicine",
            "online doctor consultation",
            "virtual appointments",
            "remote consultation service",
            "video consultation availability"
        ],
        "answer": "Telemedicine consultations are available for selected departments through scheduled video appointments."
    },

    {
        "questions": [
            "is smoking allowed inside hospital",
            "hospital smoking policy",
            "can i smoke in hospital",
            "smoking rules",
            "designated smoking areas"
        ],
        "answer": "Smoking is strictly prohibited inside hospital buildings. Designated smoking areas are available outside the facility."
    },

    {
        "questions": [
            "what are the physiotherapy timings",
            "physiotherapy department hours",
            "rehabilitation services",
            "physical therapy availability",
            "physiotherapy appointments"
        ],
        "answer": "The physiotherapy department provides scheduled rehabilitation and therapy sessions throughout regular outpatient hours."
    },

    {
        "questions": [
            "do you have blood bank service",
            "blood bank timings",
            "blood donation availability",
            "blood transfusion support",
            "is blood bank open twenty four hours"
        ],
        "answer": "The blood bank operates twenty four hours for emergency transfusion support and blood donation services."
    },

    {
        "questions": [
            "how early should i arrive for appointment",
            "appointment arrival time",
            "when should i come before appointment",
            "early arrival policy",
            "registration timing before appointment"
        ],
        "answer": "Patients are advised to arrive at least fifteen minutes before their scheduled appointment."
    },

    {
        "questions": [
            "are masks required in hospital",
            "mask policy",
            "do i need to wear mask",
            "hospital covid policy",
            "mask requirement in icu"
        ],
        "answer": "Masks are mandatory in intensive care units and recommended throughout the hospital premises."
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
