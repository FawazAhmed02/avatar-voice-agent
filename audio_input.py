"""
audio_input.py
--------------
Microphone capture with Silero VAD.

Provides the VADStream class, which asynchronously yields complete speech
phrases as numpy float32 arrays sampled at 16 kHz.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from collections import deque
from typing import AsyncGenerator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – these are large dependencies; import only when first used
# ---------------------------------------------------------------------------

def _import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is not installed. Run: pip install sounddevice"
        ) from exc


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "torch is not installed. Run: pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# Viseme / amplitude constants (shared with tts_engine)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz – VAD operates at 16 kHz


# ---------------------------------------------------------------------------
# Silero VAD loader
# ---------------------------------------------------------------------------

_vad_model = None
_vad_utils = None


def _load_silero_vad():
    """Load the Silero VAD model from the local silero-vad package (no internet required)."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils

    logger.info("Loading Silero VAD model…")
    try:
        from silero_vad import load_silero_vad  # bundled with the package, fully offline
        model = load_silero_vad()
        _vad_model = model
        _vad_utils = None
        logger.info("Silero VAD model loaded.")
        return model, None
    except Exception as exc:
        logger.error("Failed to load Silero VAD: %s", exc)
        raise


# ---------------------------------------------------------------------------
# VADStream
# ---------------------------------------------------------------------------

class VADStream:
    """
    Captures microphone audio and uses Silero VAD to segment speech phrases.

    Usage
    -----
    stream = VADStream(cfg)
    stream.start()
    async for phrase_audio in stream.iter_phrases():
        # phrase_audio is np.ndarray, float32, 16 kHz mono
        process(phrase_audio)
    stream.stop()
    """

    def __init__(self, cfg=None):
        """
        Parameters
        ----------
        cfg:
            AppConfig instance. If None, the global CFG singleton is used.
        """
        if cfg is None:
            from config import get_config
            cfg = get_config()

        self._vad_cfg = cfg.vad
        self._threshold: float = cfg.vad.threshold
        self._silence_frames: int = int(
            cfg.vad.silence_duration_ms * cfg.vad.sample_rate / 1000
        )
        self._sample_rate: int = cfg.vad.sample_rate
        self._chunk_size: int = cfg.vad.chunk_size

        self._raw_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._phrase_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)

        self._stream = None
        self._running = False
        self._vad_model = None
        self._vad_utils = None

        # Internal state for phrase accumulation
        self._ring_buffer: deque[np.ndarray] = deque()
        self._speech_buffer: list[np.ndarray] = []
        self._silence_counter: int = 0
        self._in_speech: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone stream and start capturing audio."""
        sd = _import_sounddevice()

        self._vad_model, self._vad_utils = _load_silero_vad()
        self._running = True

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_size,
                callback=self._sd_callback,
            )
            self._stream.start()
            logger.info(
                "Microphone stream started (rate=%d Hz, chunk=%d frames).",
                self._sample_rate,
                self._chunk_size,
            )
        except sd.PortAudioError as exc:
            self._running = False
            if "Invalid device" in str(exc) or "No Default Input" in str(exc):
                raise OSError(
                    "No audio input device found. Check microphone permissions and connections."
                ) from exc
            raise

    def stop(self) -> None:
        """Stop capturing and close the microphone stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error closing audio stream: %s", exc)
            self._stream = None
        # Signal the async generator to exit
        self._raw_queue.put_nowait(None)
        logger.info("Microphone stream stopped.")

    # ------------------------------------------------------------------
    # sounddevice callback (runs in a separate OS thread)
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            logger.debug("sounddevice status: %s", status)
        if not self._running:
            return
        chunk = indata[:, 0].copy()  # mono, float32
        try:
            self._raw_queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Audio input queue full – dropping chunk.")

    # ------------------------------------------------------------------
    # VAD processing (runs inside the async generator)
    # ------------------------------------------------------------------

    def _run_vad(self, chunk: np.ndarray) -> float:
        """Run Silero VAD on a single chunk, returning speech probability."""
        torch = _import_torch()
        tensor = torch.from_numpy(chunk).unsqueeze(0)  # shape [1, T]
        with torch.no_grad():
            prob = self._vad_model(tensor, self._sample_rate).item()
        return float(prob)

    def _process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed a chunk through VAD state machine.

        Returns a completed phrase (np.ndarray) when one is detected,
        otherwise returns None.
        """
        speech_prob = self._run_vad(chunk)
        is_speech = speech_prob >= self._threshold

        t_ms = time.time() * 1000
        if is_speech:
            if not self._in_speech:
                logger.debug("[VAD] Speech START  @ %.0f ms (prob=%.2f)", t_ms, speech_prob)
            self._in_speech = True
            self._silence_counter = 0
            self._speech_buffer.append(chunk)
        else:
            if self._in_speech:
                self._silence_counter += len(chunk)
                self._speech_buffer.append(chunk)  # include trailing silence

                if self._silence_counter >= self._silence_frames:
                    phrase = np.concatenate(self._speech_buffer, axis=0)
                    logger.debug(
                        "[VAD] Speech END    @ %.0f ms (phrase=%.2fs, prob=%.2f)",
                        t_ms,
                        len(phrase) / self._sample_rate,
                        speech_prob,
                    )
                    self._speech_buffer.clear()
                    self._silence_counter = 0
                    self._in_speech = False
                    return phrase
        return None

    # ------------------------------------------------------------------
    # Async generator
    # ------------------------------------------------------------------

    async def iter_phrases(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator that yields complete speech phrases.

        Each yielded value is a float32 numpy array at 16 kHz containing
        one phrase of speech (from first voiced frame to end of silence).

        Raises
        ------
        RuntimeError
            If ``start()`` has not been called before iterating.
        """
        if not self._running:
            raise RuntimeError("VADStream.start() must be called before iterating.")

        loop = asyncio.get_running_loop()

        while self._running:
            # Non-blocking poll – yield control to the event loop between polls
            try:
                chunk = self._raw_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)  # 5 ms back-off
                continue

            if chunk is None:  # sentinel – stream was stopped
                break

            # Run VAD (CPU-bound) in a thread pool so we don't block asyncio
            phrase = await loop.run_in_executor(None, self._process_chunk, chunk)
            if phrase is not None:
                # Filter extremely short clips (< 0.3 s) – likely noise
                if len(phrase) >= int(0.3 * self._sample_rate):
                    yield phrase
                else:
                    logger.debug("[VAD] Discarding short phrase (%.2f s).", len(phrase) / self._sample_rate)
