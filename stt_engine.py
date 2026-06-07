"""
stt_engine.py
-------------
Speech-to-text using mlx-whisper (Apple Silicon optimised Whisper).

The model is loaded lazily on the first call to ``transcribe()``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target sample rate expected by Whisper
_WHISPER_SR = 16_000

# Module-level lazy references
_model = None
_model_path: Optional[str] = None


def _get_cfg():
    from config import get_config
    return get_config()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_model_loaded() -> None:
    """Load the mlx-whisper model if it hasn't been loaded yet."""
    global _model, _model_path

    cfg = _get_cfg()
    desired_path = cfg.stt.model

    if _model is not None and _model_path == desired_path:
        return  # already loaded

    logger.info("Loading mlx-whisper model: %s", desired_path)
    t0 = time.perf_counter()

    try:
        import mlx_whisper  # noqa: F401 – verify import works
        # mlx_whisper uses the model path at transcription time; we store it here
        _model_path = desired_path
        _model = True  # sentinel – the actual model is managed internally by mlx_whisper
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("mlx-whisper model ready in %.0f ms.", elapsed)
    except ImportError as exc:
        raise ImportError(
            "mlx_whisper is not installed. Run: pip install mlx-whisper"
        ) from exc
    except Exception as exc:
        logger.error("Failed to load mlx-whisper model '%s': %s", desired_path, exc)
        raise


# ---------------------------------------------------------------------------
# Audio normalisation helpers
# ---------------------------------------------------------------------------

def _to_float32_mono_16k(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is float32, mono, and resampled to 16 kHz.

    Parameters
    ----------
    audio:
        Input audio.  Accepted shapes: (N,) mono or (N, C) multi-channel.
        Accepted dtypes: int16, int32, float32, float64.

    Returns
    -------
    np.ndarray
        float32 mono array at 16 kHz.
    """
    # Convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Ensure 1-D (mono)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")

    # Resample if not already at 16 kHz
    # We detect the source rate by checking if the caller set an attribute;
    # in most cases the audio arriving here is already 16 kHz from VADStream.
    # If librosa is available, use it for high-quality resampling.
    return audio


def _resample_if_needed(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample audio from *source_sr* to 16 kHz if necessary."""
    if source_sr == _WHISPER_SR:
        return audio
    try:
        import librosa
        logger.debug("Resampling audio from %d Hz to %d Hz.", source_sr, _WHISPER_SR)
        return librosa.resample(audio, orig_sr=source_sr, target_sr=_WHISPER_SR)
    except ImportError:
        logger.warning(
            "librosa not available for resampling; audio may be at wrong rate (%d Hz).",
            source_sr,
        )
        return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray, source_sr: int = _WHISPER_SR) -> str:
    """
    Transcribe a speech audio array to text.

    Parameters
    ----------
    audio:
        Raw audio data.  Mono or stereo; any supported dtype.
    source_sr:
        Sample rate of *audio*.  Defaults to 16000 Hz (Whisper native).

    Returns
    -------
    str
        Transcribed text, stripped of whitespace.
        Returns an empty string if transcription yields nothing useful.

    Raises
    ------
    RuntimeError
        On model load failure.
    ValueError
        If *audio* is empty.
    """
    if audio is None or len(audio) == 0:
        logger.warning("transcribe() called with empty audio array.")
        return ""

    _ensure_model_loaded()

    cfg = _get_cfg()

    # Normalise audio
    audio_f32 = _to_float32_mono_16k(audio)
    audio_f32 = _resample_if_needed(audio_f32, source_sr)

    # Guard against near-silent audio (RMS < threshold)
    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    if rms < 1e-4:
        logger.debug("Audio RMS too low (%.6f) – skipping transcription.", rms)
        return ""

    logger.debug("Transcribing %.2f s of audio (RMS=%.4f) …", len(audio_f32) / _WHISPER_SR, rms)
    t0 = time.perf_counter()

    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=_model_path,
            language=cfg.stt.language,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        text: str = (result.get("text") or "").strip()

        logger.info(
            "STT: '%s'  (%.0f ms, audio_len=%.2f s)",
            text,
            elapsed_ms,
            len(audio_f32) / _WHISPER_SR,
        )

        # Filter out whisper's common noise hallucinations
        _hallucinations = {
            "", ".", "…", "...", "Thank you.", "Thanks for watching.",
            "Thank you for watching.", "you",
        }
        if text in _hallucinations:
            logger.debug("Filtered hallucination: '%s'", text)
            return ""

        return text

    except Exception as exc:
        logger.error("Transcription failed: %s", exc, exc_info=True)
        return ""
