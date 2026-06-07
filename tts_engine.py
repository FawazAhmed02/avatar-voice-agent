"""
tts_engine.py
-------------
Text-to-speech synthesis + viseme extraction.

Primary backend: piper-tts (ONNX-based, very low latency on Apple Silicon).
Fallback backend: scipy sine-wave tone (for testing without a TTS model).

Viseme events drive the avatar lip animation.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Viseme label set (matches standard Oculus / ReadSpeaker mapping)
# ---------------------------------------------------------------------------

VISEME_LABELS: List[str] = [
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR",
    "aa", "E", "ih", "oh", "ou",
]

# Simple phoneme → viseme mapping (ARPAbet subset)
_PHONEME_TO_VISEME: Dict[str, str] = {
    "SIL": "sil", "SP": "sil",
    "P": "PP", "B": "PP", "M": "PP",
    "F": "FF", "V": "FF",
    "TH": "TH", "DH": "TH",
    "T": "DD", "D": "DD", "N": "nn", "L": "DD",
    "K": "kk", "G": "kk", "NG": "kk",
    "CH": "CH", "JH": "CH", "SH": "SS", "ZH": "SS",
    "S": "SS", "Z": "SS",
    "R": "RR", "ER": "RR",
    "AA": "aa", "AO": "aa", "AH": "aa",
    "AE": "aa", "EH": "E", "EY": "E",
    "IH": "ih", "IY": "ih",
    "OW": "oh", "OY": "oh",
    "UW": "ou", "UH": "ou", "AW": "ou",
    "W": "ou", "Y": "ih", "HH": "sil",
}

# Rough character → phoneme heuristic (used when no phoneme data available)
_CHAR_TO_PHONEME: Dict[str, str] = {
    "a": "AA", "e": "EH", "i": "IH", "o": "OW", "u": "UW",
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
    "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
    " ": "SIL", ".": "SIL", ",": "SIL", "!": "SIL", "?": "SIL",
}


def _get_cfg():
    from config import get_config
    return get_config()


# ---------------------------------------------------------------------------
# Fallback sine-wave synthesiser (no piper needed)
# ---------------------------------------------------------------------------

def _synthesize_fallback(text: str, sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a simple sine-wave 'beep' as a stand-in for TTS.

    The duration scales with the length of *text* (approximately speech pace).
    """
    chars_per_second = 14.0  # approximate characters per second of speech
    duration_s = max(0.5, len(text.strip()) / chars_per_second)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    freq = 220.0  # Hz
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    logger.debug("Fallback TTS: generated %.2f s sine wave for %d chars.", duration_s, len(text))
    return audio


# ---------------------------------------------------------------------------
# Amplitude (RMS) extraction
# ---------------------------------------------------------------------------

def _extract_amplitude_events(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: int = 20,
) -> List[Dict]:
    """
    Compute RMS amplitude over fixed windows of *window_ms* milliseconds.

    Returns
    -------
    list[dict]
        Each entry: ``{"time_ms": float, "amplitude": float}``
    """
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    events: List[Dict] = []
    n = len(audio)
    for start in range(0, n, window_samples):
        window = audio[start : start + window_samples]
        rms = float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
        time_ms = (start / sample_rate) * 1000
        events.append({"time_ms": time_ms, "amplitude": round(rms, 5)})
    return events


# ---------------------------------------------------------------------------
# Viseme extraction
# ---------------------------------------------------------------------------

def _text_to_viseme_events(
    text: str,
    audio_duration_s: float,
    sample_rate: int,
    audio: np.ndarray,
    window_ms: int = 20,
) -> List[Dict]:
    """
    Map text characters → phonemes → visemes, aligned to audio duration,
    and augment each event with the RMS amplitude of the corresponding window.

    Parameters
    ----------
    text:
        The synthesised sentence.
    audio_duration_s:
        Total duration of synthesised audio in seconds.
    sample_rate:
        Sample rate of *audio*.
    audio:
        Raw float32 audio samples.
    window_ms:
        Duration (ms) of each amplitude window.

    Returns
    -------
    list[dict]
        ``[{"time_ms": float, "viseme": str, "amplitude": float}, …]``
    """
    amplitude_events = _extract_amplitude_events(audio, sample_rate, window_ms)
    amp_dict: Dict[float, float] = {e["time_ms"]: e["amplitude"] for e in amplitude_events}

    # Build character sequence (skip non-alpha whitespace variations)
    chars = [c.lower() for c in text if c.strip() or c == " "]
    if not chars:
        return [{"time_ms": 0.0, "viseme": "sil", "amplitude": 0.0}]

    # Uniform time spacing across characters
    interval_ms = (audio_duration_s * 1000.0) / max(len(chars), 1)
    events: List[Dict] = []

    for i, char in enumerate(chars):
        t_ms = round(i * interval_ms, 1)
        phoneme = _CHAR_TO_PHONEME.get(char, "SIL")
        viseme = _PHONEME_TO_VISEME.get(phoneme, "sil")

        # Find closest amplitude window
        closest_amp_time = min(amp_dict.keys(), key=lambda t: abs(t - t_ms), default=0.0)
        amplitude = amp_dict.get(closest_amp_time, 0.0)

        events.append({
            "time_ms": t_ms,
            "viseme": viseme,
            "amplitude": round(amplitude, 5),
        })

    # Append silence at end
    events.append({
        "time_ms": round(audio_duration_s * 1000, 1),
        "viseme": "sil",
        "amplitude": 0.0,
    })

    return events


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Synthesises speech from text and extracts lip-sync viseme events.

    Usage
    -----
    engine = TTSEngine()
    audio, visemes = engine.synthesize_chunk("Hello, welcome to WaveTec InfoPoint!")
    engine.play_audio(audio)
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

        self._tts_cfg = cfg.tts
        self._avatar_cfg = cfg.avatar
        self._model_path = Path(cfg.tts.model_path)
        self._sample_rate = cfg.tts.sample_rate
        self._speaking_rate = cfg.tts.speaking_rate
        self._volume = cfg.tts.volume
        self._amplitude_window_ms = cfg.avatar.amplitude_window_ms

        self._piper_voice = None       # piper.PiperVoice or None
        self._playback_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_piper(self) -> bool:
        """
        Attempt to load the piper TTS model.

        Returns True on success, False if piper is unavailable or the model
        file is missing.
        """
        if self._piper_voice is not None:
            return True

        if not self._model_path.exists():
            logger.warning(
                "Piper TTS model not found at '%s'. "
                "Download from https://huggingface.co/rhasspy/piper-voices. "
                "Falling back to sine-wave synthesiser.",
                self._model_path,
            )
            return False

        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError:
            logger.warning(
                "piper-tts is not installed (pip install piper-tts). "
                "Falling back to sine-wave synthesiser."
            )
            return False

        try:
            logger.info("Loading Piper TTS model: %s", self._model_path)
            t0 = time.perf_counter()
            self._piper_voice = PiperVoice.load(
                str(self._model_path),
                config_path=str(self._model_path) + ".json",
                use_cuda=False,
            )
            # Use the sample rate reported by the model, not config
            self._sample_rate = self._piper_voice.config.sample_rate
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("Piper TTS loaded in %.0f ms (sample_rate=%d).", elapsed_ms, self._sample_rate)
            return True
        except Exception as exc:
            logger.error("Failed to load Piper model: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def _synthesize_piper(self, text: str) -> np.ndarray:
        """Synthesise using piper and return float32 audio."""
        chunks = []
        for chunk in self._piper_voice.synthesize(text):
            chunks.append(chunk.audio_float_array)

        if not chunks:
            return np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)

        raw = np.concatenate(chunks).astype(np.float32)
        raw *= self._volume
        np.clip(raw, -1.0, 1.0, out=raw)
        return raw

    def synthesize_chunk(self, text: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Synthesise speech for *text* and return audio + viseme events.

        Parameters
        ----------
        text:
            Input text to synthesise.  Should be a sentence or short phrase.

        Returns
        -------
        tuple[np.ndarray, list[dict]]
            - audio_samples: float32 array at ``tts.sample_rate`` Hz.
            - viseme_events: list of ``{"time_ms", "viseme", "amplitude"}`` dicts.
        """
        if not text or not text.strip():
            logger.debug("synthesize_chunk called with empty text.")
            silence = np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)
            return silence, [{"time_ms": 0.0, "viseme": "sil", "amplitude": 0.0}]

        t0 = time.perf_counter()
        piper_ok = self._init_piper()

        if piper_ok:
            try:
                audio = self._synthesize_piper(text)
            except Exception as exc:
                logger.error("Piper synthesis failed: %s. Using fallback.", exc)
                audio = _synthesize_fallback(text, self._sample_rate)
        else:
            audio = _synthesize_fallback(text, self._sample_rate)

        duration_s = len(audio) / self._sample_rate
        visemes = self.extract_visemes(audio, text)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "TTS: synthesised '%s…' → %.2f s audio + %d viseme events (%.0f ms).",
            text[:40],
            duration_s,
            len(visemes),
            elapsed_ms,
        )
        return audio, visemes

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_audio(self, samples: np.ndarray, done_callback=None) -> None:
        """Play audio non-blocking. Calls done_callback when playback finishes."""
        if samples is None or len(samples) == 0:
            if done_callback:
                done_callback()
            return

        def _play():
            try:
                import sounddevice as sd
                with self._playback_lock:
                    sd.play(samples, samplerate=self._sample_rate, blocking=True)
            except Exception as exc:
                logger.error("Audio playback error: %s", exc)
            finally:
                if done_callback:
                    done_callback()

        threading.Thread(target=_play, daemon=True, name="tts-playback").start()

    # ------------------------------------------------------------------
    # Viseme extraction (public)
    # ------------------------------------------------------------------

    def extract_visemes(self, audio: np.ndarray, text: str) -> List[Dict]:
        """
        Extract a list of viseme events from synthesised audio and text.

        Uses a character-to-phoneme heuristic aligned to audio duration,
        augmented with per-window RMS amplitude measurements.

        Parameters
        ----------
        audio:
            float32 audio samples at ``tts.sample_rate``.
        text:
            The text that was synthesised into *audio*.

        Returns
        -------
        list[dict]
            Each entry: ``{"time_ms": float, "viseme": str, "amplitude": float}``
        """
        duration_s = len(audio) / self._sample_rate if len(audio) > 0 else 0.1
        return _text_to_viseme_events(
            text,
            audio_duration_s=duration_s,
            sample_rate=self._sample_rate,
            audio=audio,
            window_ms=self._amplitude_window_ms,
        )
