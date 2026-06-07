"""
config.py
---------
Loads config.yaml and exposes a typed dataclass singleton CFG.
Supports environment variable overrides for key fields.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Typed dataclasses mirroring the YAML structure
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    threshold: float = 0.7
    silence_duration_ms: int = 280
    sample_rate: int = 16000
    chunk_size: int = 512


@dataclass
class STTConfig:
    model: str = "mlx-community/whisper-small.en-mlx"
    language: str = "en"


@dataclass
class RAGConfig:
    docs_dir: str = "./docs"
    index_path: str = "./rag_index.faiss"
    metadata_path: str = "./rag_metadata.pkl"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 2


@dataclass
class LLMConfig:
    model: str = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream_chunk_words: int = 5
    ttft_target_ms: int = 100


@dataclass
class TTSConfig:
    model_path: str = "./tts_models/en_US-lessac-medium.onnx"
    sample_rate: int = 22050
    speaking_rate: float = 1.0
    volume: float = 1.0


@dataclass
class AvatarConfig:
    viseme_smoothing: float = 0.3
    amplitude_window_ms: int = 20
    output_mode: str = "stdout"
    ws_port: int = 8765


@dataclass
class SystemConfig:
    device: str = "mlx"
    log_level: str = "INFO"
    pipeline_timeout_s: int = 10


@dataclass
class AppConfig:
    """Top-level application configuration."""

    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s", path)
    return data


def _apply_env_overrides(raw: dict) -> dict:
    """
    Apply environment variable overrides to the raw config dict.

    Supported variables
    -------------------
    KIOSK_LLM_MODEL   -> llm.model
    KIOSK_TTS_MODEL   -> tts.model_path
    KIOSK_DOCS_DIR    -> rag.docs_dir
    """
    overrides: dict = {}

    llm_model = os.environ.get("KIOSK_LLM_MODEL")
    if llm_model:
        overrides.setdefault("llm", {})["model"] = llm_model
        logger.info("Env override: llm.model = %s", llm_model)

    tts_model = os.environ.get("KIOSK_TTS_MODEL")
    if tts_model:
        overrides.setdefault("tts", {})["model_path"] = tts_model
        logger.info("Env override: tts.model_path = %s", tts_model)

    docs_dir = os.environ.get("KIOSK_DOCS_DIR")
    if docs_dir:
        overrides.setdefault("rag", {})["docs_dir"] = docs_dir
        logger.info("Env override: rag.docs_dir = %s", docs_dir)

    if overrides:
        raw = _deep_merge(raw, overrides)
    return raw


def _build_config(raw: dict) -> AppConfig:
    """Instantiate AppConfig from a (possibly partial) raw dict."""

    def _section(key: str) -> dict:
        return raw.get(key) or {}

    vad_raw = _section("vad")
    stt_raw = _section("stt")
    rag_raw = _section("rag")
    llm_raw = _section("llm")
    tts_raw = _section("tts")
    avatar_raw = _section("avatar")
    system_raw = _section("system")

    return AppConfig(
        vad=VADConfig(
            threshold=float(vad_raw.get("threshold", VADConfig.threshold)),
            silence_duration_ms=int(vad_raw.get("silence_duration_ms", VADConfig.silence_duration_ms)),
            sample_rate=int(vad_raw.get("sample_rate", VADConfig.sample_rate)),
            chunk_size=int(vad_raw.get("chunk_size", VADConfig.chunk_size)),
        ),
        stt=STTConfig(
            model=str(stt_raw.get("model", STTConfig.model)),
            language=str(stt_raw.get("language", STTConfig.language)),
        ),
        rag=RAGConfig(
            docs_dir=str(rag_raw.get("docs_dir", RAGConfig.docs_dir)),
            index_path=str(rag_raw.get("index_path", RAGConfig.index_path)),
            metadata_path=str(rag_raw.get("metadata_path", RAGConfig.metadata_path)),
            embedding_model=str(rag_raw.get("embedding_model", RAGConfig.embedding_model)),
            chunk_size=int(rag_raw.get("chunk_size", RAGConfig.chunk_size)),
            chunk_overlap=int(rag_raw.get("chunk_overlap", RAGConfig.chunk_overlap)),
            top_k=int(rag_raw.get("top_k", RAGConfig.top_k)),
        ),
        llm=LLMConfig(
            model=str(llm_raw.get("model", LLMConfig.model)),
            max_tokens=int(llm_raw.get("max_tokens", LLMConfig.max_tokens)),
            temperature=float(llm_raw.get("temperature", LLMConfig.temperature)),
            top_p=float(llm_raw.get("top_p", LLMConfig.top_p)),
            stream_chunk_words=int(llm_raw.get("stream_chunk_words", LLMConfig.stream_chunk_words)),
            ttft_target_ms=int(llm_raw.get("ttft_target_ms", LLMConfig.ttft_target_ms)),
        ),
        tts=TTSConfig(
            model_path=str(tts_raw.get("model_path", TTSConfig.model_path)),
            sample_rate=int(tts_raw.get("sample_rate", TTSConfig.sample_rate)),
            speaking_rate=float(tts_raw.get("speaking_rate", TTSConfig.speaking_rate)),
            volume=float(tts_raw.get("volume", TTSConfig.volume)),
        ),
        avatar=AvatarConfig(
            viseme_smoothing=float(avatar_raw.get("viseme_smoothing", AvatarConfig.viseme_smoothing)),
            amplitude_window_ms=int(avatar_raw.get("amplitude_window_ms", AvatarConfig.amplitude_window_ms)),
            output_mode=str(avatar_raw.get("output_mode", AvatarConfig.output_mode)),
            ws_port=int(avatar_raw.get("ws_port", AvatarConfig.ws_port)),
        ),
        system=SystemConfig(
            device=str(system_raw.get("device", SystemConfig.device)),
            log_level=str(system_raw.get("log_level", SystemConfig.log_level)),
            pipeline_timeout_s=int(system_raw.get("pipeline_timeout_s", SystemConfig.pipeline_timeout_s)),
        ),
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_CFG: Optional[AppConfig] = None


# Third-party loggers that are noisy and should stay at WARNING
_SILENT_LOGGERS = [
    "huggingface_hub",
    "huggingface_hub.utils",
    "transformers",
    "tokenizers",
    "sentence_transformers",
    "faiss",
    "numba",
    "urllib3",
    "filelock",
    "httpx",
    "httpcore",
    "tqdm",
    "PIL",
    "torch",
    "mlx",
    "mlx_whisper",
    "mlx_lm",
    "asyncio",
]

# Our own modules — controlled by config log_level
_APP_LOGGERS = [
    "audio_input",
    "stt_engine",
    "rag_storage",
    "llm_engine",
    "tts_engine",
    "__main__",
    "config",
    "main",
]


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the entire application.

    - Routes all app logs through RichHandler for clean, readable output.
    - Silences noisy third-party libraries.
    - Should be called once at startup before any other imports log.
    """
    try:
        from rich.logging import RichHandler
        from rich.console import Console as _Console
        _handler = RichHandler(
            console=_Console(stderr=True),
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        _handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger: WARNING so third-party noise is suppressed by default
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_handler)
    root.setLevel(logging.WARNING)

    # Explicitly silence known noisy loggers
    for name in _SILENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Our app loggers at the configured level
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(numeric_level)


def get_config(reload: bool = False) -> AppConfig:
    """
    Return the global AppConfig singleton.

    Parameters
    ----------
    reload:
        If True, force re-read of the YAML file (useful for testing).
    """
    global _CFG
    if _CFG is None or reload:
        raw = _load_yaml(_CONFIG_FILE)
        raw = _apply_env_overrides(raw)
        _CFG = _build_config(raw)
        setup_logging(_CFG.system.log_level)
        logger.debug("AppConfig loaded.")
    return _CFG


# Module-level singleton – imported directly by other modules
CFG: AppConfig = get_config()
