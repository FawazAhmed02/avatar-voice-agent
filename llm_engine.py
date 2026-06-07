"""
llm_engine.py
-------------
Local LLM inference using mlx-lm (Apple Silicon Metal Performance Shaders).

Exposes ``stream_response()`` which streams word-chunk tokens and logs
time-to-first-token (TTFT).
"""

from __future__ import annotations

import logging
import time
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "Answer in 1-2 sentences from context only."

# Hard cap on RAG context characters injected into the prompt.
# Fewer input tokens = dramatically lower TTFT (~7ms per token on M4).
_MAX_CONTEXT_CHARS = 300

# Module-level lazy references
_model = None
_tokenizer = None
_loaded_model_path: Optional[str] = None


def _get_cfg():
    from config import get_config
    return get_config()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_model_loaded() -> None:
    """Load the mlx-lm model and tokenizer if not already loaded."""
    global _model, _tokenizer, _loaded_model_path

    cfg = _get_cfg()
    desired = cfg.llm.model

    if _model is not None and _loaded_model_path == desired:
        return  # already loaded

    logger.info("Loading LLM model: %s", desired)
    t0 = time.perf_counter()

    try:
        from mlx_lm import load  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mlx_lm is not installed. Run: pip install mlx-lm"
        ) from exc

    try:
        _model, _tokenizer = load(desired)
        _loaded_model_path = desired
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("LLM model loaded in %.0f ms.", elapsed_ms)
    except Exception as exc:
        logger.error("Failed to load LLM model '%s': %s", desired, exc, exc_info=True)
        _model = None
        _tokenizer = None
        raise


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(user_query: str, context_chunks: List[str]) -> str:
    """
    Build the chat prompt using the tokenizer's apply_chat_template so the
    correct format is used automatically regardless of which model is loaded.
    RAG context is hard-capped at _MAX_CONTEXT_CHARS to keep TTFT low.
    """
    _ensure_model_loaded()

    # Truncate context to stay within token budget
    if not context_chunks or context_chunks == ["No relevant context found."]:
        context_str = ""
    else:
        seen: set = set()
        budget = _MAX_CONTEXT_CHARS
        parts: List[str] = []
        for c in context_chunks:
            if c in seen:
                continue
            seen.add(c)
            snippet = c[:budget].strip()
            parts.append(snippet)
            budget -= len(snippet)
            if budget <= 0:
                break
        context_str = " ".join(parts)

    user_msg = f"Context: {context_str}\nQ: {user_query}" if context_str else user_query

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        prompt = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers that don't support apply_chat_template
        prompt = f"{_SYSTEM_PROMPT}\n\nUser: {user_msg}\nAssistant:"

    return prompt


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------

def stream_response(
    prompt: str,
    context_chunks: List[str],
) -> Generator[str, None, None]:
    """
    Generate a streaming LLM response, yielding word-group chunks.

    Parameters
    ----------
    prompt:
        The user's question / transcribed speech.
    context_chunks:
        RAG-retrieved text chunks used as grounding context.

    Yields
    ------
    str
        A short string containing ``stream_chunk_words`` words (configurable),
        or the final partial chunk when the generation completes.

    Raises
    ------
    RuntimeError
        If the model could not be loaded.
    """
    _ensure_model_loaded()
    cfg = _get_cfg()

    full_prompt = _build_prompt(prompt, context_chunks)
    logger.debug("LLM prompt (first 200 chars): %s", full_prompt[:200])

    try:
        from mlx_lm import stream_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mlx_lm not available. Run: pip install --upgrade mlx-lm"
        ) from exc

    sampler = make_sampler(temp=cfg.llm.temperature, top_p=cfg.llm.top_p)

    word_buffer: List[str] = []
    chunk_word_target = cfg.llm.stream_chunk_words
    first_token = True
    t_start = time.perf_counter()
    ttft_ms: Optional[float] = None
    total_tokens = 0

    try:
        for response in stream_generate(
            _model,
            _tokenizer,
            prompt=full_prompt,
            max_tokens=cfg.llm.max_tokens,
            sampler=sampler,
        ):
            token_text = response.text
            if first_token:
                ttft_ms = (time.perf_counter() - t_start) * 1000
                logger.info("LLM TTFT: %.0f ms (target: %d ms)", ttft_ms, cfg.llm.ttft_target_ms)
                first_token = False

            total_tokens += 1

            # Accumulate tokens into words (tokens can be sub-word)
            word_buffer.append(token_text)

            # Count approximate words by splitting joined buffer on spaces
            joined = "".join(word_buffer)
            words = joined.split()

            if len(words) >= chunk_word_target:
                # Yield all but potentially the last partial word
                # to avoid cutting mid-word
                to_yield = " ".join(words[:chunk_word_target])
                remainder = " ".join(words[chunk_word_target:])
                yield to_yield + " "
                word_buffer = [remainder] if remainder else []

        # Yield any remaining content
        if word_buffer:
            remaining = "".join(word_buffer).strip()
            if remaining:
                yield remaining

    except Exception as exc:
        logger.error("LLM generation error: %s", exc, exc_info=True)
        yield "[Error generating response]"
        return

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "LLM complete: %d tokens, TTFT=%.0f ms, total=%.0f ms (%.1f tok/s).",
        total_tokens,
        ttft_ms or 0.0,
        total_ms,
        total_tokens / max(total_ms / 1000, 0.001),
    )


# ---------------------------------------------------------------------------
# Async wrapper (non-blocking for the asyncio pipeline)
# ---------------------------------------------------------------------------

async def astream_response(
    prompt: str,
    context_chunks: List[str],
) -> Generator[str, None, None]:
    """
    Async-compatible thin wrapper around ``stream_response``.

    Because mlx-lm generation is CPU/GPU-bound, we run it in a thread
    executor to avoid blocking the event loop.  The caller should
    iterate the returned generator inside ``asyncio.to_thread``.

    This function itself is a coroutine that returns the generator;
    the generator should be consumed in a thread via ``asyncio.to_thread``.
    """
    import asyncio

    loop = asyncio.get_running_loop()

    def _collect_all() -> List[str]:
        return list(stream_response(prompt, context_chunks))

    chunks = await loop.run_in_executor(None, _collect_all)
    return iter(chunks)
