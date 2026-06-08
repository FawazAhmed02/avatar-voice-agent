"""
main.py
-------
Central async orchestrator for the WaveTec InfoPoint kiosk voice pipeline.

Pipeline stages (each in its own asyncio task):
  VAD → STT → RAG → LLM Stream → TTS Stream → Avatar Output

CLI entry point (click):
  python main.py run     — start kiosk
  python main.py ingest  — ingest docs then exit
  python main.py status  — print system/model info
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import queue as sync_queue
import threading
import webbrowser
from pathlib import Path as _Path

try:
    import websockets
    import websockets.server
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import get_config, setup_logging

# Initialise logging before any other module logs anything
setup_logging("INFO")

logger = logging.getLogger(__name__)

# stdout console for user-facing output (You / Assistant lines)
console = Console()
# stderr console for the Live status panel — keeps it separate from log output
_panel_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Pipeline stage names (used for display)
# ---------------------------------------------------------------------------

STAGES = ["VAD", "STT", "RAG", "LLM", "TTS", "Avatar"]

_SENT_DELIMS = (". ", "! ", "? ", ".\n", "!\n", "?\n")


# ---------------------------------------------------------------------------
# Pipeline state (shared across tasks via asyncio.Queue)
# ---------------------------------------------------------------------------

class PipelineState:
    """Holds inter-stage queues and global flags."""

    def __init__(self):
        self.audio_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=4)
        self.stt_q: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.rag_q: asyncio.Queue[Dict] = asyncio.Queue(maxsize=4)
        self.llm_q: asyncio.Queue[Dict] = asyncio.Queue(maxsize=4)
        self.tts_q: asyncio.Queue[Dict] = asyncio.Queue(maxsize=4)

        self.active_stage: str = "idle"
        self.latencies: Dict[str, float] = {}
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.pipeline_busy: threading.Event = threading.Event()
        self.ws_clients: set = set()

    async def ws_broadcast(self, msg: dict) -> None:
        if not self.ws_clients:
            return
        data = json.dumps(msg)
        dead = set()
        for ws in list(self.ws_clients):
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self.ws_clients -= dead


async def _ws_handler(websocket, state: PipelineState) -> None:
    state.ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        state.ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Model pre-warming
# ---------------------------------------------------------------------------

async def _prewarm_models() -> None:
    """
    Load all heavy models before the VAD loop starts so the first user
    utterance hits warm inference paths with no cold-start penalty.
    """
    cfg = get_config()
    console.print("[cyan]Pre-warming models…[/cyan]")
    t0 = time.perf_counter()

    # Force a complete import of transformers in the main thread before concurrent
    # pre-warms start. sentence-transformers and mlx-lm both import transformers
    # simultaneously; without this, one thread sees a partially-initialised module
    # and fails to find AutoTokenizer.
    await asyncio.to_thread(__import__, "transformers")

    async def _load(label: str, fn):
        t = time.perf_counter()
        try:
            await asyncio.to_thread(fn)
            logger.info("Warmed %-20s %.0f ms", label, (time.perf_counter() - t) * 1000)
        except Exception as exc:
            logger.warning("Pre-warm failed for %s: %s", label, exc, exc_info=True)

    def _warm_stt():
        import mlx_whisper, numpy as np
        mlx_whisper.transcribe(
            np.zeros(3200, dtype=np.float32),
            path_or_hf_repo=cfg.stt.model,
            language=cfg.stt.language,
            verbose=False,
        )

    def _warm_embed():
        from rag_storage import RAGEngine
        engine = RAGEngine(cfg)
        engine._get_embedder()
        engine._embed(["warmup"])

    def _warm_llm():
        import llm_engine
        llm_engine._ensure_model_loaded()

    def _warm_tts():
        from tts_engine import TTSEngine
        TTSEngine(cfg)._init_piper()

    def _warm_cache():
        from response_cache import get_cache
        get_cache().build()

    await asyncio.gather(
        _load("STT (whisper-tiny)", _warm_stt),
        _load("Embedder (MiniLM)", _warm_embed),
        _load("LLM (Qwen2.5-1.5B)", _warm_llm),
        _load("TTS (Piper)", _warm_tts),
        _load("Response cache", _warm_cache),
    )

    console.print(f"[green]All models ready in {(time.perf_counter()-t0)*1000:.0f} ms — listening.[/green]")


# ---------------------------------------------------------------------------
# Conversational filler detection
# ---------------------------------------------------------------------------

# Words that can appear anywhere in a short acknowledgment utterance without
# making it a real question.
_ACK_WORDS = {
    "ok", "okay", "alright", "right", "perfect", "great", "nice", "wonderful",
    "awesome", "cool", "good", "fine", "noted", "understood", "got", "it",
    "sounds", "i", "see", "much", "very", "so", "a", "lot", "many",
}

# Terminal phrases that, when they close the utterance, signal a filler.
_THANKS_PHRASES  = {"thank you", "thanks", "thank you so much", "thanks a lot",
                    "many thanks", "thank you very much", "cheers", "much appreciated"}
_BYE_PHRASES     = {"bye", "goodbye", "see you", "see ya", "take care",
                    "have a good day", "have a nice day", "good day", "farewell"}
_HELLO_PHRASES   = {"hello", "hi", "hey", "good morning", "good afternoon",
                    "good evening", "hi there", "hey there", "howdy"}
_NEUTRAL_PHRASES = {"ok", "okay", "alright", "got it", "i see", "understood",
                    "noted", "sounds good", "perfect", "great", "nice",
                    "wonderful", "awesome", "cool", "good", "fine"}

def _is_conversational_filler(text: str) -> str | None:
    """
    Return a canned reply if the utterance is pure conversational filler.

    Strategy: strip punctuation, check exact matches first, then check whether
    the utterance is short (≤8 words) and ends with a known terminal phrase
    (handles compounds like "okay perfect thank you" or "alright thanks").
    """
    # Strip all punctuation from each word so "okay," == "okay"
    import re as _re
    normalized = _re.sub(r"[^\w\s]", " ", text.lower()).split()
    normalized_str = " ".join(normalized)

    # Exact match against known terminal phrases
    for phrases, reply in (
        (_THANKS_PHRASES,  "You're welcome! Is there anything else I can help you with?"),
        (_BYE_PHRASES,     "Goodbye! Have a great day."),
        (_HELLO_PHRASES,   "Hello! How can I help you today?"),
        (_NEUTRAL_PHRASES, "Glad I could help. Feel free to ask if you have more questions."),
    ):
        if normalized_str in phrases:
            return reply

    # Short compound acknowledgment: ≤8 words, ends with a known terminal,
    # and every word belongs to the ack-word vocabulary (no real question).
    words = normalized
    if len(words) <= 8 and "?" not in text:
        # Check if it ends with a 1- or 2-word terminal phrase
        for terminal, reply in (
            (_THANKS_PHRASES,  "You're welcome! Is there anything else I can help you with?"),
            (_BYE_PHRASES,     "Goodbye! Have a great day."),
            (_HELLO_PHRASES,   "Hello! How can I help you today?"),
        ):
            for phrase in terminal:
                pwords = phrase.split()
                if words[-len(pwords):] == pwords:
                    # Make sure all preceding words are benign ack words
                    prefix_words = words[:-len(pwords)]
                    if all(w in _ACK_WORDS for w in prefix_words):
                        return reply

        # Pure ack words with no terminal — e.g. "okay perfect"
        if all(w in _ACK_WORDS for w in words):
            return "Glad I could help. Feel free to ask if you have more questions."

    return None


# ---------------------------------------------------------------------------
# Stage coroutines
# ---------------------------------------------------------------------------

async def vad_stage(state: PipelineState) -> None:
    """Capture microphone audio and push detected speech phrases."""
    from audio_input import VADStream

    cfg = get_config()
    stream = VADStream(cfg)

    try:
        stream.start()
    except OSError as exc:
        console.print(f"[bold red]Microphone error:[/bold red] {exc}")
        state.shutdown_event.set()
        return

    logger.info("VAD stage started. Listening …")
    console.print("[green]Listening for speech…[/green]")

    try:
        async for phrase_audio in stream.iter_phrases():
            if state.shutdown_event.is_set():
                break
            if state.pipeline_busy.is_set():
                logger.debug("VAD: phrase dropped — pipeline busy")
                continue
            state.pipeline_busy.set()
            t0 = time.perf_counter()
            state.active_stage = "VAD"
            await asyncio.wait_for(
                state.audio_q.put({"audio": phrase_audio, "t_vad": t0}),
                timeout=cfg.system.pipeline_timeout_s,
            )
    except asyncio.TimeoutError:
        logger.warning("VAD stage: queue put timed out.")
    except Exception as exc:
        logger.error("VAD stage error: %s", exc, exc_info=True)
    finally:
        stream.stop()
        logger.info("VAD stage stopped.")


async def stt_stage(state: PipelineState) -> None:
    """Transcribe audio phrases to text."""
    import stt_engine

    cfg = get_config()
    logger.info("STT stage ready.")

    while not state.shutdown_event.is_set():
        try:
            item = await asyncio.wait_for(state.audio_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        state.active_stage = "STT"
        t0 = time.perf_counter()

        try:
            text = await asyncio.to_thread(stt_engine.transcribe, item["audio"])
        except Exception as exc:
            logger.error("STT error: %s", exc)
            state.audio_q.task_done()
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        state.latencies["stt"] = elapsed_ms
        await state.ws_broadcast({"type": "latency", "stage": "stt", "ms": round(elapsed_ms)})

        if text:
            console.print(f"[cyan]You:[/cyan] {text}")
            await state.ws_broadcast({"type": "transcript", "role": "user", "text": text})
            await state.ws_broadcast({"type": "status", "value": "processing"})
            await asyncio.wait_for(
                state.stt_q.put({"text": text, "t_stt": time.perf_counter()}),
                timeout=cfg.system.pipeline_timeout_s,
            )
        else:
            logger.debug("STT returned empty string; skipping.")

        state.audio_q.task_done()


async def rag_stage(state: PipelineState) -> None:
    """
    Cache-first retrieval stage.

    1. Check the semantic response cache — if the query matches a pre-written
       answer above the similarity threshold, bypass RAG + LLM entirely and
       send the cached answer straight to TTS (~270ms E2E).
    2. On cache miss, fall back to FAISS RAG retrieval and pass chunks to LLM.
    """
    from rag_storage import RAGEngine
    from response_cache import get_cache

    cfg = get_config()
    engine = RAGEngine(cfg)
    cache = get_cache()

    # Load/build FAISS doc index
    if not engine.load():
        console.print("[yellow]No RAG index found – ingesting docs …[/yellow]")
        try:
            n = await asyncio.to_thread(engine.ingest, cfg.rag.docs_dir)
            console.print(f"[green]Ingested {n} chunks.[/green]")
        except Exception as exc:
            logger.error("RAG ingest failed: %s", exc)

    logger.info("RAG stage ready (indexed=%s).", engine.is_indexed())

    while not state.shutdown_event.is_set():
        try:
            item = await asyncio.wait_for(state.stt_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        state.active_stage = "RAG"
        t0 = time.perf_counter()
        query = item["text"]

        # ── Conversational filler filter (pre-cache, no embedding needed) ─
        _FILLER_REPLY = _is_conversational_filler(query)
        if _FILLER_REPLY:
            await state.ws_broadcast({"type": "transcript", "role": "assistant", "text": _FILLER_REPLY})
            await state.ws_broadcast({"type": "status", "value": "speaking"})
            await asyncio.wait_for(
                state.llm_q.put({"response": _FILLER_REPLY, "t_llm": time.perf_counter()}),
                timeout=cfg.system.pipeline_timeout_s,
            )
            state.stt_q.task_done()
            continue

        # ── Embed query once, reuse for cache lookup and RAG retrieval ───
        query_vec = await asyncio.to_thread(cache.embed_query, query)

        # ── Cache lookup (fast path) ──────────────────────────────────────
        cached_answer = cache.lookup_vec(query_vec)   # FAISS search is fast inline
        if cached_answer:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            state.latencies["rag"] = elapsed_ms
            state.latencies["llm"] = 0.0
            await state.ws_broadcast({"type": "latency", "stage": "rag", "ms": round(elapsed_ms), "cached": True})
            await state.ws_broadcast({"type": "latency", "stage": "llm", "ms": 0, "cached": True})
            console.print(f"[bold green]Assistant:[/bold green] {cached_answer} [dim](cached)[/dim]")
            await state.ws_broadcast({"type": "transcript", "role": "assistant", "text": cached_answer})
            await state.ws_broadcast({"type": "status", "value": "speaking"})
            await asyncio.wait_for(
                state.llm_q.put({"response": cached_answer, "t_llm": time.perf_counter()}),
                timeout=cfg.system.pipeline_timeout_s,
            )
            state.stt_q.task_done()
            continue

        # ── Cache miss — full RAG retrieval ──────────────────────────────
        try:
            chunks = await asyncio.to_thread(engine.retrieve_vec, query_vec, cfg.rag.top_k)
        except Exception as exc:
            logger.error("RAG retrieve error: %s", exc)
            chunks = ["No relevant context found."]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        state.latencies["rag"] = elapsed_ms
        await state.ws_broadcast({"type": "latency", "stage": "rag", "ms": round(elapsed_ms)})

        await asyncio.wait_for(
            state.rag_q.put({"query": query, "chunks": chunks, "t_rag": time.perf_counter()}),
            timeout=cfg.system.pipeline_timeout_s,
        )
        state.stt_q.task_done()


async def llm_stage(state: PipelineState) -> None:
    """
    Generate LLM response and stream sentence-by-sentence to TTS.
    TTS starts playing the first sentence while the LLM generates the rest.
    """
    import llm_engine

    cfg = get_config()
    logger.info("LLM stage ready.")

    while not state.shutdown_event.is_set():
        try:
            item = await asyncio.wait_for(state.rag_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        state.active_stage = "LLM"
        t0 = time.perf_counter()
        query: str = item["query"]
        chunks: List[str] = item["chunks"]

        # Sync queue — LLM thread pushes sentences, async loop drains them
        sent_q: sync_queue.Queue = sync_queue.Queue()

        def _generate():
            buf = ""
            try:
                for token in llm_engine.stream_response(query, chunks):
                    buf += token
                    # Flush on sentence boundary
                    for delim in _SENT_DELIMS:
                        while delim in buf:
                            idx = buf.index(delim) + len(delim)
                            sentence = buf[:idx].strip()
                            buf = buf[idx:]
                            if len(sentence) > 2:
                                sent_q.put(sentence)
            except Exception as exc:
                logger.error("LLM generation error: %s", exc, exc_info=True)
            finally:
                if buf.strip():
                    sent_q.put(buf.strip())
                sent_q.put(None)  # sentinel

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        full_parts: List[str] = []
        first_sentence = True
        while True:
            # Poll non-blockingly so the event loop stays alive
            try:
                sentence = sent_q.get(timeout=0.02)
            except sync_queue.Empty:
                await asyncio.sleep(0)
                continue

            if sentence is None:
                break

            if first_sentence:
                state.latencies["llm"] = (time.perf_counter() - t0) * 1000
                first_sentence = False
                await state.ws_broadcast({"type": "latency", "stage": "llm", "ms": round(state.latencies["llm"])})

            full_parts.append(sentence)
            console.print(f"[bold green]Assistant:[/bold green] {sentence}")
            await state.ws_broadcast({"type": "transcript", "role": "assistant", "text": sentence})
            await state.ws_broadcast({"type": "status", "value": "speaking"})

            # Send each sentence to TTS immediately — don't wait for full response
            try:
                await asyncio.wait_for(
                    state.llm_q.put({"response": sentence, "t_llm": time.perf_counter()}),
                    timeout=cfg.system.pipeline_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM→TTS queue full, dropping sentence.")

        gen_thread.join(timeout=2.0)
        state.rag_q.task_done()


async def tts_stage(state: PipelineState) -> None:
    """Synthesise speech for each LLM response sentence."""
    from tts_engine import TTSEngine

    cfg = get_config()
    engine = TTSEngine(cfg)
    logger.info("TTS stage ready.")

    # Capture the event loop now (we're in the asyncio thread); background
    # threads spawned later cannot call asyncio.get_event_loop() themselves.
    _loop = asyncio.get_event_loop()

    # Thread-safe counter: how many sentences are still synthesising or playing
    _lock = threading.Lock()
    _in_flight = [0]

    def _acquire():
        with _lock:
            _in_flight[0] += 1

    def _release():
        with _lock:
            _in_flight[0] -= 1
            if _in_flight[0] > 0:
                return
            _in_flight[0] = 0

        # Cooldown: wait for physical audio to finish reverberating before
        # re-enabling the mic, otherwise the speaker output echoes back in.
        def _delayed_clear():
            time.sleep(0.7)
            state.pipeline_busy.clear()
            logger.debug("Pipeline idle — ready for next input")
            _loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(state.ws_broadcast({"type": "status", "value": "listening"}))
            )

        threading.Thread(target=_delayed_clear, daemon=True, name="tts-cooldown").start()

    while not state.shutdown_event.is_set():
        try:
            item = await asyncio.wait_for(state.llm_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        state.active_stage = "TTS"
        t0 = time.perf_counter()
        response: str = item["response"]

        if not response.strip():
            state.llm_q.task_done()
            continue

        _acquire()  # count this sentence before synthesis starts

        try:
            audio, visemes = await asyncio.to_thread(engine.synthesize_chunk, response)
        except Exception as exc:
            logger.error("TTS synthesis error: %s", exc)
            _release()
            state.llm_q.task_done()
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        state.latencies["tts"] = elapsed_ms
        await state.ws_broadcast({"type": "latency", "stage": "tts", "ms": round(elapsed_ms)})

        engine.play_audio(audio, done_callback=_release)

        await asyncio.wait_for(
            state.tts_q.put({
                "audio": audio,
                "visemes": visemes,
                "text": response,
                "t_tts": time.perf_counter(),
            }),
            timeout=cfg.system.pipeline_timeout_s,
        )
        state.llm_q.task_done()


async def avatar_stage(state: PipelineState) -> None:
    cfg = get_config()
    output_mode: str = cfg.avatar.output_mode
    logger.info("Avatar stage ready (output_mode=%s).", output_mode)

    while not state.shutdown_event.is_set():
        try:
            item = await asyncio.wait_for(state.tts_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        state.active_stage = "Avatar"
        visemes = item.get("visemes", [])
        t_play = item.get("t_tts", time.perf_counter())

        for event in visemes:
            target_ms = event["time_ms"]
            elapsed_ms = (time.perf_counter() - t_play) * 1000
            sleep_s = (target_ms - elapsed_ms) / 1000
            if sleep_s > 0.005:
                await asyncio.sleep(sleep_s)

            msg = {"type": "viseme", "viseme": event["viseme"], "amplitude": event["amplitude"]}

            if output_mode == "stdout":
                print(json.dumps(msg), flush=True)
            elif output_mode == "websocket":
                await state.ws_broadcast(msg)
            elif output_mode == "file":
                import os
                out_path = os.path.join(os.path.dirname(__file__), "viseme_output.jsonl")
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(msg) + "\n")

        # Send silence at end
        await state.ws_broadcast({"type": "viseme", "viseme": "sil", "amplitude": 0.0})

        state.tts_q.task_done()
        state.active_stage = "idle"


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------

def _build_status_table(state: PipelineState) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Latency")

    for stage in STAGES:
        is_active = state.active_stage.upper() == stage.upper()
        status_text = Text("● ACTIVE", style="bold green") if is_active else Text("○ idle", style="dim")
        lat_key = stage.lower()
        lat_val = state.latencies.get(lat_key)
        lat_text = f"{lat_val:.0f} ms" if lat_val is not None else "—"
        table.add_row(stage, status_text, lat_text)

    return Panel(table, title="[bold cyan]WaveTec InfoPoint Pipeline[/bold cyan]", border_style="cyan")


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def _run_pipeline() -> None:
    """Launch all pipeline stage tasks and wait for shutdown."""
    cfg = get_config()
    state = PipelineState()

    # Install Ctrl-C / SIGTERM handler
    loop = asyncio.get_running_loop()

    def _shutdown_handler():
        console.print("\n[yellow]Shutting down…[/yellow]")
        state.shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            pass  # Windows

    # Start WebSocket server
    ws_server = None
    if _WS_AVAILABLE:
        import websockets
        ws_server = await websockets.serve(
            lambda ws, _p=None: _ws_handler(ws, state),
            "localhost", 8765
        )
        logger.info("WebSocket server started on ws://localhost:8765")

    tasks = [
        asyncio.create_task(vad_stage(state), name="vad"),
        asyncio.create_task(stt_stage(state), name="stt"),
        asyncio.create_task(rag_stage(state), name="rag"),
        asyncio.create_task(llm_stage(state), name="llm"),
        asyncio.create_task(tts_stage(state), name="tts"),
        asyncio.create_task(avatar_stage(state), name="avatar"),
    ]

    console.print(
        Panel(
            "[bold green]WaveTec InfoPoint Kiosk[/bold green]\n"
            "Voice assistant is running. Press [bold]Ctrl+C[/bold] to quit.",
            border_style="green",
        )
    )

    # Pre-warm all models before listening starts
    await _prewarm_models()

    # Open UI in browser
    ui_path = _Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        webbrowser.open(f"file://{ui_path}")
        console.print("[cyan]Avatar UI opened in browser.[/cyan]")

    # Broadcast initial status
    await state.ws_broadcast({"type": "status", "value": "listening"})

    # Refresh live display every 500ms (rendered on stderr to avoid clashing with logs)
    try:
        with Live(
            _build_status_table(state),
            console=_panel_console,
            refresh_per_second=2,
            screen=False,
        ) as live:
            while not state.shutdown_event.is_set():
                await asyncio.sleep(0.5)
                live.update(_build_status_table(state))
    except Exception as exc:
        logger.debug("Live display error: %s", exc)

    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    if ws_server:
        ws_server.close()
        await ws_server.wait_closed()

    console.print("[bold green]Pipeline stopped.[/bold green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """WaveTec InfoPoint — Offline Kiosk Voice Assistant."""
    pass


@cli.command()
def run():
    """Start the kiosk voice assistant pipeline."""
    try:
        asyncio.run(_run_pipeline())
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")


@cli.command()
def ingest():
    """Ingest documents from the configured docs directory and exit."""
    from rag_storage import RAGEngine

    cfg = get_config()
    console.print(f"[cyan]Ingesting documents from:[/cyan] {cfg.rag.docs_dir}")
    engine = RAGEngine(cfg)

    try:
        n = engine.ingest(cfg.rag.docs_dir)
        console.print(f"[bold green]Success:[/bold green] Indexed {n} chunks.")
    except Exception as exc:
        console.print(f"[bold red]Ingest failed:[/bold red] {exc}")
        sys.exit(1)


@cli.command()
def status():
    """Print system information and configured model paths."""
    import platform

    cfg = get_config()

    table = Table(title="[bold cyan]System Status[/bold cyan]", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Value")

    table.add_row("Platform", platform.platform())
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Device", cfg.system.device)
    table.add_row("Log level", cfg.system.log_level)
    table.add_row("", "")
    table.add_row("STT model", cfg.stt.model)
    table.add_row("LLM model", cfg.llm.model)
    table.add_row("TTS model", cfg.tts.model_path)
    table.add_row("Embedding model", cfg.rag.embedding_model)
    table.add_row("Docs dir", cfg.rag.docs_dir)
    table.add_row("RAG index", cfg.rag.index_path)

    # Check TTS model file
    from pathlib import Path
    tts_exists = Path(cfg.tts.model_path).exists()
    rag_exists = Path(cfg.rag.index_path).exists()
    table.add_row("TTS model file", "[green]Found[/green]" if tts_exists else "[red]Missing[/red]")
    table.add_row("RAG index file", "[green]Found[/green]" if rag_exists else "[yellow]Not built yet[/yellow]")

    console.print(table)

    # Try importing key dependencies
    console.print("\n[bold]Dependency check:[/bold]")
    deps = [
        ("mlx", "mlx"),
        ("mlx_whisper", "mlx-whisper"),
        ("mlx_lm", "mlx-lm"),
        ("sounddevice", "sounddevice"),
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
    ]
    for mod, pkg in deps:
        try:
            __import__(mod)
            console.print(f"  [green]✓[/green] {pkg}")
        except ImportError:
            console.print(f"  [red]✗[/red] {pkg} — run: pip install {pkg}")


if __name__ == "__main__":
    cli()
