"""
test_pipeline.py
----------------
Standalone benchmark / smoke-test runner for each pipeline node.

Usage:
  python test_pipeline.py test-stt
  python test_pipeline.py test-rag
  python test_pipeline.py test-llm
  python test_pipeline.py test-tts
  python test_pipeline.py test-vad
  python test_pipeline.py test-all
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from config import get_config

console = Console()
logging.basicConfig(level=logging.WARNING)  # quiet during tests

# ---------------------------------------------------------------------------
# Latency targets (milliseconds)
# ---------------------------------------------------------------------------

TARGETS: Dict[str, float] = {
    "stt": 500.0,
    "rag": 100.0,
    "llm_ttft": 100.0,
    "llm_total": 3000.0,
    "tts": 500.0,
    "vad": 300.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass_fail(actual_ms: float, target_ms: float) -> Text:
    if actual_ms <= target_ms:
        return Text(f"{actual_ms:.0f} ms  ✓", style="bold green")
    return Text(f"{actual_ms:.0f} ms  ✗  (target {target_ms:.0f} ms)", style="bold red")


def _make_sine_audio(
    duration_s: float = 3.0,
    freq_hz: float = 440.0,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Generate a pure sine wave as fake speech audio."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def _run_test_stt() -> Dict:
    """Run STT on a 3-second 440 Hz sine wave. Returns result dict."""
    import stt_engine

    console.print("\n[bold cyan]── STT Test ──[/bold cyan]")
    console.print("Generating 3-second 440 Hz sine wave …")
    audio = _make_sine_audio(duration_s=3.0, freq_hz=440.0)

    console.print("Running transcription …")
    t0 = time.perf_counter()
    result = stt_engine.transcribe(audio)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    console.print(f"Transcript: [italic]'{result}'[/italic]")
    console.print(f"Latency:    {_pass_fail(elapsed_ms, TARGETS['stt'])}")

    return {"name": "STT", "latency_ms": elapsed_ms, "target_ms": TARGETS["stt"], "output": result}


def _run_test_rag() -> Dict:
    """Ingest docs, run 3 test queries, return result dict."""
    from rag_storage import RAGEngine

    cfg = get_config()
    console.print("\n[bold cyan]── RAG Test ──[/bold cyan]")

    engine = RAGEngine(cfg)
    if not engine.load():
        console.print("Index not found – ingesting …")
        n = engine.ingest(cfg.rag.docs_dir)
        console.print(f"Ingested {n} chunks.")

    test_queries = [
        "What are the hours of operation?",
        "What services does WaveTec InfoPoint offer?",
        "How do I get directions to the main office?",
    ]

    total_ms = 0.0
    for query in test_queries:
        t0 = time.perf_counter()
        chunks = engine.retrieve(query, top_k=cfg.rag.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[bold]Latency:[/bold] {_pass_fail(elapsed_ms, TARGETS['rag'])}")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:120].replace("\n", " ")
            console.print(f"  [{i}] {preview}…")

    avg_ms = total_ms / len(test_queries)
    return {
        "name": "RAG",
        "latency_ms": avg_ms,
        "target_ms": TARGETS["rag"],
        "output": f"{len(test_queries)} queries, avg {avg_ms:.0f} ms",
    }


def _run_test_llm() -> Dict:
    """Run sample prompt through LLM and measure TTFT + total latency."""
    import llm_engine

    console.print("\n[bold cyan]── LLM Test ──[/bold cyan]")

    sample_query = "What are the services available at WaveTec InfoPoint?"
    sample_context = [
        "WaveTec InfoPoint offers visitor registration, wayfinding, and concierge services. "
        "Operating hours are Monday–Friday 8 AM to 8 PM.",
        "The kiosk is located in the main lobby of WaveTec Tower, 1 Innovation Drive.",
    ]

    console.print(f"Query: {sample_query}")
    console.print("Streaming response …\n")

    first_token_time: Optional[float] = None
    t0 = time.perf_counter()
    full_output_parts: List[str] = []

    try:
        for chunk in llm_engine.stream_response(sample_query, sample_context):
            if first_token_time is None:
                first_token_time = (time.perf_counter() - t0) * 1000
            full_output_parts.append(chunk)
            console.print(chunk, end="", highlight=False)

    except Exception as exc:
        console.print(f"\n[red]LLM error: {exc}[/red]")
        return {
            "name": "LLM",
            "latency_ms": 0,
            "target_ms": TARGETS["llm_total"],
            "output": f"ERROR: {exc}",
        }

    total_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = first_token_time or total_ms
    full_output = "".join(full_output_parts).strip()

    console.print(f"\n\n[bold]TTFT:[/bold]  {_pass_fail(ttft_ms, TARGETS['llm_ttft'])}")
    console.print(f"[bold]Total:[/bold] {_pass_fail(total_ms, TARGETS['llm_total'])}")

    return {
        "name": "LLM (TTFT)",
        "latency_ms": ttft_ms,
        "target_ms": TARGETS["llm_ttft"],
        "output": full_output[:80] + "…" if len(full_output) > 80 else full_output,
    }


def _run_test_tts() -> Dict:
    """Synthesise a test sentence, play it, and print viseme events."""
    from tts_engine import TTSEngine

    cfg = get_config()
    console.print("\n[bold cyan]── TTS Test ──[/bold cyan]")

    engine = TTSEngine(cfg)
    test_sentence = (
        "Welcome to WaveTec InfoPoint. I am your virtual assistant. How may I help you today?"
    )

    console.print(f"Synthesising: '{test_sentence}'")
    t0 = time.perf_counter()
    audio, visemes = engine.synthesize_chunk(test_sentence)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    console.print(f"Audio:    {len(audio) / cfg.tts.sample_rate:.2f} s  ({len(audio)} samples)")
    console.print(f"Visemes:  {len(visemes)} events")
    console.print(f"Latency:  {_pass_fail(elapsed_ms, TARGETS['tts'])}")

    # Print first 10 viseme events
    console.print("\n[bold]First 10 viseme events:[/bold]")
    for ev in visemes[:10]:
        console.print(
            f"  t={ev['time_ms']:6.1f} ms  viseme={ev['viseme']:<4}  amp={ev['amplitude']:.4f}"
        )

    console.print("\nPlaying audio …")
    engine.play_audio(audio)
    import time as _time
    _time.sleep(len(audio) / cfg.tts.sample_rate + 0.5)  # wait for playback
    console.print("Done.")

    return {
        "name": "TTS",
        "latency_ms": elapsed_ms,
        "target_ms": TARGETS["tts"],
        "output": f"{len(audio) / cfg.tts.sample_rate:.2f}s audio, {len(visemes)} visemes",
    }


async def _run_test_vad_async() -> Dict:
    """Open microphone for 5 seconds and print detected phrases."""
    from audio_input import VADStream

    cfg = get_config()
    console.print("\n[bold cyan]── VAD Test ──[/bold cyan]")
    console.print("Opening microphone for 5 seconds. Speak now …")

    stream = VADStream(cfg)
    phrase_count = 0
    latencies: List[float] = []

    try:
        stream.start()
    except OSError as exc:
        console.print(f"[red]Microphone error: {exc}[/red]")
        return {
            "name": "VAD",
            "latency_ms": 0.0,
            "target_ms": TARGETS["vad"],
            "output": f"ERROR: {exc}",
        }

    async def _collect():
        nonlocal phrase_count
        async for phrase in stream.iter_phrases():
            t_vad = time.perf_counter()
            duration_s = len(phrase) / cfg.vad.sample_rate
            latencies.append(duration_s * 1000)
            phrase_count += 1
            console.print(
                f"  Phrase {phrase_count}: {duration_s:.2f}s  "
                f"({len(phrase)} samples)"
            )

    try:
        await asyncio.wait_for(_collect(), timeout=5.0)
    except asyncio.TimeoutError:
        pass
    finally:
        stream.stop()

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    console.print(f"\nDetected {phrase_count} phrase(s) in 5 seconds.")
    console.print(f"Avg phrase duration: {_pass_fail(avg_lat, TARGETS['vad'])}")

    return {
        "name": "VAD",
        "latency_ms": avg_lat,
        "target_ms": TARGETS["vad"],
        "output": f"{phrase_count} phrase(s) detected",
    }


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """WaveTec InfoPoint — Pipeline test runner."""
    pass


@cli.command("test-stt")
def cmd_test_stt():
    """Test the STT engine with a synthetic sine-wave audio input."""
    _run_test_stt()


@cli.command("test-rag")
def cmd_test_rag():
    """Ingest docs and run 3 retrieval queries, printing results and latency."""
    _run_test_rag()


@cli.command("test-llm")
def cmd_test_llm():
    """Run a sample query through the LLM, printing streamed output and TTFT."""
    _run_test_llm()


@cli.command("test-tts")
def cmd_test_tts():
    """Synthesise a test sentence, play it, and print viseme events."""
    _run_test_tts()


@cli.command("test-vad")
def cmd_test_vad():
    """Open the microphone for 5 seconds and print detected speech phrases."""
    asyncio.run(_run_test_vad_async())


@cli.command("test-all")
def cmd_test_all():
    """Run all pipeline tests sequentially and print a summary table."""
    console.print(
        "\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]"
    )
    console.print("[bold cyan]        WaveTec InfoPoint — Full Pipeline Benchmark[/bold cyan]")
    console.print(
        "[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n"
    )

    results: List[Dict] = []

    # Run each test and collect results
    tests = [
        ("STT", _run_test_stt),
        ("RAG", _run_test_rag),
        ("LLM", _run_test_llm),
        ("TTS", _run_test_tts),
    ]

    for name, fn in tests:
        try:
            res = fn()
            results.append(res)
        except Exception as exc:
            console.print(f"\n[red]{name} test FAILED: {exc}[/red]")
            results.append({
                "name": name,
                "latency_ms": float("inf"),
                "target_ms": 500.0,
                "output": f"EXCEPTION: {exc}",
            })

    # VAD test
    try:
        vad_res = asyncio.run(_run_test_vad_async())
        results.append(vad_res)
    except Exception as exc:
        console.print(f"\n[red]VAD test FAILED: {exc}[/red]")
        results.append({
            "name": "VAD",
            "latency_ms": float("inf"),
            "target_ms": TARGETS["vad"],
            "output": f"EXCEPTION: {exc}",
        })

    # Summary table
    console.print(
        "\n\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━  SUMMARY  ━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", min_width=12)
    table.add_column("Latency", justify="right", min_width=12)
    table.add_column("Target", justify="right", min_width=12)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Output", min_width=30)

    passed = 0
    failed = 0

    for r in results:
        lat = r["latency_ms"]
        tgt = r["target_ms"]
        if lat <= tgt:
            status = Text("PASS", style="bold green")
            passed += 1
        else:
            status = Text("FAIL", style="bold red")
            failed += 1

        lat_str = f"{lat:.0f} ms" if lat != float("inf") else "N/A"
        table.add_row(
            r["name"],
            lat_str,
            f"{tgt:.0f} ms",
            status,
            str(r.get("output", ""))[:50],
        )

    console.print(table)
    console.print(
        f"\n[bold green]{passed} passed[/bold green]  "
        f"[bold red]{failed} failed[/bold red]  "
        f"out of {len(results)} tests."
    )


if __name__ == "__main__":
    cli()
