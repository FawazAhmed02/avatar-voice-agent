"""
download_models.py
------------------
Downloads all models required by the kiosk voice assistant pipeline
and places them in the exact locations the application expects.

Models downloaded:
  1. Silero VAD          → torch hub cache (~/.cache/torch/hub/)
  2. mlx-whisper small   → HuggingFace cache (~/.cache/huggingface/)
  3. all-MiniLM-L6-v2    → sentence-transformers cache
  4. Phi-3-mini 4-bit    → HuggingFace cache (~/.cache/huggingface/)
  5. Piper TTS (lessac)  → ./tts_models/

Usage:
  python download_models.py            # download all
  python download_models.py --skip-llm # skip the large LLM (2.3 GB)
  python download_models.py --only tts vad  # download specific models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich console (graceful fallback if not installed yet)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False
    class Console:  # type: ignore
        def print(self, *a, **kw): print(*a)
        def log(self, *a, **kw): print(*a)
    rprint = print  # type: ignore

console = Console()

# ---------------------------------------------------------------------------
# Paths relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
TTS_MODEL_DIR = SCRIPT_DIR / "tts_models"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: (key, display_name, approx_size_mb)
MODELS = {
    "vad":      ("Silero VAD",              "~5 MB"),
    "stt":      ("mlx-whisper small.en",    "~150 MB"),
    "embed":    ("all-MiniLM-L6-v2",        "~90 MB"),
    "llm":      ("Phi-3-mini 4-bit (MLX)",  "~2.3 GB"),
    "tts":      ("Piper TTS en_US-lessac",  "~63 MB"),
}

# Piper model files on HuggingFace
PIPER_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    "/en/en_US/lessac/medium"
)
PIPER_FILES = [
    "en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_import(package: str) -> bool:
    import importlib
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _download_file(url: str, dest: Path, label: str) -> bool:
    """Download a single file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        console.print(f"  [green]✓ Already exists:[/green] {dest.name} ({_sizeof_fmt(dest.stat().st_size)})")
        return True

    console.print(f"  [cyan]↓ Downloading:[/cyan] {label}")
    console.print(f"    URL : {url}")
    console.print(f"    Dest: {dest}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        if _RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"  {dest.name}", total=None)

                def _reporthook(count, block_size, total_size):
                    if total_size > 0:
                        progress.update(task, total=total_size, completed=count * block_size)

                urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        else:
            urllib.request.urlretrieve(url, tmp)

        tmp.rename(dest)
        console.print(f"  [green]✓ Saved:[/green] {dest.name} ({_sizeof_fmt(dest.stat().st_size)})")
        return True

    except Exception as exc:
        console.print(f"  [red]✗ Download failed:[/red] {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


# ---------------------------------------------------------------------------
# Per-model download functions
# ---------------------------------------------------------------------------

def download_vad() -> bool:
    """Download Silero VAD via torch.hub (caches to ~/.cache/torch/hub/)."""
    console.print("\n[bold]── 1 / 5  Silero VAD ──[/bold]")
    if not _check_import("torch"):
        console.print("  [red]✗ torch not installed. Run: pip install torch[/red]")
        return False
    try:
        import torch
        hub_dir = Path(torch.hub.get_dir())
        # Check if already cached
        vad_cache = list(hub_dir.glob("snakers4_silero-vad*"))
        if vad_cache:
            console.print(f"  [green]✓ Already cached at:[/green] {vad_cache[0]}")
            return True

        console.print("  Pulling from torch.hub (snakers4/silero-vad) …")
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        console.print("  [green]✓ Silero VAD cached successfully[/green]")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ Failed:[/red] {exc}")
        return False


def download_stt() -> bool:
    """Download mlx-whisper small.en via huggingface_hub snapshot_download."""
    console.print("\n[bold]── 2 / 5  mlx-whisper (whisper-small.en-mlx) ──[/bold]")

    if not _check_import("huggingface_hub"):
        console.print("  [red]✗ huggingface_hub not installed. Run: pip install huggingface-hub[/red]")
        return False

    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        repo_id = "mlx-community/whisper-small.en-mlx"

        # Check if already cached
        cached = try_to_load_from_cache(repo_id, "config.json")
        if cached and Path(cached).exists():
            console.print(f"  [green]✓ Already cached:[/green] {repo_id}")
            return True

        console.print(f"  Downloading {repo_id} …")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        console.print(f"  [green]✓ Cached at:[/green] {path}")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ Failed:[/red] {exc}")
        return False


def download_embed() -> bool:
    """Download all-MiniLM-L6-v2 via sentence-transformers."""
    console.print("\n[bold]── 3 / 5  Embedding model (all-MiniLM-L6-v2) ──[/bold]")

    if not _check_import("sentence_transformers"):
        console.print("  [red]✗ sentence-transformers not installed. Run: pip install sentence-transformers[/red]")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        # Check HF cache first
        if _check_import("huggingface_hub"):
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
            if cached and Path(cached).exists():
                console.print("  [green]✓ Already cached:[/green] sentence-transformers/all-MiniLM-L6-v2")
                return True

        console.print("  Downloading all-MiniLM-L6-v2 …")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Verify it works
        _ = model.encode(["test"], convert_to_numpy=True)
        cache_dir = Path(model._model_card_data.base_model if hasattr(model, '_model_card_data') else "~/.cache")
        console.print("  [green]✓ Model downloaded and verified (test encode passed)[/green]")
        del model
        return True
    except Exception as exc:
        console.print(f"  [red]✗ Failed:[/red] {exc}")
        return False


def download_llm() -> bool:
    """Download Phi-3-mini 4-bit MLX model via huggingface_hub."""
    console.print("\n[bold]── 4 / 5  LLM (Phi-3-mini-4k-instruct-4bit, MLX) ──[/bold]")
    console.print("  [yellow]⚠  This model is ~2.3 GB. Use --skip-llm to skip.[/yellow]")

    if not _check_import("huggingface_hub"):
        console.print("  [red]✗ huggingface_hub not installed. Run: pip install huggingface-hub[/red]")
        return False

    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        repo_id = "mlx-community/Phi-3-mini-4k-instruct-4bit"

        cached = try_to_load_from_cache(repo_id, "config.json")
        if cached and Path(cached).exists():
            console.print(f"  [green]✓ Already cached:[/green] {repo_id}")
            return True

        console.print(f"  Downloading {repo_id} …")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        console.print(f"  [green]✓ Cached at:[/green] {path}")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ Failed:[/red] {exc}")
        return False


def download_tts() -> bool:
    """Download Piper TTS en_US-lessac-medium model to ./tts_models/."""
    console.print("\n[bold]── 5 / 5  Piper TTS (en_US-lessac-medium) ──[/bold]")

    TTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"  Target dir: {TTS_MODEL_DIR}")

    all_ok = True
    for fname in PIPER_FILES:
        url = f"{PIPER_BASE_URL}/{fname}"
        dest = TTS_MODEL_DIR / fname
        ok = _download_file(url, dest, fname)
        if not ok:
            all_ok = False

    if all_ok:
        console.print("  [green]✓ Piper TTS model ready[/green]")
    return all_ok


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: dict[str, bool]) -> None:
    if not _RICH:
        print("\n=== Download Summary ===")
        for key, ok in results.items():
            status = "OK" if ok else "FAILED"
            name = MODELS[key][0]
            print(f"  [{status}] {name}")
        return

    from rich.table import Table
    table = Table(title="Download Summary", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Approx Size", justify="right")
    table.add_column("Status", justify="center")

    for key, ok in results.items():
        name, size = MODELS[key]
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAILED[/red]"
        table.add_row(name, size, status)

    console.print()
    console.print(table)

    failed = [k for k, v in results.items() if not v]
    if failed:
        console.print(f"\n[red]Some downloads failed: {', '.join(failed)}[/red]")
        console.print("Check your internet connection and re-run. Already-downloaded models will be skipped.")
        sys.exit(1)
    else:
        console.print("\n[bold green]All models downloaded successfully. You're ready to run the pipeline![/bold green]")
        console.print(f"\nNext steps:")
        console.print("  1.  [cyan]python main.py ingest[/cyan]   ← index your docs")
        console.print("  2.  [cyan]python main.py run[/cyan]      ← start the kiosk")
        console.print("  3.  [cyan]python test_pipeline.py test-all[/cyan]  ← benchmark each node")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download all models for the kiosk voice assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py                   # download everything
  python download_models.py --skip-llm        # skip the 2.3 GB LLM
  python download_models.py --only tts vad    # only TTS + VAD
  python download_models.py --only embed stt  # only embedding + STT
""",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM download (~2.3 GB). Useful for testing other nodes.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(MODELS.keys()),
        metavar="MODEL",
        help=f"Download only these models. Choices: {', '.join(MODELS.keys())}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which models to download
    if args.only:
        keys = args.only
    else:
        keys = list(MODELS.keys())
        if args.skip_llm:
            keys = [k for k in keys if k != "llm"]

    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Kiosk Voice Assistant — Model Downloader    [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print(f"Downloading: [yellow]{', '.join(MODELS[k][0] for k in keys)}[/yellow]")

    download_fn = {
        "vad":   download_vad,
        "stt":   download_stt,
        "embed": download_embed,
        "llm":   download_llm,
        "tts":   download_tts,
    }

    results: dict[str, bool] = {}
    t0 = time.perf_counter()

    for key in keys:
        results[key] = download_fn[key]()

    elapsed = time.perf_counter() - t0
    console.print(f"\nTotal time: [yellow]{elapsed:.1f}s[/yellow]")

    _print_summary(results)


if __name__ == "__main__":
    main()
