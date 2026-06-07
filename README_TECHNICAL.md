# WaveTec InfoPoint — Technical Reference

## Architecture Overview

A fully offline, real-time voice assistant pipeline designed for Apple Silicon kiosks. Every component runs locally using MLX (Apple's ML framework), Metal Performance Shaders, and ONNX Runtime. No internet is required at runtime.

```
Microphone
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (asyncio)                        │
│                                                                 │
│  ┌───────┐   ┌───────┐   ┌───────────┐   ┌───────┐   ┌───────┐│
│  │  VAD  │──▶│  STT  │──▶│ RAG/Cache │──▶│  LLM  │──▶│  TTS  ││
│  │Silero │   │Whisper│   │FAISS+MiniLM│  │ Qwen  │   │ Piper ││
│  └───────┘   └───────┘   └───────────┘   └───────┘   └───────┘│
│       asyncio.Queue between each stage (maxsize=4)              │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ WebSocket (port 8765)
                                   ▼
                          ui/index.html (browser)
                          Canvas avatar + chat + metrics
```

All stages run as **concurrent asyncio tasks**. Audio playback happens in a background thread to avoid blocking the event loop.

---

## Pipeline Stages

### 1. VAD — Voice Activity Detection
**File:** `audio_input.py`  
**Model:** Silero VAD v5 (via `torch.hub`)

- Captures 16 kHz mono audio in 512-sample chunks via `sounddevice`
- Runs Silero VAD on each chunk; probability threshold: `0.7`
- Declares speech end after `280 ms` of silence
- Emits a phrase buffer (numpy float32 array) to `audio_q`
- **Echo gate:** drops all phrases while `pipeline_busy` event is set; 700 ms cooldown after TTS playback ends before accepting new input

### 2. STT — Speech-to-Text
**File:** `stt_engine.py`  
**Model:** `mlx-community/whisper-tiny.en-mlx`

- Receives phrase audio from `audio_q`
- Transcribes using `mlx_whisper.transcribe()` (greedy decoding — beam search not supported)
- Typical latency: **~30–50 ms** on M4
- Broadcasts `transcript` (user) + `status: processing` + `latency.stt` over WebSocket
- Empty transcripts are discarded

### 3. RAG — Retrieval-Augmented Generation
**File:** `rag_storage.py`, `response_cache.py`  
**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, on MPS)  
**Index:** FAISS `IndexFlatIP` (cosine similarity via L2-normalised vectors)

Two paths:

**Cache hit path (~40 ms):**
- `response_cache.py` embeds the query and searches a small pre-built FAISS index of canonical Q&A pairs
- Similarity threshold: `0.78`
- On hit, the cached answer bypasses RAG and LLM entirely and goes straight to TTS
- ~50–60 Q&A variants across 10 topic groups (hours, location, Wi-Fi, parking, etc.)

**Cache miss path (~80–120 ms):**
- Full document index searched; returns top-`k=3` chunks
- Chunks passed to LLM stage via `rag_q`

The embedding model runs on **MPS** (Metal Performance Shaders), giving ~24× speedup vs CPU (~35 ms vs ~830 ms).

### 4. LLM — Language Model
**File:** `llm_engine.py`  
**Model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit`

- Uses `mlx_lm.stream_generate()` with `make_sampler(temp=0.0)` (greedy)
- Runs in a background thread; sentences are streamed into `sent_q` (sync queue) as they complete
- The asyncio event loop dequeues sentences and forwards each to TTS immediately — **TTS starts before LLM finishes**
- System prompt: `"Answer in 1-2 sentences from context only."`
- Context capped at `300` characters to keep prompt short and latency low
- `max_tokens: 80`, `temperature: 0.0`
- TTFT (time-to-first-token) recorded and broadcast as `latency.llm`

Sentence splitting delimiters: `. ! ?` — each chunk is sent to TTS as soon as it ends.

### 5. TTS — Text-to-Speech
**File:** `tts_engine.py`  
**Model:** Piper `en_US-lessac-medium.onnx` (ONNX Runtime)  
**Fallback:** sine-wave tone (if model missing)

- `piper.voice.PiperVoice.synthesize()` returns `Iterable[AudioChunk]`; `.audio_float_array` is concatenated
- Sample rate read from model config (not hard-coded)
- Viseme events extracted from audio: character → ARPAbet phoneme → 15-class viseme mapping, aligned to audio duration, RMS amplitude computed per 20 ms window
- **In-flight counter** (`_in_flight: list[int]` + `threading.Lock`) tracks sentences being synthesised or played; `pipeline_busy` cleared only when counter hits zero

### 6. Avatar Output
**File:** `ui/index.html` (Canvas 2D, browser)  
**Transport:** WebSocket on `ws://localhost:8765`

- Avatar stage in `main.py` replays timed viseme events synced to playback start timestamp
- Browser receives `{"type": "viseme", "viseme": "aa", "amplitude": 0.42}` messages
- Canvas lerps `mouth.open` and `mouth.wide` at 0.22/frame toward target viseme values
- Idle animations: head tilt sway (`sin` oscillation ±0.008 rad), breathing bob (±2.2 px), random eye wander, blink (randomised 2.5–6.3 s interval)

---

## File Structure

```
avatar voice agent/
├── main.py               # Async orchestrator, WebSocket server, CLI
├── config.py             # Typed dataclass config, logging setup
├── config.yaml           # All tunable parameters
├── audio_input.py        # VAD + microphone capture
├── stt_engine.py         # Whisper STT wrapper
���── rag_storage.py        # FAISS document index + retrieval
├── response_cache.py     # Semantic Q&A cache (fast path)
├── llm_engine.py         # Qwen LLM wrapper with sentence streaming
├── tts_engine.py         # Piper TTS + viseme extraction
├── download_models.py    # One-off model download script
├── test_pipeline.py      # Per-stage test harness
├── requirements.txt
├── config.yaml
├── docs/                 # Drop PDF or .txt files here for ingestion
├── tts_models/
│   └── en_US-lessac-medium.onnx
└── ui/
    └── index.html        # Canvas avatar + WebSocket client
```

---

## Configuration Reference

All parameters live in `config.yaml`.

```yaml
vad:
  threshold: 0.7           # Silero probability threshold (0–1)
  silence_duration_ms: 280 # Silence gap before phrase is emitted
  sample_rate: 16000
  chunk_size: 512

stt:
  model: "mlx-community/whisper-tiny.en-mlx"
  language: "en"

rag:
  docs_dir: "./docs"
  index_path: "./rag_index.faiss"
  metadata_path: "./rag_metadata.pkl"
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 300          # Characters per chunk
  chunk_overlap: 50
  top_k: 3                 # Chunks retrieved per query

llm:
  model: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  max_tokens: 80
  temperature: 0.0
  top_p: 1.0

tts:
  model_path: "./tts_models/en_US-lessac-medium.onnx"
  sample_rate: 22050        # Overridden by model's own sample rate at load
  speaking_rate: 1.0
  volume: 1.0

avatar:
  amplitude_window_ms: 20  # RMS window for lip amplitude
  output_mode: "websocket"
  ws_port: 8765

system:
  device: "mlx"
  log_level: "INFO"
  pipeline_timeout_s: 10
```

---

## Setup

### Requirements
- macOS 13+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.10–3.12
- Conda environment recommended

### Install

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
```

### Download Models

```bash
python download_models.py
```

Downloads to `~/.cache/huggingface/hub/` (Whisper, Qwen, sentence-transformers) and `./tts_models/` (Piper ONNX). Total ~2 GB. Idempotent — safe to re-run.

### Ingest Documents

Drop `.pdf` or `.txt` files into `docs/`, then:

```bash
python main.py ingest
```

Builds `rag_index.faiss` and `rag_metadata.pkl`. Re-run whenever documents change.

### Run

```bash
python main.py run
```

Opens `ui/index.html` in the default browser automatically. The WebSocket server starts on port `8765`.

---

## CLI Commands

```bash
python main.py run       # Start the full pipeline
python main.py ingest    # Index documents and exit
python main.py status    # Print model paths, device info, index status
```

---

## Testing Individual Stages

```bash
python test_pipeline.py test-vad      # Record 5 s of audio, print VAD events
python test_pipeline.py test-stt      # Transcribe a short phrase
python test_pipeline.py test-rag      # Query the FAISS index
python test_pipeline.py test-llm      # Run the LLM with a prompt, print TTFT
python test_pipeline.py test-tts      # Synthesise a sentence, play it
python test_pipeline.py test-cache    # Run a cache lookup
```

---

## Latency Budget (Apple M4 Air)

| Stage | Typical | Notes |
|---|---|---|
| VAD | ~0 ms | Runs in parallel with audio capture |
| STT | 30–50 ms | Whisper tiny, greedy |
| RAG (cache hit) | 35–50 ms | MiniLM on MPS |
| RAG (cache miss) | 80–120 ms | FAISS search + embedding |
| LLM TTFT | 500–700 ms | Qwen 1.5B 4-bit, first token |
| TTS synthesis | 80–150 ms | Piper ONNX, per sentence |
| **Total (cache hit)** | **~270 ms** | STT + cache + TTS |
| **Total (cache miss)** | **~800–1100 ms** | STT + RAG + LLM + TTS |

---

## WebSocket Message Protocol

All messages are JSON. Browser → server: no messages (receive-only client).  
Server → browser:

| `type` | Fields | Description |
|---|---|---|
| `status` | `value: "listening" \| "processing" \| "speaking" \| "idle"` | Pipeline state change |
| `transcript` | `role: "user" \| "assistant"`, `text: str` | New chat line |
| `viseme` | `viseme: str`, `amplitude: float` | Lip-sync event |
| `latency` | `stage: str`, `ms: int`, `cached?: bool` | Per-stage timing |

---

## Echo Cancellation

No hardware AEC is used. The software approach:

1. `pipeline_busy` threading event is set the moment VAD captures a phrase
2. VAD drops all subsequent phrases while `pipeline_busy` is set
3. After TTS playback finishes (sounddevice `done_callback`), a 700 ms cooldown timer runs before `pipeline_busy` is cleared
4. This prevents the microphone from picking up the speaker output

Increase `time.sleep(0.7)` in `tts_stage._delayed_clear()` if echo persists in a reverberant room.

---

## Adding Custom Q&A

Edit `response_cache.py` → `DEFAULT_QA`:

```python
{
    "questions": [
        "what is the meeting room booking process",
        "how do I book a meeting room",
    ],
    "answer": "Book meeting rooms via the Outlook calendar or the receptionist.",
},
```

The cache is rebuilt on next startup. No re-ingestion required.

---

## Dependencies

| Package | Purpose |
|---|---|
| `mlx`, `mlx-lm` | LLM inference on Apple Silicon GPU |
| `mlx-whisper` | STT on Apple Silicon GPU |
| `silero-vad` | Voice activity detection |
| `piper-tts` | ONNX-based TTS |
| `sentence-transformers` | Embeddings for RAG and cache |
| `faiss-cpu` | Vector similarity search |
| `sounddevice` | Microphone capture and audio playback |
| `onnxruntime` | ONNX model execution (TTS) |
| `websockets` | WebSocket server for browser communication |
| `rich` | Terminal logging and status panel |
| `click` | CLI interface |
| `pypdf` | PDF document ingestion |
