# WaveTec InfoPoint — Voice Avatar Kiosk

An interactive voice assistant kiosk that listens, understands, and responds in real-time through an animated avatar — entirely offline, running locally on Apple Silicon.

---

## What It Does

Visitors walk up and speak naturally. The kiosk hears them, understands the question, and the avatar speaks a relevant answer back — no internet required, no cloud, no subscription.

**A typical interaction takes under one second from end of speech to first spoken word.**

---

## How It Works (From a Visitor's Perspective)

1. **Walk up and speak** — the kiosk listens continuously and detects when someone starts talking
2. **Your words appear on screen** — live transcript of what you said
3. **The avatar responds** — it animates its mouth in sync with the spoken answer
4. **The conversation stays on screen** — a running chat history shows what was said

---

## What the Avatar Can Help With

Out of the box the kiosk is configured for a WaveTec InfoPoint lobby, covering:

| Topic | Example Questions |
|---|---|
| Opening hours | "When do you open?" / "Are you open weekends?" |
| Location | "Where is the office?" / "What floor are you on?" |
| Services | "What can you help me with?" / "What is this kiosk for?" |
| Wi-Fi | "What is the Wi-Fi password?" |
| Parking | "Where can I park?" / "Is there free parking?" |
| Contact | "How do I reach reception?" |
| Restrooms | "Where are the toilets?" |
| Events | "What is happening today?" |
| Visitor sign-in | "I have an appointment" / "How do I check in?" |
| Accessibility | "Is there wheelchair access?" / "Where is the lift?" |

For anything not in the above list, the assistant searches the documents you provide (PDF or text files) and generates a relevant answer.

---

## Key Highlights

- **Fully offline** — no internet connection needed after initial setup
- **Fast** — common questions answered in ~270 ms; general questions in under 1 second
- **Private** — no audio or data leaves the device
- **Customisable** — swap in your own documents and the assistant learns from them
- **Visual** — animated avatar with lip-sync makes it feel like a real conversation
- **Self-contained** — one command to start, opens automatically in the browser

---

## Running It

```bash
# First time: download all models (one-off, ~2 GB)
python download_models.py

# Add your documents (PDF or .txt files)
cp your-docs.pdf docs/
python main.py ingest

# Start the kiosk
python main.py run
```

The browser opens automatically. Speak to the avatar.

---

## Hardware Requirements

- **Mac with Apple Silicon** (M1, M2, M3, M4 — any variant)
- 8 GB RAM minimum, 16 GB recommended
- Microphone and speakers
- macOS 13 or later

---

## What Is Shown on the Display

The kiosk screen shows three things at once:

- **The animated avatar** — a 3D-style character that moves its mouth, blinks, and looks around naturally
- **Pipeline timing** — small real-time indicators showing how long each step (speech recognition, search, AI generation, voice synthesis) took in milliseconds
- **Conversation history** — a scrollable chat showing everything that was said

---

*Built for WaveTec InfoPoint · Runs entirely on-device · No cloud required*
