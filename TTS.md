# Conversational Speech Backend Architecture

## Goal

Build a realtime conversational AI voice system that feels emotionally alive like ChatGPT Voice while remaining CPU-friendly.

The real problem is NOT TTS.

The real problem is:

```txt
speech planning
```

Most TTS systems do:

```txt
text → waveform
```

But advanced conversational systems do:

```txt
intent
→ emotion analysis
→ pacing planning
→ prosody planning
→ speech generation
```

---

# Core Architecture

```txt
Client
  ↓
Realtime Gateway
  ↓
Conversation Engine
  ↓
Speech Planning Engine
  ↓
Streaming TTS Engine
  ↓
Audio Stream
```

---

# 1. Realtime Gateway

## Purpose

* websocket handling
* audio streaming
* interruption handling
* low latency communication
* session management

## Stack

```txt
FastAPI
WebSocket
uvloop
asyncio
```

## Responsibilities

```txt
mic audio in
partial transcripts
audio out
interruptions
session state
```

---

# 2. Conversation Engine

## Purpose

* manages LLM interaction
* conversational memory
* speaking mode selection
* response orchestration

## Modes

```txt
interviewer
mentor
teacher
friend
technical_explainer
```

## Output

NEVER send raw LLM text directly to TTS.

Instead:

```json
{
  "text": "I don't think this is a good idea.",
  "intent": "analysis",
  "emotion": "thoughtful",
  "priority": "normal"
}
```

---

# 3. Speech Planning Engine (MOST IMPORTANT)

This is the actual moat.

## Modules

```txt
SemanticChunker
EmotionDetector
ProsodyPlanner
PausePlanner
BreathingPlanner
SpeechRewriter
```

## Pipeline

```txt
raw llm text
 ↓
rewrite for speech
 ↓
semantic chunking
 ↓
emotion tagging
 ↓
prosody planning
 ↓
speech segments
```

## Example Output

```json
{
  "segments": [
    {
      "text": "I don't think this is a good idea...",
      "emotion": "concerned",
      "speed": 0.84,
      "pause_after_ms": 220
    },
    {
      "text": "to move in this direction of the problem.",
      "emotion": "analytical",
      "speed": 0.96,
      "pause_after_ms": 140
    }
  ]
}
```

This layer matters MORE than the TTS model.

---

# Semantic Chunking

Humans do NOT speak in full paragraphs.

Bad:

```txt
I don't think this is a good idea to move in this direction of the problem.
```

Good:

```txt
I don't think this is a good idea...

to move in this direction of the problem.
```

This alone massively improves realism.

---

# Dynamic Speech Behavior

## Speed Variation

Humans constantly change speaking speed.

Example:

```python
[
  ("Well...", 0.82),
  ("I don't think this is a good idea...", 0.87),
  ("Can you elaborate a little bit more?", 0.94)
]
```

## Silence / Breathing

Tiny pauses make speech feel alive.

Example:

```python
pause = np.zeros(int(24000 * 0.18))
```

## Emotion Mapping

Map:

```txt
concern
uncertainty
excitement
teaching
storytelling
empathy
analysis
```

to:

```txt
speed
pause
pitch
emphasis
chunk size
```

---

# 4. Streaming TTS Engine

DO NOT synthesize full paragraphs.

Do:

```txt
segment-by-segment synthesis
```

## Flow

```txt
speech segment
 ↓
TTS worker
 ↓
audio chunk
 ↓
stream immediately
```

This massively reduces perceived latency.

---

# 5. Audio Streamer

## Responsibilities

* stitch audio chunks
* inject silence
* smooth transitions
* realtime playback

Tiny silence gaps are extremely important.

---

# Recommended Models

| Model      | CPU       | Naturalness | Speed     |
| ---------- | --------- | ----------- | --------- |
| Kokoro     | Excellent | Medium      | Fast      |
| Piper      | Excellent | Medium+     | Very Fast |
| XTTS v2    | Okay      | High        | Slow      |
| StyleTTS2  | Poor      | Very High   | Very Slow |
| Parler-TTS | Moderate  | High        | Slow      |

---

# Hardware Plan

## SOLO

```txt
Intel Core 2 Duo
~2GB RAM
```

Use:

```txt
Rule-based SPP
+
Piper or Kokoro
```

Avoid:

```txt
XTTS
Transformers
Large Ollama models
```

SOLO should act as:

```txt
fallback inference node
```

---

## LUKE

```txt
AMD A6
~4GB RAM
```

Use:

```txt
HybridSPP
+
Kokoro
```

or:

```txt
OllamaSPP
+
Kokoro
```

Recommended Ollama models:

```txt
qwen2.5:1.5b
phi3-mini
```

LUKE becomes:

```txt
main conversational node
```

---

# Production Architecture

```txt
Client
 ↓
FastAPI Gateway
 ↓
Redis Streams
 ↓
Conversation Workers
 ↓
Speech Planner Workers
 ↓
TTS Workers
 ↓
Audio Stream
```

---

# Worker Separation

Separate:

```txt
LLM workers
SPP workers
TTS workers
```

Never combine all into one process.

Reason:

```txt
tts blocking kills latency
```

---

# Queue-Based Pipeline

Use:

```txt
Redis Streams
asyncio.Queue
```

Flow:

```txt
text arrives
 ↓
planner queue
 ↓
tts queue
 ↓
audio queue
```

This enables:

```txt
streaming
interruptions
cancellation
parallel synthesis
```

---

# Recommended MVP

## Phase 1

```txt
FastAPI websocket
rule-based SPP
Kokoro streaming
pause injection
semantic chunking
```

## Phase 2

```txt
emotion detection
dynamic speed
breathing planner
```

## Phase 3

```txt
local ollama spp
realtime interruption
adaptive conversation modes
```

## Phase 4

```txt
custom speech-token model
```

---

# Biggest Insight

Do NOT focus mainly on:

```txt
better tts models
```

Focus on:

```txt
better conversational speech orchestration
```

That is what actually creates OpenAI-level realism.

---

# Final Vision

```txt
LLM
 ↓
Speech Planner
 ↓
Emotion + Prosody Engine
 ↓
Streaming TTS
 ↓
Realtime Conversational Audio
```

The speech planner is the most important component in the entire system.
