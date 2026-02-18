# Whisper Hotkey Dictation

Global hotkey dictation app built with OpenAI's official Whisper model (`openai-whisper`).

## What it does

- Runs in the background.
- Uses a global hotkey as toggle (`ON/OFF`).
- While `ON`, captures mic audio continuously.
- Uses a rolling audio window and runs Whisper repeatedly for near-live updates.
- Force-flushes the current partial segment when you pause.
- Types the transcribed text into whichever app currently has cursor focus.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. macOS permissions (required):
- Microphone access for your terminal/Python runtime.
- Accessibility access for your terminal/Python runtime (needed to type into other apps).

4. If needed, list audio devices:

```bash
python -m sounddevice
```

## Run

```bash
python whisper_hotkey_dictation.py --model base
```

Default hotkeys:

- macOS: `Command + Shift + G` (`<cmd>+<shift>+g`)
- Windows: `Alt + Ctrl + Shift + M` (`<alt>+<ctrl>+<shift>+m`)
- Linux fallback: `<ctrl>+<alt>+<space>`

## Useful flags

- `--hotkey "<cmd>+<shift>+g"`: override global toggle hotkey.
- `--model base`: Whisper model (`tiny`, `base`, `small`, `medium`, `large`).
- `--language en`: fixed language (otherwise auto-detect).
- `--device 0`: input device index.
- `--live-interval 0.35`: seconds between rolling transcription passes.
- `--min-chunk-secs 0.45`: minimum unprocessed audio before each transcription pass.
- `--pause-silence-secs 0.30`: seconds of silence before force-flush.
- `--trailing-silence-secs 0.20`: finalize last segment when tail silence is present.
- `--silence-threshold-db -36`: voice activity threshold for pause detection.
- `--same-out-limit 2`: repeated partial outputs before committing.

## Notes

- This is rolling near-live dictation: text should appear while speaking and flush quickly on pauses.
- Larger Whisper models improve quality but increase latency and CPU/GPU use.
