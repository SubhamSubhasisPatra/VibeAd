#!/usr/bin/env python3
"""Global hotkey dictation using OpenAI Whisper - near real-time via rolling window.

Inspired by collabora/WhisperLive: instead of waiting for silence, audio is
accumulated into a rolling buffer and transcribed every second. Whisper returns
multiple segments; all but the last are considered "complete" and typed immediately.
The last segment is held back until Whisper stops changing it (same-output guard).
"""

from __future__ import annotations

import argparse
import platform
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard


# ---------------------------------------------------------------------------
# Hotkey helpers (unchanged)
# ---------------------------------------------------------------------------

def normalize_hotkey(hotkey: str) -> str:
    aliases = {
        "control": "ctrl", "option": "alt", "command": "cmd",
        "return": "enter", "escape": "esc", "spacebar": "space",
    }
    named_keys = {
        "alt", "alt_l", "alt_r", "backspace", "caps_lock", "cmd", "cmd_l",
        "cmd_r", "ctrl", "ctrl_l", "ctrl_r", "delete", "down", "end",
        "enter", "esc", "home", "insert", "left", "menu", "page_down",
        "page_up", "right", "shift", "shift_l", "shift_r", "space", "tab", "up",
    }
    parts: List[str] = []
    for raw in hotkey.split("+"):
        part = raw.strip()
        if not part:
            continue
        wrapped = part.startswith("<") and part.endswith(">")
        name = part[1:-1].strip().lower() if wrapped else part.lower()
        name = aliases.get(name, name)
        is_fn = len(name) >= 2 and name.startswith("f") and name[1:].isdigit()
        if wrapped or name in named_keys or is_fn:
            parts.append(f"<{name}>")
        else:
            parts.append(part)
    return "+".join(parts)


def validate_hotkey(hotkey: str) -> str:
    normalized = normalize_hotkey(hotkey)
    try:
        keyboard.HotKey.parse(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid hotkey '{hotkey}'. Use format like '<ctrl>+<alt>+<space>'"
        ) from exc
    return normalized


def platform_default_hotkey() -> str:
    s = platform.system().lower()
    if s == "darwin":
        return "<cmd>+<shift>+g"
    if s == "windows":
        return "<alt>+<ctrl>+<shift>+m"
    return "<ctrl>+<alt>+<space>"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DictationConfig:
    hotkey: str
    model: str
    language: Optional[str]
    sample_rate: int
    block_duration: float   # audio callback chunk size in seconds
    no_speech_thresh: float # discard segments with no_speech_prob above this
    same_out_limit: int     # repeat count before forcing a partial segment out
    max_buffer_secs: float  # rolling buffer max length before trimming
    device: Optional[int]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WhisperHotkeyDictation:
    TRIM_SECS = 10  # how much to trim when buffer gets too long

    def __init__(self, config: DictationConfig) -> None:
        self.config = config
        self.enabled = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()

        # Rolling audio buffer (collabora/WhisperLive style)
        self.frames_np: Optional[np.ndarray] = None
        self.frames_offset = 0.0    # seconds that have been trimmed off the front
        self.timestamp_offset = 0.0 # seconds of audio confirmed as transcribed

        # Same-output guard: if Whisper emits the same last segment N times in a
        # row, treat it as confirmed and type it.
        self.prev_out = ""
        self.same_out_count = 0
        self.end_time_for_same_out: Optional[float] = None

        self.keyboard_controller = keyboard.Controller()
        self.worker = threading.Thread(
            target=self._transcription_worker, name="transcription-worker", daemon=True
        )

        print(f"[model] loading Whisper model '{config.model}'...")
        self.model = whisper.load_model(config.model)
        print("[model] ready")

    # ------------------------------------------------------------------
    # Audio callback – just fill the rolling buffer
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        if not self.enabled.is_set() or self.stopping.is_set():
            return

        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        max_samples = int(self.config.max_buffer_secs * self.config.sample_rate)
        trim_samples = int(self.TRIM_SECS * self.config.sample_rate)

        with self.lock:
            if self.frames_np is None:
                self.frames_np = mono
            else:
                self.frames_np = np.concatenate((self.frames_np, mono))

            # Trim the oldest audio when the buffer grows too large
            if len(self.frames_np) > max_samples:
                self.frames_offset += self.TRIM_SECS
                self.frames_np = self.frames_np[trim_samples:]
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

    # ------------------------------------------------------------------
    # Rolling-window transcription worker
    # ------------------------------------------------------------------

    def _get_unprocessed_chunk(self) -> Optional[np.ndarray]:
        """Return audio after timestamp_offset, or None if too short."""
        with self.lock:
            if self.frames_np is None:
                return None
            start = int(max(0, self.timestamp_offset - self.frames_offset) * self.config.sample_rate)
            chunk = self.frames_np[start:].copy()
        if len(chunk) < self.config.sample_rate:  # need at least 1 second
            return None
        return chunk

    def _transcription_worker(self) -> None:
        while not self.stopping.is_set():
            if not self.enabled.is_set():
                time.sleep(0.1)
                continue

            chunk = self._get_unprocessed_chunk()
            if chunk is None:
                time.sleep(0.1)
                continue

            try:
                result = self.model.transcribe(
                    chunk,
                    language=self.config.language,
                    task="transcribe",
                    fp16=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
            except Exception as exc:
                print(f"[error] transcription: {exc}", flush=True)
                time.sleep(0.1)
                continue

            segments = result.get("segments", [])
            if not segments:
                time.sleep(0.25)
                continue

            self._process_segments(segments, len(chunk) / self.config.sample_rate)

    def _process_segments(self, segments: list, duration: float) -> None:
        """
        Core algorithm from WhisperLive/base.py:
        - All segments except the last are "complete" — type them and advance offset.
        - The last segment is "live" (may grow) — hold it until it stabilises.
        """
        new_text = ""
        offset: Optional[float] = None

        # Complete segments (everything except the last)
        if len(segments) > 1:
            last_no_speech = segments[-1].get("no_speech_prob", 0)
            if last_no_speech <= self.config.no_speech_thresh:
                for seg in segments[:-1]:
                    if seg.get("no_speech_prob", 0) <= self.config.no_speech_thresh:
                        new_text += seg["text"]
                    offset = min(duration, seg["end"])

        # Live / last segment
        last = segments[-1]
        current_out = ""
        if last.get("no_speech_prob", 0) <= self.config.no_speech_thresh:
            current_out = last["text"]

        # Same-output guard: if Whisper keeps saying the same thing, confirm it
        if current_out.strip() and current_out.strip() == self.prev_out.strip():
            self.same_out_count += 1
            if self.end_time_for_same_out is None:
                self.end_time_for_same_out = last["end"]

            if self.same_out_count >= self.config.same_out_limit:
                new_text += current_out
                offset = min(duration, self.end_time_for_same_out)
                self.same_out_count = 0
                self.end_time_for_same_out = None
                current_out = ""
        else:
            self.same_out_count = 0
            self.end_time_for_same_out = None

        self.prev_out = current_out

        # Type confirmed text and advance offset
        if new_text.strip():
            text = new_text.strip()
            payload = text if text.endswith((" ", "\n", "\t")) else f"{text} "
            self.keyboard_controller.type(payload)
            print(f"[transcript] {text}", flush=True)

        if offset is not None:
            self.timestamp_offset += offset

    # ------------------------------------------------------------------
    # Hotkey toggle
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self.enabled.is_set():
            self.enabled.clear()
            # Final pass: transcribe whatever is left in the buffer
            chunk = self._get_unprocessed_chunk()
            if chunk is not None:
                try:
                    result = self.model.transcribe(
                        chunk, language=self.config.language,
                        fp16=False, temperature=0.0, condition_on_previous_text=False,
                    )
                    text = (result.get("text") or "").strip()
                    if text and text.strip() != self.prev_out.strip():
                        payload = text if text.endswith((" ", "\n", "\t")) else f"{text} "
                        self.keyboard_controller.type(payload)
                        print(f"[transcript] {text}", flush=True)
                except Exception as exc:
                    print(f"[error] final flush: {exc}", flush=True)
            self._reset()
            print("[dictation] OFF", flush=True)
        else:
            self._reset()
            self.enabled.set()
            print("[dictation] ON", flush=True)

    def _reset(self) -> None:
        with self.lock:
            self.frames_np = None
            self.frames_offset = 0.0
            self.timestamp_offset = 0.0
        self.prev_out = ""
        self.same_out_count = 0
        self.end_time_for_same_out = None

    def stop(self) -> None:
        self.stopping.set()
        self.enabled.clear()

    def run(self) -> None:
        self.worker.start()
        hotkeys = keyboard.GlobalHotKeys({self.config.hotkey: self.toggle})
        hotkeys.start()

        blocksize = max(1, int(self.config.sample_rate * self.config.block_duration))
        print(
            f"[ready] Press {self.config.hotkey} to toggle dictation. Ctrl+C to quit.",
            flush=True,
        )
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=blocksize,
                device=self.config.device,
            ):
                while not self.stopping.is_set():
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[exit] stopping...", flush=True)
        finally:
            self.stop()
            hotkeys.stop()
            self.worker.join(timeout=5.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> DictationConfig:
    default_hotkey = platform_default_hotkey()
    parser = argparse.ArgumentParser(
        description="Global hotkey dictation using OpenAI Whisper (rolling-window near-realtime)."
    )
    parser.add_argument("--hotkey", default=default_hotkey, type=validate_hotkey,
                        help=f"Toggle hotkey (default: '{default_hotkey}').")
    parser.add_argument("--model", default="base",
                        help="Whisper model: tiny, base, small, medium, large.")
    parser.add_argument("--language", default=None,
                        help="Language code, e.g. 'en'. Omit for auto-detect.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-duration", type=float, default=0.10,
                        help="Audio callback chunk size in seconds.")
    parser.add_argument("--no-speech-thresh", type=float, default=0.4,
                        help="Discard segments with no_speech_prob above this (0-1).")
    parser.add_argument("--same-out-limit", type=int, default=5,
                        help="Repeated identical outputs before forcing segment out.")
    parser.add_argument("--max-buffer-secs", type=float, default=30.0,
                        help="Rolling buffer max length before trimming old audio.")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index from `python -m sounddevice`.")

    args = parser.parse_args()
    return DictationConfig(
        hotkey=args.hotkey,
        model=args.model,
        language=args.language,
        sample_rate=args.sample_rate,
        block_duration=args.block_duration,
        no_speech_thresh=args.no_speech_thresh,
        same_out_limit=args.same_out_limit,
        max_buffer_secs=args.max_buffer_secs,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    app = WhisperHotkeyDictation(config)
    app.run()


if __name__ == "__main__":
    main()
