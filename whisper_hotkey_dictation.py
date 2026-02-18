#!/usr/bin/env python3
"""Global hotkey dictation using OpenAI Whisper.

Press the configured hotkey to toggle dictation on/off.
While on, audio is captured continuously from the microphone and chunked by silence.
Each finalized speech chunk is transcribed with OpenAI Whisper and typed at the
current cursor location.
"""

from __future__ import annotations

import argparse
import platform
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard


def normalize_hotkey(hotkey: str) -> str:
    """Normalize common hotkey spellings into pynput GlobalHotKeys format."""
    aliases = {
        "control": "ctrl",
        "option": "alt",
        "command": "cmd",
        "return": "enter",
        "escape": "esc",
        "spacebar": "space",
    }
    named_keys = {
        "alt",
        "alt_l",
        "alt_r",
        "backspace",
        "caps_lock",
        "cmd",
        "cmd_l",
        "cmd_r",
        "ctrl",
        "ctrl_l",
        "ctrl_r",
        "delete",
        "down",
        "end",
        "enter",
        "esc",
        "home",
        "insert",
        "left",
        "menu",
        "page_down",
        "page_up",
        "right",
        "shift",
        "shift_l",
        "shift_r",
        "space",
        "tab",
        "up",
    }

    normalized_parts: List[str] = []
    for raw_part in hotkey.split("+"):
        part = raw_part.strip()
        if not part:
            continue

        wrapped = part.startswith("<") and part.endswith(">")
        name = part[1:-1].strip().lower() if wrapped else part.lower()
        name = aliases.get(name, name)

        is_function_key = (
            len(name) >= 2 and name.startswith("f") and name[1:].isdigit()
        )

        if wrapped or name in named_keys or is_function_key:
            normalized_parts.append(f"<{name}>")
            continue

        normalized_parts.append(part)

    return "+".join(normalized_parts)


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
    system_name = platform.system().lower()
    if system_name == "darwin":
        return "<cmd>+<shift>+g"
    if system_name == "windows":
        return "<alt>+<ctrl>+<shift>+m"
    return "<ctrl>+<alt>+<space>"


@dataclass
class DictationConfig:
    hotkey: str
    model: str
    language: Optional[str]
    sample_rate: int
    block_duration: float
    silence_threshold_db: float
    silence_duration: float
    min_speech_duration: float
    max_speech_duration: float
    pre_speech_duration: float
    device: Optional[int]


class WhisperHotkeyDictation:
    def __init__(self, config: DictationConfig) -> None:
        self.config = config
        self.enabled = threading.Event()
        self.stopping = threading.Event()
        self.segment_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self.state_lock = threading.Lock()

        self.pre_speech_chunks: Deque[np.ndarray] = deque(
            maxlen=max(1, int(config.pre_speech_duration / config.block_duration))
        )
        self.current_chunks: List[np.ndarray] = []
        self.speaking = False
        self.speech_started_at = 0.0
        self.last_voice_at = 0.0

        self.keyboard_controller = keyboard.Controller()
        self.worker = threading.Thread(
            target=self._transcription_worker, name="transcription-worker", daemon=True
        )

        print(f"[model] loading OpenAI Whisper model '{config.model}'...")
        self.model = whisper.load_model(config.model)
        print("[model] ready")

    @staticmethod
    def _chunk_db(chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)

    def _audio_callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        del frames, t
        if status:
            print(f"[audio] {status}", flush=True)
        if not self.enabled.is_set() or self.stopping.is_set():
            return

        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        level_db = self._chunk_db(mono)
        voiced = level_db >= self.config.silence_threshold_db
        now = time.monotonic()

        with self.state_lock:
            self.pre_speech_chunks.append(mono)
            if voiced:
                if not self.speaking:
                    self.speaking = True
                    self.speech_started_at = now
                    self.current_chunks = list(self.pre_speech_chunks)
                self.current_chunks.append(mono)
                self.last_voice_at = now
                if now - self.speech_started_at >= self.config.max_speech_duration:
                    self._finalize_segment_locked()
                return

            if self.speaking:
                self.current_chunks.append(mono)
                if now - self.last_voice_at >= self.config.silence_duration:
                    self._finalize_segment_locked()

    def _reset_segment_state_locked(self) -> None:
        self.current_chunks = []
        self.speaking = False
        self.speech_started_at = 0.0
        self.last_voice_at = 0.0

    def _finalize_segment_locked(self) -> None:
        if not self.current_chunks:
            self._reset_segment_state_locked()
            return

        merged = np.concatenate(self.current_chunks, axis=0)
        duration = len(merged) / float(self.config.sample_rate)
        if duration >= self.config.min_speech_duration:
            self.segment_queue.put(merged)
        self._reset_segment_state_locked()

    def _flush_active_segment(self) -> None:
        with self.state_lock:
            if self.speaking and self.current_chunks:
                self._finalize_segment_locked()
            else:
                self._reset_segment_state_locked()

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(
            audio=audio,
            language=self.config.language,
            task="transcribe",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()

    def _type_text(self, text: str) -> None:
        if not text:
            return
        payload = text if text.endswith((" ", "\n", "\t")) else f"{text} "
        self.keyboard_controller.type(payload)

    def _transcription_worker(self) -> None:
        while not self.stopping.is_set():
            try:
                item = self.segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.segment_queue.task_done()
                break

            try:
                transcript = self._transcribe_chunk(item)
                if transcript:
                    self._type_text(transcript)
                    print(f"[transcript] {transcript}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] transcription failed: {exc}", flush=True)
            finally:
                self.segment_queue.task_done()

    def toggle(self) -> None:
        if self.enabled.is_set():
            self.enabled.clear()
            self._flush_active_segment()
            print("[dictation] OFF", flush=True)
        else:
            self.enabled.set()
            print("[dictation] ON", flush=True)

    def stop(self) -> None:
        self.stopping.set()
        self.enabled.clear()
        self._flush_active_segment()
        self.segment_queue.put(None)

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


def parse_args() -> DictationConfig:
    default_hotkey = platform_default_hotkey()
    parser = argparse.ArgumentParser(
        description="Global hotkey dictation tool using OpenAI Whisper."
    )
    parser.add_argument(
        "--hotkey",
        default=default_hotkey,
        type=validate_hotkey,
        help=(
            "Global hotkey to toggle dictation "
            f"(default on this OS: '{default_hotkey}')."
        ),
    )
    parser.add_argument(
        "--model",
        default="base",
        help="OpenAI Whisper model name (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code like 'en'. If omitted, Whisper auto-detects.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate.",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.10,
        help="Audio callback chunk size in seconds.",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-38.0,
        help="Voice activity threshold in dBFS (higher is more sensitive).",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.80,
        help="Silence length (seconds) used to close a speech chunk.",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=0.25,
        help="Ignore chunks shorter than this duration (seconds).",
    )
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=12.0,
        help="Force chunk flush when speech exceeds this duration (seconds).",
    )
    parser.add_argument(
        "--pre-speech-duration",
        type=float,
        default=0.20,
        help="Retain this much audio before speech starts (seconds).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional input device index from `python -m sounddevice`.",
    )

    args = parser.parse_args()
    return DictationConfig(
        hotkey=args.hotkey,
        model=args.model,
        language=args.language,
        sample_rate=args.sample_rate,
        block_duration=args.block_duration,
        silence_threshold_db=args.silence_threshold_db,
        silence_duration=args.silence_duration,
        min_speech_duration=args.min_speech_duration,
        max_speech_duration=args.max_speech_duration,
        pre_speech_duration=args.pre_speech_duration,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    app = WhisperHotkeyDictation(config)
    app.run()


if __name__ == "__main__":
    main()
