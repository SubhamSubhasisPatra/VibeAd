#!/usr/bin/env python3
"""Global hotkey dictation using OpenAI Whisper with rolling near-live updates.

While dictation is ON:
- audio is appended into a rolling buffer;
- a background worker transcribes only the unconfirmed tail every interval;
- confirmed segments are typed immediately;
- on pause, the current partial segment is force-flushed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard


def normalize_hotkey(hotkey: str) -> str:
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
    model_cache_dir: Optional[str]
    offline_strict: bool
    print_transcripts: bool
    language: Optional[str]
    sample_rate: int
    block_duration: float
    live_interval: float
    min_chunk_secs: float
    pause_silence_secs: float
    trailing_silence_secs: float
    silence_threshold_db: float
    no_speech_thresh: float
    same_out_limit: int
    max_buffer_secs: float
    trim_secs: float
    device: Optional[int]


class WhisperHotkeyDictation:
    def __init__(self, config: DictationConfig) -> None:
        self.config = config
        self.enabled = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.transcribe_lock = threading.Lock()

        self.buffer_chunks: Deque[np.ndarray] = deque()
        self.buffer_samples = 0
        self.buffer_start_time = 0.0
        self.timestamp_offset = 0.0
        self.last_voice_ts = 0.0

        self.prev_out = ""
        self.same_out_count = 0

        self.keyboard_controller = keyboard.Controller()
        self.worker = threading.Thread(
            target=self._transcription_worker,
            name="transcription-worker",
            daemon=True,
        )

        if self.config.offline_strict:
            self._assert_model_available_offline()

        print(f"[model] loading OpenAI Whisper model '{config.model}'...")
        self.model = whisper.load_model(
            config.model,
            download_root=config.model_cache_dir,
        )
        print("[model] ready")

    @staticmethod
    def _default_whisper_cache_dir() -> str:
        base_cache = os.path.join(os.path.expanduser("~"), ".cache")
        root = os.getenv("XDG_CACHE_HOME", base_cache)
        return os.path.join(root, "whisper")

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _assert_model_available_offline(self) -> None:
        if os.path.isfile(self.config.model):
            return

        model_map = getattr(whisper, "_MODELS", {})
        model_url = model_map.get(self.config.model)
        if not model_url:
            raise RuntimeError(
                "offline strict mode requires either a local model file path or an "
                f"official whisper model name. received '{self.config.model}'."
            )

        model_cache_dir = self.config.model_cache_dir or self._default_whisper_cache_dir()
        expected_sha256 = model_url.split("/")[-2]
        model_filename = os.path.basename(model_url)
        cached_model_path = os.path.join(model_cache_dir, model_filename)

        if not os.path.isfile(cached_model_path):
            raise RuntimeError(
                "offline strict mode blocked startup: model file not found at "
                f"'{cached_model_path}'. pre-download it before running."
            )

        actual_sha256 = self._file_sha256(cached_model_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "offline strict mode blocked startup: cached model checksum mismatch "
                f"for '{cached_model_path}'. expected {expected_sha256}, "
                f"got {actual_sha256}."
            )

    @staticmethod
    def _chunk_db(chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)

    def _drop_oldest_samples_locked(self, samples_to_drop: int) -> None:
        remaining = samples_to_drop
        while remaining > 0 and self.buffer_chunks:
            head = self.buffer_chunks[0]
            head_len = len(head)
            if head_len <= remaining:
                self.buffer_chunks.popleft()
                self.buffer_samples -= head_len
                self.buffer_start_time += head_len / float(self.config.sample_rate)
                remaining -= head_len
            else:
                self.buffer_chunks[0] = head[remaining:]
                self.buffer_samples -= remaining
                self.buffer_start_time += remaining / float(self.config.sample_rate)
                remaining = 0

        if self.timestamp_offset < self.buffer_start_time:
            self.timestamp_offset = self.buffer_start_time

    def _audio_callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        del frames, t
        if status:
            print(f"[audio] {status}", flush=True)
        if not self.enabled.is_set() or self.stopping.is_set():
            return

        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        voiced = self._chunk_db(mono) >= self.config.silence_threshold_db

        max_samples = int(self.config.max_buffer_secs * self.config.sample_rate)
        trim_samples = int(self.config.trim_secs * self.config.sample_rate)
        now = time.monotonic()

        with self.lock:
            self.buffer_chunks.append(mono)
            self.buffer_samples += len(mono)
            if voiced:
                self.last_voice_ts = now

            while self.buffer_samples > max_samples and trim_samples > 0:
                self._drop_oldest_samples_locked(trim_samples)

    def _snapshot_unprocessed_chunk(
        self, min_duration: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        min_secs = self.config.min_chunk_secs if min_duration is None else min_duration
        with self.lock:
            if self.buffer_samples == 0:
                return None
            start_sec = max(self.timestamp_offset, self.buffer_start_time)
            start_sample = int(
                max(0.0, start_sec - self.buffer_start_time) * self.config.sample_rate
            )
            full = np.concatenate(list(self.buffer_chunks), axis=0)
            if start_sample >= len(full):
                return None
            chunk = full[start_sample:].copy()

        duration = len(chunk) / float(self.config.sample_rate)
        if duration < min_secs:
            return None
        return chunk, duration

    def _is_paused(self) -> bool:
        with self.lock:
            last_voice = self.last_voice_ts
        if last_voice <= 0:
            return False
        return (time.monotonic() - last_voice) >= self.config.pause_silence_secs

    def _advance_timestamp(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self.lock:
            self.timestamp_offset += seconds
            if self.timestamp_offset < self.buffer_start_time:
                self.timestamp_offset = self.buffer_start_time

    def _type_text(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        payload = clean if clean.endswith((" ", "\n", "\t")) else f"{clean} "
        self.keyboard_controller.type(payload)
        if self.config.print_transcripts:
            print(f"[transcript] {clean}", flush=True)

    def _process_segments(
        self,
        segments: List[dict],
        duration: float,
        force_finalize_last: bool,
    ) -> None:
        if not segments:
            return

        confirmed_parts: List[str] = []
        confirmed_end = 0.0

        for seg in segments[:-1]:
            seg_end = float(seg.get("end", 0.0))
            confirmed_end = max(confirmed_end, min(duration, seg_end))
            if float(seg.get("no_speech_prob", 0.0)) <= self.config.no_speech_thresh:
                seg_text = (seg.get("text") or "").strip()
                if seg_text:
                    confirmed_parts.append(seg_text)

        last = segments[-1]
        last_end = float(last.get("end", duration))
        last_end = max(0.0, min(duration, last_end))
        trailing_silence = max(0.0, duration - last_end)
        finalize_last = (
            force_finalize_last
            or trailing_silence >= self.config.trailing_silence_secs
        )

        last_text = ""
        last_no_speech = float(last.get("no_speech_prob", 0.0))
        if (
            last_no_speech <= self.config.no_speech_thresh
            or finalize_last
        ):
            last_text = (last.get("text") or "").strip()

        if finalize_last and last_text:
            confirmed_parts.append(last_text)
            confirmed_end = max(confirmed_end, last_end)
            self.prev_out = ""
            self.same_out_count = 0
        elif last_text:
            if last_text == self.prev_out:
                self.same_out_count += 1
                if self.same_out_count >= self.config.same_out_limit:
                    confirmed_parts.append(last_text)
                    confirmed_end = max(confirmed_end, last_end)
                    self.prev_out = ""
                    self.same_out_count = 0
            else:
                self.prev_out = last_text
                self.same_out_count = 0
        elif (last.get("text") or "").strip():
            raw_last_text = (last.get("text") or "").strip()
            if raw_last_text == self.prev_out:
                self.same_out_count += 1
                # WhisperLive-like guard for sticky one-word segments.
                if self.same_out_count >= self.config.same_out_limit + 1:
                    confirmed_parts.append(raw_last_text)
                    confirmed_end = max(confirmed_end, last_end)
                    self.prev_out = ""
                    self.same_out_count = 0
            else:
                self.prev_out = raw_last_text
                self.same_out_count = 0
        else:
            self.prev_out = ""
            self.same_out_count = 0
            if finalize_last:
                confirmed_end = max(confirmed_end, last_end)

        if confirmed_parts:
            self._type_text(" ".join(confirmed_parts))

        if confirmed_end > 0:
            self._advance_timestamp(confirmed_end)

    def _transcribe_chunk(self, audio: np.ndarray) -> dict:
        with self.transcribe_lock:
            return self.model.transcribe(
                audio,
                language=self.config.language,
                task="transcribe",
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=False,
            )

    def _transcription_worker(self) -> None:
        next_tick = time.monotonic()
        while not self.stopping.is_set():
            if not self.enabled.is_set():
                next_tick = time.monotonic()
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick = now + self.config.live_interval

            snapshot = self._snapshot_unprocessed_chunk()
            if snapshot is None:
                continue

            chunk, duration = snapshot
            force_finalize_last = self._is_paused()
            try:
                result = self._transcribe_chunk(chunk)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] transcription failed: {exc}", flush=True)
                continue

            if not self.enabled.is_set():
                continue

            segments = result.get("segments") or []
            if not segments:
                if force_finalize_last:
                    self._advance_timestamp(duration)
                continue

            self._process_segments(segments, duration, force_finalize_last)

    def _flush_remaining(self) -> None:
        snapshot = self._snapshot_unprocessed_chunk(min_duration=0.15)
        if snapshot is None:
            return

        chunk, duration = snapshot
        try:
            result = self._transcribe_chunk(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] final flush failed: {exc}", flush=True)
            return

        final_text = (result.get("text") or "").strip()
        if final_text:
            self._type_text(final_text)
        self._advance_timestamp(duration)

    def _reset_state(self) -> None:
        with self.lock:
            self.buffer_chunks.clear()
            self.buffer_samples = 0
            self.buffer_start_time = 0.0
            self.timestamp_offset = 0.0
            self.last_voice_ts = 0.0
        self.prev_out = ""
        self.same_out_count = 0

    def toggle(self) -> None:
        if self.enabled.is_set():
            self.enabled.clear()
            self._flush_remaining()
            self._reset_state()
            print("[dictation] OFF", flush=True)
            return

        self._reset_state()
        self.enabled.set()
        print("[dictation] ON", flush=True)

    def stop(self) -> None:
        self.stopping.set()
        self.enabled.clear()
        self._flush_remaining()

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
        description="Global hotkey dictation using OpenAI Whisper with rolling near-live transcription."
    )
    parser.add_argument(
        "--hotkey",
        default=default_hotkey,
        type=validate_hotkey,
        help=f"Toggle hotkey (default on this OS: '{default_hotkey}').",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Optional Whisper model cache directory (default: ~/.cache/whisper).",
    )
    parser.add_argument(
        "--offline-strict",
        action="store_true",
        help="Disallow any model download. Require local cached model with valid checksum.",
    )
    parser.add_argument(
        "--no-console-transcript",
        action="store_true",
        help="Do not print recognized text to stdout.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'en'. Omit for auto-detect.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.10,
        help="Audio callback block size in seconds.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=0.35,
        help="Seconds between rolling transcription passes.",
    )
    parser.add_argument(
        "--min-chunk-secs",
        type=float,
        default=0.45,
        help="Minimum unprocessed audio length before running Whisper.",
    )
    parser.add_argument(
        "--pause-silence-secs",
        type=float,
        default=0.30,
        help="Silence length used to force-flush the current partial segment.",
    )
    parser.add_argument(
        "--trailing-silence-secs",
        type=float,
        default=0.20,
        help="If audio tail after last segment exceeds this, finalize the segment.",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-36.0,
        help="Voice threshold in dBFS used for pause detection.",
    )
    parser.add_argument(
        "--no-speech-thresh",
        type=float,
        default=0.55,
        help="Discard segments with no_speech_prob above this.",
    )
    parser.add_argument(
        "--same-out-limit",
        type=int,
        default=2,
        help="Repeated identical partial outputs before confirming them.",
    )
    parser.add_argument(
        "--max-buffer-secs",
        type=float,
        default=30.0,
        help="Maximum rolling buffer size in seconds.",
    )
    parser.add_argument(
        "--trim-secs",
        type=float,
        default=6.0,
        help="Seconds trimmed from the front when buffer exceeds max size.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index from `python -m sounddevice`.",
    )

    args = parser.parse_args()
    return DictationConfig(
        hotkey=args.hotkey,
        model=args.model,
        model_cache_dir=args.model_cache_dir,
        offline_strict=args.offline_strict,
        print_transcripts=not args.no_console_transcript,
        language=args.language,
        sample_rate=args.sample_rate,
        block_duration=args.block_duration,
        live_interval=args.live_interval,
        min_chunk_secs=args.min_chunk_secs,
        pause_silence_secs=args.pause_silence_secs,
        trailing_silence_secs=args.trailing_silence_secs,
        silence_threshold_db=args.silence_threshold_db,
        no_speech_thresh=args.no_speech_thresh,
        same_out_limit=args.same_out_limit,
        max_buffer_secs=args.max_buffer_secs,
        trim_secs=args.trim_secs,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    app = WhisperHotkeyDictation(config)
    app.run()


if __name__ == "__main__":
    main()
