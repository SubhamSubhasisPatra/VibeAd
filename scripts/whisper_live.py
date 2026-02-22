#!/usr/bin/env python3
"""Optional Python wrapper for whisper.cpp stream binary.

Auto-discovers the whisper-stream binary and model from the project's
local/ directory.  Adds:

  - Clean transcript output with noise filtering
  - Optional hotkey toggle (pause/resume)
  - Optional auto-typing into the focused application
  - Audio device listing

This wrapper is entirely optional — the shell/PowerShell run scripts
work without Python.

Usage:
  python scripts/whisper_live.py
  python scripts/whisper_live.py --type-output
  python scripts/whisper_live.py --hotkey '<cmd>+<shift>+g'
  python scripts/whisper_live.py --list-devices
  python scripts/whisper_live.py --step 300 --length 3000
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths — everything relative to the project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DIR = PROJECT_ROOT / "local"
BIN_DIR = LOCAL_DIR / "bin"
MODELS_DIR = LOCAL_DIR / "models"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_B = "\033[1m"
_G = "\033[32m"
_Y = "\033[33m"
_R = "\033[31m"
_C = "\033[36m"
_0 = "\033[0m"

if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    _B = _G = _Y = _R = _C = _0 = ""


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_0}"


def log_info(msg: str) -> None:
    print(f"  {_c(_G, '✓')} {msg}")


def log_warn(msg: str) -> None:
    print(f"  {_c(_Y, '⚠')} {msg}")


def log_error(msg: str) -> None:
    print(f"  {_c(_R, '✗')} {msg}", file=sys.stderr)


def log_step(msg: str) -> None:
    print(f"\n{_c(_B, '▸')} {msg}")


def log_transcript(text: str) -> None:
    """Print a cleaned transcript line."""
    print(f"  {_c(_C, '▶')} {text}")


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _is_mac() -> bool:
    return platform.system().lower() == "darwin"


def cpu_count_safe() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def default_threads() -> int:
    return max(1, min(cpu_count_safe(), 8))


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------


def find_stream_binary(explicit: Optional[str] = None) -> Path:
    """Locate the whisper-stream binary."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified stream binary not found: {explicit}")

    if _is_windows():
        names = ["whisper-stream.exe", "stream.exe"]
    else:
        names = ["whisper-stream", "stream"]

    for name in names:
        candidate = BIN_DIR / name
        if candidate.is_file():
            return candidate

    # Broader search under local/
    for name in names:
        for p in LOCAL_DIR.rglob(name):
            if p.is_file() and os.access(p, os.X_OK):
                return p

    raise FileNotFoundError(
        f"whisper-stream binary not found in {BIN_DIR}.\n"
        f"Run the setup script first:\n"
        f"  macOS:   cd scripts && ./setup-mac.sh\n"
        f"  Windows: cd scripts; .\\setup-windows.ps1"
    )


def find_model(explicit: Optional[str] = None) -> Path:
    """Locate a GGML model file."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        # Maybe it's just a filename — look in models dir
        candidate = MODELS_DIR / explicit
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Specified model not found: {explicit}")

    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {MODELS_DIR}\nRun the setup script first."
        )

    # Prefer quantized models, then plain ggml
    ggml_files = sorted(MODELS_DIR.glob("ggml-*.bin"), reverse=True)

    # Prioritise quantized (smaller/faster)
    quantized = [f for f in ggml_files if re.search(r"-q\d", f.name)]
    if quantized:
        return quantized[0]

    if ggml_files:
        return ggml_files[0]

    raise FileNotFoundError(
        f"No GGML model files found in {MODELS_DIR}.\nRun the setup script first."
    )


# ---------------------------------------------------------------------------
# Transcript cleaning
# ---------------------------------------------------------------------------

# Patterns that whisper.cpp emits as noise / non-speech
_NOISE_PATTERNS = [
    re.compile(r"^\s*\[.*\]\s*$"),  # [BLANK_AUDIO], [MUSIC], etc.
    re.compile(r"^\s*\(.*\)\s*$"),  # (inaudible), (music), etc.
    re.compile(r"^\s*$"),  # empty
    re.compile(r"^\s*\.+\s*$"),  # just dots
    re.compile(r"^\s*\*+\s*$"),  # just asterisks
]

# Common whisper hallucination phrases when there's silence
_HALLUCINATION_PHRASES = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thanks for listening",
        "subscribe",
        "like and subscribe",
        "you",
        "the",
        "bye",
        "bye bye",
        "see you next time",
        "so",
    }
)


def _clean_transcript_line(raw: str) -> Optional[str]:
    """Clean a raw transcript line from whisper-stream.

    Returns None if the line should be suppressed (noise/hallucination).
    """
    # Strip ANSI escape codes
    text = re.sub(r"\033\[[0-9;]*m", "", raw)

    # Strip leading timestamps like [00:00:00.000 --> 00:00:05.000]
    text = re.sub(
        r"^\s*\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*",
        "",
        text,
    )

    text = text.strip()

    if not text:
        return None

    for pat in _NOISE_PATTERNS:
        if pat.match(text):
            return None

    # Check for common hallucinations (case-insensitive, exact match)
    if text.lower().rstrip(".!?,") in _HALLUCINATION_PHRASES:
        return None

    return text


# ---------------------------------------------------------------------------
# Keyboard typer (optional, requires pynput)
# ---------------------------------------------------------------------------


class _KeyboardTyper:
    """Types text into the focused application using pynput."""

    def __init__(self) -> None:
        self._controller = None
        self._available = False
        try:
            from pynput.keyboard import Controller

            self._controller = Controller()
            self._available = True
        except ImportError:
            log_warn(
                "pynput not installed — --type-output disabled.\n"
                "  Install with: pip install pynput"
            )
        except Exception as exc:
            log_warn(f"pynput init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def type_text(self, text: str) -> None:
        if not self._available or not self._controller:
            return
        try:
            self._controller.type(text + " ")
        except Exception as exc:
            log_warn(f"Typing failed: {exc}")


# ---------------------------------------------------------------------------
# Hotkey toggle (optional, requires pynput)
# ---------------------------------------------------------------------------


class _HotkeyToggle:
    """Global hotkey listener to pause/resume transcription."""

    def __init__(self, hotkey_str: str, callback) -> None:
        self._listener = None
        self._hotkey_str = hotkey_str
        self._callback = callback
        self._available = False

        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys({hotkey_str: self._fire})
            self._available = True
        except ImportError:
            log_warn(
                "pynput not installed — --hotkey disabled.\n"
                "  Install with: pip install pynput"
            )
        except Exception as exc:
            log_warn(f"Hotkey setup failed: {exc}")

    def _fire(self) -> None:
        try:
            self._callback()
        except Exception:
            pass

    def start(self) -> None:
        if self._listener:
            self._listener.start()

    def stop(self) -> None:
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# Main WhisperLive class
# ---------------------------------------------------------------------------


class WhisperLive:
    """Spawns whisper-stream and processes its output."""

    def __init__(
        self,
        stream_bin: Path,
        model_path: Path,
        threads: int = 4,
        step: int = 500,
        length: int = 5000,
        keep: int = 200,
        language: str = "en",
        type_output: bool = False,
        hotkey: Optional[str] = None,
        show_timestamps: bool = False,
        capture_id: int = -1,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.stream_bin = stream_bin
        self.model_path = model_path
        self.threads = threads
        self.step = step
        self.length = length
        self.keep = keep
        self.language = language
        self.type_output = type_output
        self.show_timestamps = show_timestamps
        self.capture_id = capture_id
        self.extra_args = extra_args or []

        self._process: Optional[subprocess.Popen] = None
        self._paused = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Optional components
        self._typer: Optional[_KeyboardTyper] = None
        if type_output:
            self._typer = _KeyboardTyper()

        self._hotkey: Optional[_HotkeyToggle] = None
        if hotkey:
            self._hotkey = _HotkeyToggle(hotkey, self._toggle_pause)

    def _toggle_pause(self) -> None:
        with self._lock:
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "LISTENING"
            log_info(f"Hotkey toggled → {state}")

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def _build_cmd(self) -> List[str]:
        cmd = [
            str(self.stream_bin),
            "-m",
            str(self.model_path),
            "--threads",
            str(self.threads),
            "--step",
            str(self.step),
            "--length",
            str(self.length),
            "--keep",
            str(self.keep),
            "--language",
            self.language,
        ]

        if self.capture_id >= 0:
            cmd.extend(["--capture", str(self.capture_id)])

        cmd.extend(self.extra_args)
        return cmd

    def _read_stream(self, proc: subprocess.Popen) -> None:
        """Read stdout from whisper-stream and process lines."""
        assert proc.stdout is not None

        for raw_line in iter(proc.stdout.readline, ""):
            if self._stop_event.is_set():
                break

            raw_line = raw_line.rstrip("\n\r")

            if self.paused:
                continue

            if self.show_timestamps:
                log_transcript(raw_line)
                continue

            cleaned = _clean_transcript_line(raw_line)
            if cleaned:
                log_transcript(cleaned)

                if self._typer and self._typer.available:
                    self._typer.type_text(cleaned)

    def start(self) -> None:
        """Start the whisper-stream process and read its output."""
        cmd = self._build_cmd()

        log_step("Starting whisper-stream …")
        log_info(f"Binary : {self.stream_bin}")
        log_info(f"Model  : {self.model_path}")
        log_info(f"Threads: {self.threads}")
        log_info(
            f"Step   : {self.step} ms  |  Length: {self.length} ms  |  Keep: {self.keep} ms"
        )

        if self._hotkey and self._hotkey.available:
            log_info(f"Hotkey : toggle with {self._hotkey._hotkey_str}")

        if self._typer and self._typer.available:
            log_info("Typing : enabled (output will be typed into focused app)")

        print()
        print(f"  {_c(_B, 'Press Ctrl+C to stop.')}")
        print()

        # Set up library path for shared libs
        env = os.environ.copy()
        bin_dir_str = str(BIN_DIR)

        if _is_mac():
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = (
                f"{bin_dir_str}:{existing}" if existing else bin_dir_str
            )
        elif not _is_windows():
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{bin_dir_str}:{existing}" if existing else bin_dir_str
            )

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError:
            log_error(f"Binary not found: {self.stream_bin}")
            return
        except PermissionError:
            log_error(f"Permission denied: {self.stream_bin}")
            log_info("Try: chmod +x " + str(self.stream_bin))
            return
        except Exception as exc:
            log_error(f"Failed to start whisper-stream: {exc}")
            return

        if self._hotkey:
            self._hotkey.start()

        # Read stderr in a separate thread (for init messages)
        def _read_stderr():
            assert self._process is not None and self._process.stderr is not None
            for line in iter(self._process.stderr.readline, ""):
                line = line.rstrip("\n\r")
                if line:
                    # Show init messages, suppress noise
                    if any(
                        kw in line.lower()
                        for kw in ("init:", "whisper_", "ggml_", "main:", "[start")
                    ):
                        print(f"  {_c(_Y, '·')} {line}")

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Give the process a moment to initialise
        time.sleep(0.5)

        if self._process.poll() is not None:
            log_error(
                f"whisper-stream exited immediately with code {self._process.returncode}"
            )
            remaining_err = self._process.stderr.read() if self._process.stderr else ""
            if remaining_err:
                log_error(f"stderr: {remaining_err[:1000]}")
            return

        try:
            self._read_stream(self._process)
        except KeyboardInterrupt:
            log_info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the stream process and clean up."""
        self._stop_event.set()

        if self._hotkey:
            self._hotkey.stop()

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            except Exception:
                pass

        log_info("Stopped.")


# ---------------------------------------------------------------------------
# List audio devices
# ---------------------------------------------------------------------------


def list_audio_devices() -> None:
    """List available audio input devices using sounddevice."""
    try:
        import sounddevice as sd
    except ImportError:
        log_error("sounddevice not installed.\n  Install with: pip install sounddevice")
        sys.exit(1)

    log_step("Available audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default_marker = " (default)" if i == sd.default.device[0] else ""
            print(
                f"  #{i}: {dev['name']}  "
                f"({dev['max_input_channels']}ch, {int(dev['default_samplerate'])}Hz)"
                f"{default_marker}"
            )


# ---------------------------------------------------------------------------
# Default hotkey per platform
# ---------------------------------------------------------------------------


def _default_hotkey() -> Optional[str]:
    if _is_mac():
        return "<cmd>+<shift>+g"
    elif _is_windows():
        return "<alt>+<ctrl>+<shift>+m"
    else:
        return "<ctrl>+<alt>+<space>"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Python wrapper for whisper.cpp stream — adds transcript filtering, "
            "hotkey toggle, and optional auto-typing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/whisper_live.py
              python scripts/whisper_live.py --type-output
              python scripts/whisper_live.py --step 300 --length 3000
              python scripts/whisper_live.py --hotkey '<cmd>+<shift>+g'
              python scripts/whisper_live.py --list-devices
              python scripts/whisper_live.py --capture 2
              python scripts/whisper_live.py -- --vad-thold 0.6 --no-fallback
        """),
    )

    parser.add_argument(
        "--stream-bin",
        default=None,
        help="Explicit path to whisper-stream binary (auto-discovered by default).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path or filename (auto-discovered from local/models/ by default).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=default_threads(),
        help=f"CPU thread count (default: {default_threads()}).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=500,
        help="Audio step size in ms (default: 500).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=5000,
        help="Audio buffer length in ms (default: 5000).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=200,
        help="Audio kept from previous step in ms (default: 200).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Spoken language code (default: en).",
    )
    parser.add_argument(
        "--capture",
        type=int,
        default=-1,
        help="Audio capture device ID (-1 = default).",
    )
    parser.add_argument(
        "--type-output",
        action="store_true",
        help="Type recognised text into the focused application (requires pynput).",
    )
    parser.add_argument(
        "--show-timestamps",
        action="store_true",
        help="Show raw timestamp lines from whisper-stream.",
    )
    parser.add_argument(
        "--hotkey",
        default=None,
        nargs="?",
        const="__default__",
        help=(
            "Global hotkey to toggle pause/resume (requires pynput). "
            "If no key is specified, uses platform default."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit (requires sounddevice).",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra arguments passed directly to whisper-stream.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Resolve hotkey
    hotkey = None
    if args.hotkey:
        if args.hotkey == "__default__":
            hotkey = _default_hotkey()
        else:
            hotkey = args.hotkey

    # Find binary and model
    try:
        stream_bin = find_stream_binary(args.stream_bin)
    except FileNotFoundError as exc:
        log_error(str(exc))
        sys.exit(1)

    try:
        model_path = find_model(args.model)
    except FileNotFoundError as exc:
        log_error(str(exc))
        sys.exit(1)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *_: None)

    wl = WhisperLive(
        stream_bin=stream_bin,
        model_path=model_path,
        threads=args.threads,
        step=args.step,
        length=args.length,
        keep=args.keep,
        language=args.language,
        type_output=args.type_output,
        hotkey=hotkey,
        show_timestamps=args.show_timestamps,
        capture_id=args.capture,
        extra_args=args.extra,
    )

    wl.start()


if __name__ == "__main__":
    main()
