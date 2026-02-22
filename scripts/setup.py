#!/usr/bin/env python3
"""Cross-platform setup for whisper.cpp live transcription.

Downloads the official OpenAI Whisper model (.pt) from Azure CDN,
converts it to GGML format using whisper.cpp's conversion tooling,
and obtains the whisper-stream binary (prebuilt on Windows, built
from source on macOS).

Everything is stored under <project_root>/local/:
  local/whisper.cpp/   — shallow clone (conversion scripts + source)
  local/models/        — .pt and .bin model files
  local/bin/           — binaries and shared libraries

*** ZERO HuggingFace dependencies.  Every model byte comes from
    https://openaipublic.azureedge.net — the official OpenAI CDN. ***
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — everything relative to the project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DIR = PROJECT_ROOT / "local"
MODELS_DIR = LOCAL_DIR / "models"
BIN_DIR = LOCAL_DIR / "bin"
WHISPER_CPP_DIR = LOCAL_DIR / "whisper.cpp"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Official OpenAI Azure CDN model URLs — the ONLY source we ever touch.
OPENAI_MODEL_URLS: Dict[str, str] = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/"
    "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/"
    "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/"
    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/"
    "f953ad0fd29cacd07d5a9eda5b4496e2193f090f0a3f20d9c00f2826d1a5399d/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/"
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/"
    "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/"
    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/"
    "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/"
    "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/"
    "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/"
    "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

WHISPER_CPP_GIT_URL = "https://github.com/ggerganov/whisper.cpp.git"
GITHUB_RELEASES_API = (
    "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest"
)

MIN_MODEL_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB (tiny is ~75 MB)

# HuggingFace domains we BLOCK.
_BLOCKED_DOMAINS = frozenset(
    {
        "huggingface.co",
        "hf.co",
        "hf-mirror.com",
        "cdn-lfs.huggingface.co",
        "cdn-lfs.hf.co",
    }
)

# Extensions that are never real binaries.
_NON_BINARY_EXTENSIONS = frozenset(
    {
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".py",
        ".sh",
        ".bat",
        ".ps1",
        ".txt",
        ".md",
        ".cmake",
        ".pc",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".o",
        ".obj",
        ".a",
        ".lib",
        ".d",
        ".make",
        ".log",
    }
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_B = "\033[1m"
_G = "\033[32m"
_Y = "\033[33m"
_R = "\033[31m"
_0 = "\033[0m"

if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    _B = _G = _Y = _R = _0 = ""


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


def fatal(msg: str) -> None:
    log_error(msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# URL safety gate
# ---------------------------------------------------------------------------


def _assert_not_huggingface(url: str) -> None:
    from urllib.parse import urlparse

    hostname = (urlparse(url).hostname or "").lower()
    for blocked in _BLOCKED_DOMAINS:
        if hostname == blocked or hostname.endswith(f".{blocked}"):
            fatal(
                f"BLOCKED: refusing to download from HuggingFace domain "
                f"'{hostname}'.\n       URL: {url}\n"
                f"       This project only uses official OpenAI Azure CDN."
            )


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def detect_platform() -> Tuple[str, str]:
    os_name = platform.system().lower()
    arch = platform.machine().lower()
    if os_name not in ("darwin", "windows", "linux"):
        fatal(f"Unsupported OS: {os_name}")
    return os_name, arch


def cpu_count_safe() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def sha256_from_url(url: str) -> Optional[str]:
    """Extract the SHA256 hash embedded in the OpenAI model URL path."""
    parts = url.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if part == "models" and i + 2 < len(parts):
            candidate = parts[i + 1]
            if re.fullmatch(r"[0-9a-f]{64}", candidate):
                return candidate
    return None


def safe_download(
    url: str,
    dest: Path,
    label: str = "",
    expected_sha256: Optional[str] = None,
    min_size: int = 0,
) -> Path:
    """Download *url* → *dest* with HuggingFace guard and integrity checks."""
    _assert_not_huggingface(url)
    desc = label or dest.name
    log_info(f"Downloading {desc} …")
    log_info(f"  URL: {url}")

    hasher = hashlib.sha256()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "whisper-setup/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(
                            f"\r    [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
                            end="",
                            flush=True,
                        )
            if total > 0:
                print(flush=True)
    except Exception as exc:
        fatal(f"Download failed for {desc}: {exc}")

    actual_size = dest.stat().st_size
    if min_size > 0 and actual_size < min_size:
        fatal(
            f"Downloaded file {dest.name} is too small "
            f"({actual_size} bytes, expected >= {min_size})."
        )

    if expected_sha256:
        actual_sha = hasher.hexdigest()
        if actual_sha != expected_sha256:
            fatal(
                f"SHA256 mismatch for {dest.name}!\n"
                f"  expected: {expected_sha256}\n"
                f"  actual:   {actual_sha}"
            )
        log_info(f"SHA256 verified: {expected_sha256[:16]}…")

    log_info(f"Saved → {dest}  ({actual_size / (1024 * 1024):.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# Step 1 — Clone whisper.cpp (for conversion script + source)
# ---------------------------------------------------------------------------


def clone_whisper_cpp() -> Path:
    """Shallow-clone whisper.cpp into local/whisper.cpp/."""
    if WHISPER_CPP_DIR.exists() and (WHISPER_CPP_DIR / "models").is_dir():
        log_info("whisper.cpp already cloned. Skipping.")
        return WHISPER_CPP_DIR

    log_step("Cloning whisper.cpp (shallow) …")
    if WHISPER_CPP_DIR.exists():
        shutil.rmtree(WHISPER_CPP_DIR)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", WHISPER_CPP_GIT_URL, str(WHISPER_CPP_DIR)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        fatal(
            "'git' is not installed or not on PATH.\n  macOS: xcode-select --install\n  Windows: https://git-scm.com/download/win"
        )
    except subprocess.CalledProcessError as exc:
        fatal(f"git clone failed:\n{exc.stderr}")
    except subprocess.TimeoutExpired:
        fatal("git clone timed out after 120 s.")

    log_info(f"Cloned → {WHISPER_CPP_DIR}")
    return WHISPER_CPP_DIR


# ---------------------------------------------------------------------------
# Step 2 — Download .pt model from OpenAI Azure CDN
# ---------------------------------------------------------------------------


def download_pt_model(model_name: str) -> Path:
    """Download official OpenAI .pt model."""
    if model_name not in OPENAI_MODEL_URLS:
        fatal(
            f"Unknown model '{model_name}'. Available:\n"
            + ", ".join(sorted(OPENAI_MODEL_URLS.keys()))
        )

    url = OPENAI_MODEL_URLS[model_name]
    expected_sha = sha256_from_url(url)
    dest = MODELS_DIR / f"{model_name}.pt"

    if dest.exists() and dest.stat().st_size > MIN_MODEL_SIZE_BYTES:
        if expected_sha:
            log_info(f"Verifying existing {dest.name} …")
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() == expected_sha:
                log_info(f"Model {dest.name} already present and verified. Skipping.")
                return dest
            else:
                log_warn(f"Existing {dest.name} has bad checksum. Re-downloading.")
        else:
            log_info(f"Model {dest.name} already present. Skipping.")
            return dest

    safe_download(
        url=url,
        dest=dest,
        label=f"OpenAI {model_name} model (.pt)",
        expected_sha256=expected_sha,
        min_size=MIN_MODEL_SIZE_BYTES,
    )
    return dest


# ---------------------------------------------------------------------------
# Step 3 — Convert .pt → ggml
# ---------------------------------------------------------------------------


def find_convert_script() -> Path:
    """Locate convert-pt-to-ggml.py inside the whisper.cpp clone."""
    candidates = [
        WHISPER_CPP_DIR / "models" / "convert-pt-to-ggml.py",
        WHISPER_CPP_DIR / "convert-pt-to-ggml.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Fallback: search recursively
    for p in WHISPER_CPP_DIR.rglob("convert-pt-to-ggml.py"):
        return p
    fatal(
        f"Cannot find convert-pt-to-ggml.py inside {WHISPER_CPP_DIR}.\n"
        "The whisper.cpp repository structure may have changed."
    )
    return Path()  # unreachable


def convert_pt_to_ggml(pt_path: Path, model_name: str) -> Path:
    """Convert a .pt model to ggml .bin format.

    This is pure Python — no cmake, no C compiler needed.
    Requires: torch, numpy, openai-whisper (installed in the venv).
    """
    output_bin = MODELS_DIR / f"ggml-{model_name}.bin"
    if output_bin.exists() and output_bin.stat().st_size > MIN_MODEL_SIZE_BYTES:
        log_info(f"Converted model {output_bin.name} already exists. Skipping.")
        return output_bin

    log_step(f"Converting {pt_path.name} → ggml format …")

    # Verify Python packages
    missing: List[str] = []
    for pkg in ("torch", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    try:
        __import__("whisper")
    except ImportError:
        missing.append("openai-whisper")

    if missing:
        fatal(
            f"Missing Python packages for conversion: {', '.join(missing)}\n"
            f"These should have been installed by the setup wrapper script.\n"
            f"Manual install: pip install {' '.join(missing)}"
        )

    convert_script = find_convert_script()
    log_info(f"Using conversion script: {convert_script}")

    # The convert script wants: <model.pt> <whisper-source-dir> [output-dir]
    import whisper as _whisper_pkg

    if _whisper_pkg.__file__ is None:
        fatal("Cannot determine whisper package location.")
    whisper_package_dir = str(Path(_whisper_pkg.__file__).parent.parent)
    log_info(f"Whisper package source: {whisper_package_dir}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        str(pt_path),
        whisper_package_dir,
        str(MODELS_DIR),
    ]

    log_info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(WHISPER_CPP_DIR),
        )
        if result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            # Show first few and last few lines
            show = lines[:3] + (["  ..."] if len(lines) > 6 else []) + lines[-3:]
            for line in show:
                log_info(f"  [convert] {line}")
    except subprocess.CalledProcessError as exc:
        log_error(f"Conversion stdout:\n{exc.stdout}")
        log_error(f"Conversion stderr:\n{exc.stderr}")
        fatal(f"Model conversion failed (exit code {exc.returncode}).")
    except subprocess.TimeoutExpired:
        fatal("Model conversion timed out after 300 s.")

    # Find the output file
    if output_bin.exists():
        log_info(
            f"Converted → {output_bin}  ({output_bin.stat().st_size / (1024 * 1024):.1f} MB)"
        )
        return output_bin

    # Fallback: search for any ggml*.bin
    for p in sorted(MODELS_DIR.glob("ggml*.bin")):
        if p.stat().st_size > MIN_MODEL_SIZE_BYTES:
            if p.name != output_bin.name:
                log_warn(f"Expected {output_bin.name} but found {p.name}. Renaming.")
                p.rename(output_bin)
            log_info(
                f"Converted → {output_bin}  ({output_bin.stat().st_size / (1024 * 1024):.1f} MB)"
            )
            return output_bin

    # Search whisper.cpp dir too
    for p in sorted(WHISPER_CPP_DIR.rglob("ggml*.bin")):
        if p.stat().st_size > MIN_MODEL_SIZE_BYTES:
            target = MODELS_DIR / output_bin.name
            shutil.move(str(p), str(target))
            log_info(
                f"Converted → {target}  ({target.stat().st_size / (1024 * 1024):.1f} MB)"
            )
            return target

    fatal(
        f"Conversion completed but output not found.\n"
        f"Expected: {output_bin}\n"
        f"Contents of {MODELS_DIR}:\n"
        + "\n".join(f"  {p.name}" for p in MODELS_DIR.iterdir())
    )
    return Path()  # unreachable


# ---------------------------------------------------------------------------
# Step 4 — Obtain whisper-stream binary
# ---------------------------------------------------------------------------


def _find_executable(root: Path, names: List[str]) -> Optional[Path]:
    """Find the first real executable matching one of *names* under *root*.

    Names are tried in priority order — the first name in the list is
    preferred over later ones.  This matters because e.g. ``whisper-stream``
    (the real 768 KB binary) must be preferred over ``stream`` (a 35 KB
    deprecation wrapper that the latest whisper.cpp builds also emit).
    """
    # Collect all candidate executables, keyed by stem
    candidates: Dict[str, List[Path]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            p = Path(dirpath) / fname
            if p.suffix.lower() in _NON_BINARY_EXTENSIONS:
                continue
            if not os.access(p, os.X_OK):
                continue
            stem = p.stem.lower()
            if stem in names:
                candidates.setdefault(stem, []).append(p)

    # Return the first match by *name priority* (not walk order).
    # Within the same name, prefer the largest file (real binary > wrapper).
    for name in names:
        if name in candidates:
            return max(candidates[name], key=lambda p: p.stat().st_size)
    return None


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-setup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _copy_shared_libs(build_dir: Path) -> None:
    """Copy .dylib / .so files from the build tree into BIN_DIR."""
    lib_search_dirs = [
        build_dir / "src",
        build_dir / "ggml" / "src",
        build_dir / "ggml" / "src" / "ggml-metal",
        build_dir / "ggml" / "src" / "ggml-blas",
    ]
    count = 0
    for lib_dir in lib_search_dirs:
        if not lib_dir.is_dir():
            continue
        for pat in ("*.dylib", "*.so", "*.so.*"):
            for lib_file in lib_dir.glob(pat):
                if lib_file.is_file():
                    dest = BIN_DIR / lib_file.name
                    if not dest.exists():
                        shutil.copy2(lib_file, dest)
                        count += 1
    if count:
        log_info(f"Copied {count} shared libraries into local/bin/")


def _install_binary(src: Optional[Path], name: str) -> Optional[Path]:
    """Copy a single binary into BIN_DIR and make it executable."""
    if src is None or not src.exists():
        return None
    dest = BIN_DIR / src.name
    shutil.copy2(src, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    log_info(f"Installed {name}: {dest.name}")
    return dest


# ── macOS / Linux: build from source ────────────────────────────────────────


def build_from_source(os_name: str) -> Path:
    """Build whisper.cpp from the cloned source using cmake.

    Returns the path to the installed whisper-stream binary.
    """
    log_step("Building whisper.cpp from source (cmake) …")

    # Check tools
    has_cmake = shutil.which("cmake") is not None
    has_make = shutil.which("make") is not None
    has_cc = any(shutil.which(x) for x in ("cc", "gcc", "clang"))

    if not (has_cmake and has_make and has_cc):
        hints = []
        if not has_cmake:
            hints.append("cmake  → brew install cmake")
        if not has_cc:
            hints.append("C/C++  → xcode-select --install")
        fatal(
            "Building from source requires cmake, make, and a C/C++ compiler.\n"
            "  Missing:\n" + "\n".join(f"    {h}" for h in hints) + "\n"
            "  After installing, re-run this setup."
        )

    # Check SDL2 (required for the stream binary's mic capture)
    sdl2_ok = (
        shutil.which("sdl2-config") is not None
        or Path("/opt/homebrew/include/SDL2").is_dir()
        or Path("/usr/local/include/SDL2").is_dir()
        or Path("/usr/include/SDL2").is_dir()
    )
    if not sdl2_ok:
        fatal(
            "SDL2 is required for the stream binary (live microphone capture).\n"
            "  Install:  brew install sdl2\n"
            "  Then re-run this setup."
        )

    build_dir = WHISPER_CPP_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    nproc = min(cpu_count_safe(), 12)

    # cmake configure
    log_info("Configuring with cmake …")
    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(WHISPER_CPP_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DWHISPER_BUILD_EXAMPLES=ON",
        "-DWHISPER_SDL2=ON",
    ]
    if os_name == "darwin":
        configure_cmd.append("-DWHISPER_METAL=ON")

    try:
        result = subprocess.run(
            configure_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(WHISPER_CPP_DIR),
        )
        for line in result.stdout.strip().splitlines()[-3:]:
            log_info(f"  [cmake] {line}")
    except subprocess.CalledProcessError as exc:
        log_error(f"cmake stderr:\n{exc.stderr[-2000:]}")
        fatal(f"cmake configure failed (exit {exc.returncode}).")
    except subprocess.TimeoutExpired:
        fatal("cmake configure timed out.")

    # cmake build
    log_info(f"Building with {nproc} parallel jobs …")
    try:
        result = subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "-j",
                str(nproc),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(WHISPER_CPP_DIR),
        )
        for line in result.stdout.strip().splitlines()[-3:]:
            log_info(f"  [build] {line}")
    except subprocess.CalledProcessError as exc:
        log_error(f"build stderr:\n{exc.stderr[-2000:]}")
        fatal(f"cmake build failed (exit {exc.returncode}).")
    except subprocess.TimeoutExpired:
        fatal("cmake build timed out after 600 s.")

    log_info("Build complete.")

    # Locate built binaries
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    stream_bin = _find_executable(build_dir, ["whisper-stream", "stream"])
    cli_bin = _find_executable(build_dir, ["whisper-cli"])
    quantize_bin = _find_executable(build_dir, ["whisper-quantize"])

    if stream_bin is None:
        # Broader search
        for p in build_dir.rglob("*stream*"):
            if (
                p.is_file()
                and os.access(p, os.X_OK)
                and p.suffix.lower() not in _NON_BINARY_EXTENSIONS
            ):
                stream_bin = p
                break

    if stream_bin is None:
        fatal(
            "Build succeeded but whisper-stream binary not found.\n"
            "Ensure SDL2 is installed: brew install sdl2"
        )

    installed_stream = _install_binary(stream_bin, "whisper-stream")
    _install_binary(cli_bin, "whisper-cli")
    _install_binary(quantize_bin, "whisper-quantize")

    # Copy shared libraries
    _copy_shared_libs(build_dir)

    # Copy Metal shader if present
    for metallib in build_dir.rglob("*.metallib"):
        dest = BIN_DIR / metallib.name
        if not dest.exists():
            shutil.copy2(metallib, dest)
            log_info(f"Installed Metal shader: {dest.name}")

    assert installed_stream is not None
    return installed_stream


# ── Windows: download prebuilt binary ────────────────────────────────────────


def download_prebuilt_windows(arch: str) -> Path:
    """Download prebuilt whisper.cpp binary for Windows from GitHub releases."""
    log_step("Downloading prebuilt whisper.cpp binary for Windows …")

    try:
        release_data = _fetch_json(GITHUB_RELEASES_API)
    except Exception as exc:
        fatal(f"Failed to fetch GitHub releases: {exc}")

    tag = release_data.get("tag_name", "unknown")
    assets = release_data.get("assets", [])
    log_info(f"Latest release: {tag}")

    # Find a suitable Windows archive
    arch_pattern = "x64" if "64" in arch or arch in ("amd64", "x86_64") else "Win32"
    best_url = None

    for asset in assets:
        name = asset.get("name", "").lower()
        url = asset.get("browser_download_url", "")
        if not url:
            continue
        # Prefer the plain binary (not cublas/blas) for simplicity
        if arch_pattern.lower() in name and name.endswith(".zip"):
            if "cublas" not in name and "blas" not in name:
                best_url = url
                break

    # Fallback: accept blas variant
    if best_url is None:
        for asset in assets:
            name = asset.get("name", "").lower()
            url = asset.get("browser_download_url", "")
            if arch_pattern.lower() in name and name.endswith(".zip"):
                best_url = url
                break

    if best_url is None:
        fatal(
            f"No Windows {arch_pattern} binary found in release {tag}.\n"
            f"Available assets:\n" + "\n".join(f"  - {a['name']}" for a in assets)
        )

    _assert_not_huggingface(best_url)

    archive_path = LOCAL_DIR / "whisper-windows.zip"
    safe_download(best_url, archive_path, label=f"whisper.cpp {tag} (Windows)")

    # Extract
    extract_dir = LOCAL_DIR / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    log_info("Extracting archive …")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find root of extracted content
    items = list(extract_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        bin_root = items[0]
    else:
        bin_root = extract_dir

    # Install binaries
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    stream_bin = _find_executable(bin_root, ["whisper-stream", "stream"])
    cli_bin = _find_executable(bin_root, ["whisper-cli", "main", "whisper"])
    quantize_bin = _find_executable(bin_root, ["whisper-quantize", "quantize"])

    # On Windows, also check for .exe explicitly
    if stream_bin is None:
        for p in bin_root.rglob("*.exe"):
            if "stream" in p.stem.lower():
                stream_bin = p
                break

    if stream_bin is None:
        # The prebuilt Windows releases may not include stream binary
        # Fall back to the CLI binary approach
        log_warn("whisper-stream.exe not found in release. Checking for alternatives …")

        # Check if they renamed it or if we need to use whisper-cli
        for p in bin_root.rglob("*.exe"):
            log_info(f"  Found: {p.name}")

        if cli_bin is not None:
            log_warn(
                "whisper-stream.exe is not included in Windows prebuilt releases.\n"
                "  The stream binary requires SDL2 and must be built from source.\n"
                "  You can still use whisper-cli for file-based transcription."
            )
            _install_binary(cli_bin, "whisper-cli")
            _install_binary(quantize_bin, "whisper-quantize")

            # Copy all DLLs
            for dll in bin_root.rglob("*.dll"):
                dest = BIN_DIR / dll.name
                if not dest.exists():
                    shutil.copy2(dll, dest)

            fatal(
                "The Windows prebuilt release does not include whisper-stream.exe.\n"
                "Options:\n"
                "  1. Install Visual Studio Build Tools + cmake + SDL2, then re-run\n"
                "     with --force-build to build from source.\n"
                "  2. Use the Python wrapper (scripts/whisper_live.py) which can\n"
                "     capture audio via sounddevice and pipe to whisper-cli."
            )

    installed_stream = _install_binary(stream_bin, "whisper-stream")
    _install_binary(cli_bin, "whisper-cli")
    _install_binary(quantize_bin, "whisper-quantize")

    # Copy all DLLs
    for dll in bin_root.rglob("*.dll"):
        dest = BIN_DIR / dll.name
        if not dest.exists():
            shutil.copy2(dll, dest)

    # Cleanup
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if archive_path.exists():
        archive_path.unlink()

    assert installed_stream is not None
    return installed_stream


def obtain_stream_binary(os_name: str, arch: str, force_build: bool = False) -> Path:
    """Get the whisper-stream binary — build on macOS/Linux, download on Windows."""
    # Check if already installed
    if os_name == "windows":
        existing = BIN_DIR / "whisper-stream.exe"
    else:
        existing = BIN_DIR / "whisper-stream"
    if existing.exists() and existing.stat().st_size > 1024:
        log_info(f"Stream binary already installed: {existing.name}")
        return existing

    if os_name == "windows" and not force_build:
        return download_prebuilt_windows(arch)
    else:
        return build_from_source(os_name)


# ---------------------------------------------------------------------------
# Step 5 (optional) — Quantize
# ---------------------------------------------------------------------------


def quantize_model(ggml_path: Path, quant_type: str) -> Path:
    """Quantize the ggml model for faster CPU inference."""
    quantized_name = ggml_path.stem + f"-{quant_type}" + ggml_path.suffix
    quantized_path = MODELS_DIR / quantized_name

    if quantized_path.exists() and quantized_path.stat().st_size > 1024 * 1024:
        log_info(f"Quantized model already exists: {quantized_name}")
        return quantized_path

    quantize_bin = _find_executable(BIN_DIR, ["whisper-quantize", "quantize"])
    if quantize_bin is None:
        log_warn("Quantize binary not found. Skipping quantization.")
        return ggml_path

    log_step(f"Quantizing → {quant_type} …")
    cmd = [str(quantize_bin), str(ggml_path), str(quantized_path), quant_type]
    log_info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[-3:]:
                log_info(f"  [quantize] {line}")
    except subprocess.CalledProcessError as exc:
        log_warn(f"Quantization failed: {exc.stderr[:500]}")
        return ggml_path
    except subprocess.TimeoutExpired:
        log_warn("Quantization timed out. Using unquantized model.")
        return ggml_path

    if quantized_path.exists() and quantized_path.stat().st_size > 1024 * 1024:
        orig_mb = ggml_path.stat().st_size / (1024 * 1024)
        quant_mb = quantized_path.stat().st_size / (1024 * 1024)
        log_info(f"Quantized: {orig_mb:.1f} MB → {quant_mb:.1f} MB ({quant_type})")
        return quantized_path

    log_warn("Quantized file not found. Using unquantized model.")
    return ggml_path


# ---------------------------------------------------------------------------
# Generate run scripts at project root
# ---------------------------------------------------------------------------


def generate_run_scripts(model_path: Path, stream_bin: Path, os_name: str) -> None:
    """Write launcher scripts at the project root."""
    threads = max(1, min(cpu_count_safe(), 8))

    # Paths relative to project root
    model_rel = os.path.relpath(model_path, PROJECT_ROOT)
    stream_rel = os.path.relpath(stream_bin, PROJECT_ROOT)
    bin_dir_rel = os.path.relpath(BIN_DIR, PROJECT_ROOT)

    if os_name in ("darwin", "linux"):
        # Library path export for shared libs
        if os_name == "darwin":
            lib_export = f'export DYLD_LIBRARY_PATH="$SCRIPT_DIR/{bin_dir_rel}:${{DYLD_LIBRARY_PATH:-}}"'
        else:
            lib_export = f'export LD_LIBRARY_PATH="$SCRIPT_DIR/{bin_dir_rel}:${{LD_LIBRARY_PATH:-}}"'

        # run-live.sh
        run_sh = PROJECT_ROOT / "run-live.sh"
        run_sh.write_text(
            textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # whisper-live — real-time transcription via whisper.cpp
            set -euo pipefail

            SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
            STREAM_BIN="$SCRIPT_DIR/{stream_rel}"
            MODEL="$SCRIPT_DIR/{model_rel}"

            if [[ ! -x "$STREAM_BIN" ]]; then
              echo "[error] stream binary not found. Run: cd scripts && ./setup-mac.sh" >&2
              exit 1
            fi
            if [[ ! -f "$MODEL" ]]; then
              echo "[error] model not found. Run: cd scripts && ./setup-mac.sh" >&2
              exit 1
            fi

            {lib_export}

            echo "╔══════════════════════════════════════════════╗"
            echo "║     whisper-live · real-time transcription   ║"
            echo "╠══════════════════════════════════════════════╣"
            echo "║  Model  : {model_path.name:<35s}║"
            echo "║  Threads: {threads:<35d}║"
            echo "║  Ctrl+C to stop                             ║"
            echo "╚══════════════════════════════════════════════╝"
            echo ""

            exec "$STREAM_BIN" \\
              -m "$MODEL" \\
              --threads {threads} \\
              --step 500 \\
              --length 5000 \\
              "$@"
        """)
        )
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        log_info(f"Created {run_sh.name}")

        # run-live-fast.sh
        fast_sh = PROJECT_ROOT / "run-live-fast.sh"
        fast_sh.write_text(
            textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # whisper-live (ultra-low-latency) — step=300ms, length=3000ms
            set -euo pipefail

            SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
            {lib_export}

            exec "$SCRIPT_DIR/{stream_rel}" \\
              -m "$SCRIPT_DIR/{model_rel}" \\
              --threads {threads} \\
              --step 300 \\
              --length 3000 \\
              --keep 200 \\
              "$@"
        """)
        )
        fast_sh.chmod(
            fast_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )
        log_info(f"Created {fast_sh.name}")

    if os_name == "windows":
        # run-live.ps1
        ps1 = PROJECT_ROOT / "run-live.ps1"
        ps1.write_text(
            textwrap.dedent(f"""\
            # whisper-live — real-time transcription via whisper.cpp
            $ErrorActionPreference = "Stop"
            $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
            $StreamBin = Join-Path $ScriptDir "{stream_rel.replace("/", "\\\\")}"
            $Model     = Join-Path $ScriptDir "{model_rel.replace("/", "\\\\")}"

            if (-not (Test-Path $StreamBin)) {{
                Write-Error "stream binary not found. Run: cd scripts; .\\setup-windows.ps1"
                exit 1
            }}

            Write-Host ""
            Write-Host "========================================="
            Write-Host "   whisper-live - real-time transcription"
            Write-Host "   Model  : {model_path.name}"
            Write-Host "   Threads: {threads}"
            Write-Host "   Ctrl+C to stop"
            Write-Host "========================================="
            Write-Host ""

            & $StreamBin -m $Model --threads {threads} --step 500 --length 5000 @args
        """)
        )
        log_info(f"Created {ps1.name}")

        # run-live-fast.ps1
        fast_ps1 = PROJECT_ROOT / "run-live-fast.ps1"
        fast_ps1.write_text(
            textwrap.dedent(f"""\
            # whisper-live (ultra-low-latency)
            $ErrorActionPreference = "Stop"
            $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
            $StreamBin = Join-Path $ScriptDir "{stream_rel.replace("/", "\\\\")}"
            $Model     = Join-Path $ScriptDir "{model_rel.replace("/", "\\\\")}"

            & $StreamBin -m $Model --threads {threads} --step 300 --length 3000 --keep 200 @args
        """)
        )
        log_info(f"Created {fast_ps1.name}")

        # run-live.bat
        bat = PROJECT_ROOT / "run-live.bat"
        bat.write_text(
            textwrap.dedent(f"""\
            @echo off
            REM whisper-live — real-time transcription
            cd /d "%~dp0"
            "{stream_rel.replace("/", "\\\\")}" -m "{model_rel.replace("/", "\\\\")}" --threads {threads} --step 500 --length 5000 %*
        """)
        )
        log_info(f"Created {bat.name}")


# ---------------------------------------------------------------------------
# Main setup orchestrator
# ---------------------------------------------------------------------------


def setup(args: argparse.Namespace) -> None:
    os_name, arch = detect_platform()

    print(f"\n{'=' * 50}")
    print(f"  whisper-live setup")
    print(f"  OS: {os_name}  Arch: {arch}")
    print(f"  Model: {args.model}")
    print(f"  Quantize: {args.quantize or 'no'}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'=' * 50}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Clone whisper.cpp
    log_step("Step 1/4 — Clone whisper.cpp (for conversion + build)")
    clone_whisper_cpp()

    # Step 2 — Download .pt model
    log_step("Step 2/4 — Download official OpenAI model (.pt)")
    pt_path = download_pt_model(args.model)

    # Step 3 — Convert .pt → ggml (pure Python, no cmake)
    log_step("Step 3/4 — Convert .pt → ggml format (pure Python)")
    ggml_path = convert_pt_to_ggml(pt_path, args.model)

    # Step 4 — Get stream binary
    log_step("Step 4/4 — Obtain whisper-stream binary")
    stream_bin = obtain_stream_binary(
        os_name,
        arch,
        force_build=getattr(args, "force_build", False),
    )

    # Optional: quantize
    final_model = ggml_path
    if args.quantize:
        final_model = quantize_model(ggml_path, args.quantize)

    # Generate run scripts
    log_step("Generating launcher scripts …")
    generate_run_scripts(final_model, stream_bin, os_name)

    # Done
    threads = max(1, min(cpu_count_safe(), 8))
    print(f"\n{'=' * 50}")
    print(f"  {_c(_G, 'Setup complete!')}")
    print(f"{'=' * 50}")
    print()
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Model        : local/models/{final_model.name}")
    print(f"  Binary       : local/bin/{stream_bin.name}")
    print(f"  Threads      : {threads}")
    print()

    if os_name in ("darwin", "linux"):
        print(f"  Start live transcription:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    ./run-live.sh")
        print()
        print(f"  Ultra-low-latency:")
        print(f"    ./run-live-fast.sh")
        if os_name == "darwin":
            print()
            print(f"  If mic capture fails:")
            print(f"    System Settings → Privacy & Security → Microphone")
    else:
        print(f"  Start live transcription:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    .\\run-live.ps1")
        print()
        print(f"  Ultra-low-latency:")
        print(f"    .\\run-live-fast.ps1")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup whisper.cpp for live transcription. "
        "Downloads models from official OpenAI Azure CDN (no HuggingFace).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python setup.py
              python setup.py --model base.en --quantize q5_0
              python setup.py --model small.en
        """),
    )

    parser.add_argument(
        "--model",
        default="tiny.en",
        choices=sorted(OPENAI_MODEL_URLS.keys()),
        help="Whisper model to download (default: tiny.en).",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=["q8_0", "q5_0", "q5_1", "q4_0", "q4_1"],
        help="Optional quantization for faster CPU inference.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force building from source even on Windows.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup(args)


if __name__ == "__main__":
    main()
