#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# whisper-live · macOS setup
#
# This script:
#   1. Validates the environment (macOS 12+, Python 3.10–3.13, git, cmake, SDL2)
#   2. Creates a temporary virtual-env for heavy conversion dependencies
#   3. Installs torch + openai-whisper + numpy (one-time, for .pt → ggml)
#   4. Runs scripts/setup.py (downloads from OpenAI Azure CDN only)
#   5. Tears down the temporary venv
#
# After setup completes, the runtime needs ZERO Python dependencies —
# it's pure whisper.cpp (C/C++).
#
# Everything is stored under <project_root>/local/ (gitignored).
#
# *** No HuggingFace URLs are ever contacted. ***
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  _B='\033[1m'  _G='\033[32m'  _Y='\033[33m'  _R='\033[31m'  _0='\033[0m'
else
  _B='' _G='' _Y='' _R='' _0=''
fi

log()  { printf "${_G}  ✓${_0} %s\n" "$*"; }
warn() { printf "${_Y}  ⚠${_0} %s\n" "$*"; }
err()  { printf "${_R}  ✗${_0} %s\n" "$*" >&2; }
step() { printf "\n${_B}▸${_0} %s\n" "$*"; }

fatal() { err "$@"; exit 1; }

# ── Locate project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SETUP_PY="${SCRIPT_DIR}/setup.py"

if [[ ! -f "$SETUP_PY" ]]; then
  fatal "Cannot find setup.py at ${SETUP_PY}"
fi

# ── Parse arguments (pass-through to setup.py) ──────────────────────────────
MODEL="tiny.en"
QUANTIZE=""
EXTRA_ARGS=()

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model NAME       Whisper model name (default: tiny.en)
                     Choices: tiny.en tiny base.en base small.en small
                              medium.en medium large-v1 large-v2 large-v3 large
  --quantize TYPE    Quantization type (q8_0, q5_0, q5_1, q4_0, q4_1)
  --help             Show this help and exit

Everything is installed under <project_root>/local/ (gitignored).

Examples:
  ./setup-mac.sh
  ./setup-mac.sh --model base.en --quantize q8_0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --quantize)
      QUANTIZE="${2:?--quantize requires a value}"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── Validate: macOS ──────────────────────────────────────────────────────────
step "Checking environment …"

if [[ "$(uname -s)" != "Darwin" ]]; then
  fatal "This script is for macOS only. Use setup-windows.ps1 on Windows."
fi

mac_version="$(sw_vers -productVersion)"
mac_major="${mac_version%%.*}"
if [[ -z "$mac_major" ]] || (( mac_major < 12 )); then
  fatal "macOS 12+ required (detected ${mac_version})."
fi
log "macOS ${mac_version} ($(uname -m))"

# ── Validate: git ────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
  fatal "git is not installed. Install Xcode CLT:  xcode-select --install"
fi
log "git found: $(git --version)"

# ── Validate: cmake ──────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
  fatal "cmake is not installed. Install with:  brew install cmake"
fi
log "cmake found: $(cmake --version | head -1)"

# ── Validate: SDL2 ───────────────────────────────────────────────────────────
SDL2_OK=false
if command -v sdl2-config &>/dev/null; then
  SDL2_OK=true
elif [[ -d /opt/homebrew/include/SDL2 ]] || [[ -d /usr/local/include/SDL2 ]]; then
  SDL2_OK=true
fi

if [[ "$SDL2_OK" != "true" ]]; then
  fatal "SDL2 is required for the live mic capture binary.\n       Install with:  brew install sdl2"
fi
log "SDL2 found"

# ── Validate: Python 3.10–3.13 (torch doesn't support 3.14 yet) ─────────────
PYTHON=""
MAX_MINOR=13
for candidate in python3.12 python3.13 python3.11 python3.10 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    py_version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ -n "$py_version" ]]; then
      py_major="${py_version%%.*}"
      py_minor="${py_version#*.}"
      if (( py_major == 3 )) && (( py_minor >= 10 )) && (( py_minor <= MAX_MINOR )); then
        PYTHON="$candidate"
        break
      fi
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  fatal "Python 3.10–3.${MAX_MINOR} is required but not found on PATH.\n       Install via: brew install python@3.13\n       (Python 3.14+ is not yet supported by torch/openai-whisper)"
fi
log "Python: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

# ── Create temporary venv ────────────────────────────────────────────────────
step "Creating temporary Python venv for conversion dependencies …"

VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/whisper-setup-venv.XXXXXX")"

cleanup_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    log "Cleaning up temporary venv at ${VENV_DIR} …"
    rm -rf "$VENV_DIR"
  fi
}
trap cleanup_venv EXIT

"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Temporary venv → ${VENV_DIR}"

# ── Install conversion dependencies ─────────────────────────────────────────
step "Installing conversion dependencies (torch, numpy, openai-whisper) …"
log "This is a one-time cost. The runtime needs no Python."

pip install --quiet --upgrade pip wheel 2>/dev/null || true

pip install --quiet "numpy>=1.24" || fatal "Failed to install numpy."
log "numpy installed"

pip install --quiet "torch>=2.0" --index-url "https://download.pytorch.org/whl/cpu" 2>/dev/null \
  || pip install --quiet "torch>=2.0" \
  || fatal "Failed to install torch."
log "torch installed"

pip install --quiet "openai-whisper" || fatal "Failed to install openai-whisper."
log "openai-whisper installed"

# ── Verify no HuggingFace leakage ───────────────────────────────────────────
step "Verifying dependency integrity …"
if grep -iE 'https?://[^ ]*huggingface|https?://[^ ]*hf\.co|https?://[^ ]*hf-mirror' "$SETUP_PY"; then
  fatal "INTEGRITY CHECK FAILED: setup.py contains HuggingFace download URLs!"
fi
log "No HuggingFace download URLs found in setup script ✓"

# ── Build setup.py arguments ─────────────────────────────────────────────────
SETUP_ARGS=("--model" "$MODEL")

if [[ -n "$QUANTIZE" ]]; then
  SETUP_ARGS+=("--quantize" "$QUANTIZE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  SETUP_ARGS+=("${EXTRA_ARGS[@]}")
fi

# ── Run the main setup ──────────────────────────────────────────────────────
step "Running setup.py …"
log "Arguments: ${SETUP_ARGS[*]}"
echo ""

python "${SETUP_PY}" "${SETUP_ARGS[@]}"
setup_exit=$?

if [[ $setup_exit -ne 0 ]]; then
  fatal "setup.py exited with code ${setup_exit}"
fi

# ── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${_B}macOS setup complete!${_0}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Quick start:"
echo "    cd ${PROJECT_ROOT}"
echo "    ./run-live.sh"
echo ""
echo "  Ultra-low-latency:"
echo "    ./run-live-fast.sh"
echo ""
echo "  If microphone capture fails:"
echo "    System Settings → Privacy & Security → Microphone"
echo "    Grant access to Terminal / iTerm / your shell app."
echo ""
echo "  The temporary venv will be removed automatically."
echo ""
