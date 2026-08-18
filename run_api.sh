#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_ID="${GPU_ID:-0}"
NO_TUNNEL="${NO_TUNNEL:-0}"
API_PID=""
TUNNEL_PID=""
RUNTIME_DIR="$ROOT_DIR/.runtime"
TUNNEL_LOG="$RUNTIME_DIR/cloudflared.log"

usage() {
  cat <<'EOF'
Usage: bash run_api.sh [--port PORT] [--no-tunnel]

Options:
  --port PORT    Listen on this port (default: 8000, or the PORT environment variable)
  --no-tunnel    Start only the local API
  -h, --help     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "Error: --port requires a value." >&2; usage >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --no-tunnel)
      NO_TUNNEL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Error: port must be an integer from 1 through 65535 (received: $PORT)." >&2
  exit 2
fi

# Initialize Conda in this shell before checking imports or starting Uvicorn.
# shellcheck source=path.sh
source "$ROOT_DIR/path.sh"

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping LoMTL API and tunnel..."
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$TUNNEL_PID" ]] && wait "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

command -v python >/dev/null || { echo "Error: python is not available." >&2; exit 1; }
command -v curl >/dev/null || { echo "Error: curl is required for readiness checks and downloads." >&2; exit 1; }
python -c 'import fastapi, peft, torch, transformers, uvicorn' 2>/dev/null || {
  echo "Error: Python runtime dependencies are missing. Run: pip install -r requirements.txt" >&2
  exit 1
}

mkdir -p "$RUNTIME_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_COMPILE_DISABLE=1

python -m uvicorn api.main:app --host "$HOST" --port "$PORT" &
API_PID=$!

echo "Loading google/gemma-2-2b-it and LoMTL adapter on GPU ${GPU_ID}..."
for _ in $(seq 1 "${STARTUP_TIMEOUT:-600}"); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" || true
    echo "Error: API exited before becoming ready. Check the log above." >&2
    exit 1
  fi
  if curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "Error: API did not become ready within ${STARTUP_TIMEOUT:-600} seconds." >&2
  exit 1
fi

PUBLIC_URL=""
if [[ "$NO_TUNNEL" != "1" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED="$(command -v cloudflared)"
  else
    CLOUDFLARED="$RUNTIME_DIR/cloudflared"
    if [[ ! -x "$CLOUDFLARED" ]]; then
      case "$(uname -m)" in
        x86_64|amd64) CF_ARCH="amd64" ;;
        aarch64|arm64) CF_ARCH="arm64" ;;
        *) echo "Error: unsupported architecture $(uname -m); install cloudflared in PATH." >&2; exit 1 ;;
      esac
      echo "Downloading the official cloudflared binary..."
      DOWNLOAD_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"
      if ! curl --fail --location --retry 3 --output "$CLOUDFLARED.tmp" "$DOWNLOAD_URL"; then
        rm -f "$CLOUDFLARED.tmp"
        echo "Error: cloudflared download failed. Install cloudflared in PATH and retry." >&2
        exit 1
      fi
      mv "$CLOUDFLARED.tmp" "$CLOUDFLARED"
      chmod +x "$CLOUDFLARED"
    fi
  fi

  : >"$TUNNEL_LOG"
  "$CLOUDFLARED" tunnel --url "http://${HOST}:${PORT}" --no-autoupdate >"$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 1 60); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      cat "$TUNNEL_LOG" >&2
      echo "Error: cloudflared exited before creating a Quick Tunnel." >&2
      exit 1
    fi
    PUBLIC_URL="$(sed -nE 's#.*(https://[a-zA-Z0-9-]+\.trycloudflare\.com).*#\1#p' "$TUNNEL_LOG" | head -n 1)"
    [[ -n "$PUBLIC_URL" ]] && break
    sleep 1
  done
  if [[ -z "$PUBLIC_URL" ]]; then
    cat "$TUNNEL_LOG" >&2
    echo "Error: cloudflared did not provide a public URL within 60 seconds." >&2
    exit 1
  fi
fi

cat <<EOF
==============================================
 AITutor-EvalKit LoMTL API
==============================================
Model: google/gemma-2-2b-it
Evaluator: LoMTL
GPU: ${GPU_ID}
API status: ready

Local API: http://${HOST}:${PORT}
Local Swagger: http://${HOST}:${PORT}/docs
EOF
if [[ -n "$PUBLIC_URL" ]]; then
  echo
  echo "Cloudflare random public URL (ready):"
  echo "${PUBLIC_URL}"
  echo
  echo "Public Swagger: ${PUBLIC_URL}/docs"
  echo "Remote test: python scripts/test_api.py --base-url ${PUBLIC_URL}"
else
  echo "Cloudflare: disabled (local-only mode)"
fi
echo "=============================================="
echo "Press Ctrl+C to stop all processes."

wait "$API_PID"
