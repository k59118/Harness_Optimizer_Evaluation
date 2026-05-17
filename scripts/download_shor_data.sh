#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
TMP_PARENT="$REPO_ROOT/.tmp"
REPO_ID="LangAGI-Lab/SHOR"
ARCHIVE_NAME="raw_artifacts.tar.zst"

if [[ -e "$DATA_DIR" ]]; then
  echo "data directory already exists: $DATA_DIR"
  echo "Remove or move it before running this script."
  exit 0
fi

mkdir -p "$TMP_PARENT"

# Some networks stall on Hugging Face's Xet/CAS download backend at 0 bytes.
# Prefer the regular Hub/LFS download path unless the caller explicitly opts in.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf)
elif command -v uvx >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMP_PARENT/uv-cache}"
  HF_CMD=(uvx --from huggingface_hub hf)
elif command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMP_PARENT/uv-cache}"
  HF_CMD=(uv tool run --from huggingface_hub hf)
else
  echo "Missing Hugging Face CLI. Install it with: uv tool install huggingface_hub" >&2
  exit 1
fi

for cmd in tar; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

DOWNLOAD_DIR="$TMP_PARENT/shor_raw_artifacts"
ARCHIVE="$DOWNLOAD_DIR/$ARCHIVE_NAME"
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading SHOR raw artifacts from Hugging Face..."
set +e
"${HF_CMD[@]}" download "$REPO_ID" "$ARCHIVE_NAME" --repo-type dataset --local-dir "$DOWNLOAD_DIR"
download_status=$?
set -e

if [[ "$download_status" -eq 130 ]]; then
  echo "Download interrupted. Re-run this script to resume."
  exit 130
fi

if [[ "$download_status" -ne 0 ]]; then
  echo "Hugging Face download failed. Starting Hugging Face login..."
  "${HF_CMD[@]}" auth login
  "${HF_CMD[@]}" download "$REPO_ID" "$ARCHIVE_NAME" --repo-type dataset --local-dir "$DOWNLOAD_DIR"
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Download completed, but archive was not found: $ARCHIVE" >&2
  exit 1
fi

echo "Extracting archive into repository root..."
tar -xf "$ARCHIVE" -C "$REPO_ROOT"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Archive extraction completed, but data directory was not created." >&2
  exit 1
fi

rm -rf "$DOWNLOAD_DIR"
echo "Created data directory: $DATA_DIR"
