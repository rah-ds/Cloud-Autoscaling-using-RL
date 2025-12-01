#!/usr/bin/env bash
set -euo pipefail

# Config
ROOT_DIR="google_cluster_data"
DOCS_DIR="$ROOT_DIR/docs"
DATA_DIR="$ROOT_DIR/data_sample"
SCHEMA_URL="https://storage.googleapis.com/google-clusterdata-2019/schema.csv"
SAMPLE_URL="https://storage.googleapis.com/google-clusterdata-2019/task_usage/part-00000-of-00500.csv.gz"
SCHEMA_OUT="$DOCS_DIR/schema.csv"
SAMPLE_GZ_OUT="$DATA_DIR/part-00000-of-00500.csv.gz"
SAMPLE_CSV_OUT="$DATA_DIR/part-00000-of-00500.csv"

# Downloader selection
downloader() {
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$2" "$1"
  else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
  fi
}

echo "==> Creating directories"
mkdir -p "$DOCS_DIR" "$DATA_DIR"

echo "==> Downloading schema to $SCHEMA_OUT"
downloader "$SCHEMA_URL" "$SCHEMA_OUT"

echo "==> Downloading sample usage file to $SAMPLE_GZ_OUT"
downloader "$SAMPLE_URL" "$SAMPLE_GZ_OUT"

echo "==> Decompressing sample"
# Only gunzip if the CSV doesn't already exist
if [ -f "$SAMPLE_CSV_OUT" ]; then
  echo "    $SAMPLE_CSV_OUT already exists; skipping gunzip."
else
  gunzip -f "$SAMPLE_GZ_OUT"
fi

echo "==> Verifying files"
ls -lh "$SCHEMA_OUT" "$SAMPLE_CSV_OUT"

echo "==> Done."
echo "    Schema: $SCHEMA_OUT"
echo "    Sample: $SAMPLE_CSV_OUT"
