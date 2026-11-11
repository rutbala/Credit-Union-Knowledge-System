#!/usr/bin/env bash
# Grader dry run: clone the committed repo, install requirements into a brand-new venv,
# and run the public eval with no API keys and no dev cache.
#
#   dev/fresh_clone_check.sh [workdir] [python]
#
# Fails loudly if the clone can't install, can't load the model offline, errors on any
# question, or ingests too slowly. Snapshots are copied from this checkout (they are not in git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
PY="${2:-python3}"
echo "workdir: $WORK   python: $($PY --version 2>&1)"

rm -rf "$WORK/repo" && git clone -q "$ROOT" "$WORK/repo"
cp -R "$ROOT/snapshots_zip" "$WORK/repo/" 2>/dev/null || true      # tracked, but be safe
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/pip" install -q -r "$WORK/repo/submission/requirements.txt"

cd "$WORK/repo"
# No keys, no cache, offline model: exactly what the graders have.
env -i PATH="$PATH" HOME="$HOME" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore \
  "$WORK/venv/bin/python" harness/run_eval.py --submission submission \
  --eval eval/public_eval.jsonl --out "$WORK/results" 2>&1 | tee "$WORK/eval.log" | grep -vE "^ingest"

grep -q '"errors": 0' "$WORK/results/report.json" || { echo "FAIL: errors"; exit 1; }
grep -q '"ingest_over_limit": false' "$WORK/results/report.json" || { echo "FAIL: ingest over limit"; exit 1; }
if grep -q "using BM25 only" "$WORK/eval.log" && [ -z "${ALLOW_BM25:-}" ]; then echo "FAIL: semantic search did not load (BM25 fallback); set ALLOW_BM25=1 to accept"; exit 1; fi
echo "PASS"
