#!/usr/bin/env bash
# Run TransGQA test phases that do not require a full model checkpoint.
# Usage: from repo root,  bash scripts/transgqa_testing/run_all_tests.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG_PARENT="${PKG_PARENT:-/tmp/vllm_transgqa_pkg_run}"
mkdir -p "$PKG_PARENT"
ln -sfn "$ROOT" "$PKG_PARENT/vllm"
export PYTHONPATH="${PKG_PARENT}:${PYTHONPATH:-}"
cd "$ROOT"

echo "=== Phase 1: Triton sparse_gqa (CUDA required) ==="
python3 v1/attention/ops/test_sparse_gqa.py

echo ""
echo "=== Phase 2: TileLang sparse_gqa (optional) ==="
python3 v1/attention/ops/test_sparse_gqa_tilelang.py

echo ""
echo "=== Phase 3: Indexer math (CPU OK) ==="
python3 v1/attention/ops/test_indexer_math.py

echo ""
echo "=== Phase 4: Checkpoint layout (optional path arg) ==="
if [[ -n "${TRANSGQA_MODEL_DIR:-}" ]]; then
  EXTRA=()
  if [[ "${TRANSGQA_ALLOW_MISSING_SHARDS:-}" == "1" ]]; then
    EXTRA+=(--allow-missing-shards)
  fi
  python3 scripts/transgqa_testing/verify_checkpoint.py --model-dir "$TRANSGQA_MODEL_DIR" "${EXTRA[@]}"
else
  echo "Skip Phase 4 (set TRANSGQA_MODEL_DIR to a directory with config.json + safetensors)"
fi

echo ""
echo "All runnable phases completed."
