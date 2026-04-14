#!/usr/bin/env bash
# Run only TileLang sparse GQA vs PyTorch reference (Phase 2 first on GPU).
# Usage: bash scripts/transgqa_testing/run_tilelang_only.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG_PARENT="${PKG_PARENT:-/tmp/vllm_transgqa_pkg_tl}"
mkdir -p "$PKG_PARENT"
ln -sfn "$ROOT" "$PKG_PARENT/vllm"
export PYTHONPATH="${PKG_PARENT}:${PYTHONPATH:-}"
cd "$ROOT"

exec python3 v1/attention/ops/test_sparse_gqa_tilelang.py
