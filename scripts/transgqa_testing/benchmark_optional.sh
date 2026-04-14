#!/usr/bin/env bash
# TransGQA Phase 6 (optional): latency benchmark entrypoint.
# Requires a valid model directory and GPU.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG_PARENT="${PKG_PARENT:-/tmp/vllm_transgqa_pkg_bench}"
mkdir -p "$PKG_PARENT"
ln -sfn "$ROOT" "$PKG_PARENT/vllm"
export PYTHONPATH="${PKG_PARENT}:${PYTHONPATH:-}"

MODEL="${1:?Usage: $0 /path/to/model [tensor_parallel_size]}"
TP="${2:-1}"

exec python3 -m vllm.entrypoints.cli.benchmark.latency \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --input-len 4096 \
  --output-len 128
