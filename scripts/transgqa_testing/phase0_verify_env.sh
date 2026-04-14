#!/usr/bin/env bash
# TransGQA Phase 0: verify Python + CUDA + Triton + vllm package import.
# Run from repo root or anywhere after setting VLLM_ROOT to the checkout.

set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# Python imports use package name ``vllm``; if the checkout folder is not
# literally named ``vllm``, expose it via a parent directory + symlink.
VLLM_PKG_PARENT="${VLLM_PKG_PARENT:-/tmp/vllm_transgqa_pkg}"
mkdir -p "$VLLM_PKG_PARENT"
ln -sfn "$VLLM_ROOT" "$VLLM_PKG_PARENT/vllm"
export PYTHONPATH="${VLLM_PKG_PARENT}:${PYTHONPATH:-}"

echo "VLLM_ROOT=$VLLM_ROOT"
echo "PYTHONPATH=$PYTHONPATH (vllm -> $VLLM_ROOT)"
echo ""

python3 <<'PY'
import sys
print("python:", sys.version)

import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available (kernel tests will fail)")

try:
    import triton
    print("triton:", triton.__version__)
except ImportError as e:
    print("WARN: triton not installed:", e)

try:
    from vllm.v1.attention.ops.sparse_gqa import sparse_gqa_attention
    print("import sparse_gqa_attention: OK")
except ImportError as e:
    print("WARN: sparse_gqa import failed:", e)

try:
    from vllm.utils import has_tilelang
    print("has_tilelang:", has_tilelang())
except Exception as e:
    print("has_tilelang check failed:", e)

print("\nPhase 0 checks finished.")
PY
