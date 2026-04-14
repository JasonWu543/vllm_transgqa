#!/usr/bin/env python3
"""TransGQA Phase 5: minimal end-to-end load + one generate (requires GPU + weights).

Usage:
  export PYTHONPATH=/path/to/vllm_transgqa:$PYTHONPATH
  python scripts/transgqa_testing/e2e_minimal.py --model /path/to/model --tp 2
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="HF model directory")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    if not os.path.isdir(args.model):
        print(f"ERROR: model path is not a directory: {args.model}", file=sys.stderr)
        return 1

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print("ERROR: cannot import vllm:", e, file=sys.stderr)
        return 1

    print("Loading model (this may take several minutes)...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    out = llm.generate(["Hello, world!"], sp)
    text = out[0].outputs[0].text
    print("generated:", repr(text))
    print("Phase 5 e2e: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
