#!/usr/bin/env python3
"""TransGQA Phase 4: verify model directory layout (config + index + shards).

Usage:
  python scripts/transgqa_testing/verify_checkpoint.py --model-dir /path/to/model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and model shards",
    )
    p.add_argument(
        "--allow-missing-shards",
        action="store_true",
        help="Only validate index.json keys; do not fail if .safetensors files are absent",
    )
    args = p.parse_args()
    d: Path = args.model_dir.resolve()
    if not d.is_dir():
        print(f"ERROR: not a directory: {d}", file=sys.stderr)
        return 1

    cfg = d / "config.json"
    idx = d / "model.safetensors.index.json"
    if not cfg.is_file():
        print(f"WARN: missing {cfg} (skipping HF config checks)")
    if not idx.is_file():
        print(f"WARN: missing {idx} (optional if single-file weights)")

    if cfg.is_file():
        with cfg.open() as f:
            meta = json.load(f)
    else:
        meta = {}
    if meta:
        arch = meta.get("architectures", [])
        print("architectures:", arch)
        for k in ("index_topk", "freqfold", "rank_of_wo", "use_hisa"):
            if k in meta:
                print(f"  {k}: {meta[k]}")

    if idx.is_file():
        with idx.open() as f:
            wm = json.load(f)
        weight_map = wm.get("weight_map", {})
        keys = list(weight_map.keys())
        indexer = [k for k in keys if "indexer" in k]
        print(f"indexer-related keys: {len(indexer)}")
        if indexer:
            print("  sample:", indexer[:3])
        shard_names = set(weight_map.values())
        missing = []
        for shard in sorted(shard_names):
            sp = d / shard
            if not sp.is_file():
                missing.append(str(sp))
        if missing:
            if args.allow_missing_shards:
                print(f"WARN: {len(missing)} shard files missing (allowed by flag)")
            else:
                print("ERROR: shard files missing on disk:")
                for m in missing[:20]:
                    print(" ", m)
                if len(missing) > 20:
                    print(f"  ... and {len(missing) - 20} more")
                return 1
        else:
            print(f"shard files present: {len(shard_names)} unique filenames")

    if not cfg.is_file() and not idx.is_file():
        print("ERROR: need at least config.json or model.safetensors.index.json",
              file=sys.stderr)
        return 1

    print("Phase 4 verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
