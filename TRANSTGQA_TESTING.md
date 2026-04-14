# TransGQA 分阶段测试（云服务器 / 本地）

导入路径为 ``vllm.*``。若仓库目录名不是 ``vllm``，请用父目录 + 软链接（脚本已自动处理）：

```bash
export VLLM_ROOT=/path/to/vllm_transgqa
mkdir -p /tmp/vllm_pkg && ln -sfn "$VLLM_ROOT" /tmp/vllm_pkg/vllm
export PYTHONPATH=/tmp/vllm_pkg:$PYTHONPATH
```

或运行 ``bash scripts/transgqa_testing/phase0_verify_env.sh``（内部会创建 ``/tmp/vllm_transgqa_pkg/vllm``）。

## Phase 0 — 环境

```bash
bash scripts/transgqa_testing/phase0_verify_env.sh
```

或手动：`python -c "import torch; print(torch.cuda.get_device_name(0))"` 等。

## Phase 1 — Triton sparse GQA

需要 **CUDA**。

```bash
python v1/attention/ops/test_sparse_gqa.py
```

## Phase 2 — TileLang sparse GQA（可选）

需要 **CUDA** + `pip install tilelang`。未安装时会打印 SKIP 并以 0 退出。

**无需编译 `vllm._C`**：`sparse_gqa_tilelang.py` 已避免在 import 时拉取 `forward_context` / `platforms`（否则会链到 `vllm._C`）。仅跑本测试时只需源码树 + `PYTHONPATH`（见文首）+ PyTorch + CUDA + tilelang。完整推理仍建议 `pip install -e .` 以编译扩展。

**对齐对象**：当前脚本将 TileLang 输出与 **PyTorch reference**（与 `test_sparse_gqa.py` 中同一套数学）对比，而不是与 Triton kernel 逐字节对比；若两者都正确，应都与 reference 接近。

**优先只测 TileLang**（不跑 Triton）：

```bash
bash scripts/transgqa_testing/run_tilelang_only.sh
```

或手动（需先设置 `PYTHONPATH`，见文首）：

```bash
python v1/attention/ops/test_sparse_gqa_tilelang.py
```

## Phase 3 — Indexer 数学（CPU 可跑）

```bash
python v1/attention/ops/test_indexer_math.py
```

## Phase 4 — Checkpoint 目录

```bash
python scripts/transgqa_testing/verify_checkpoint.py --model-dir /path/to/model
# 若仅有 index.json、权重在别处：
python scripts/transgqa_testing/verify_checkpoint.py --model-dir /path/to/model \
  --allow-missing-shards
```

## Phase 5 — 端到端最小生成

需要完整权重目录与 GPU（大模型需多卡 `--tp`）。

```bash
python scripts/transgqa_testing/e2e_minimal.py --model /path/to/model --tp 2
```

## Phase 6 — 延迟基准（可选）

```bash
bash scripts/transgqa_testing/benchmark_optional.sh /path/to/model 2
```

## 一键跑 Phase 1–3（及可选 Phase 4）

```bash
bash scripts/transgqa_testing/run_all_tests.sh
# 若要做 Phase 4：
export TRANSGQA_MODEL_DIR=/path/to/model
bash scripts/transgqa_testing/run_all_tests.sh
```
