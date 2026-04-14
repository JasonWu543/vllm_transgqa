# TransGQA: MLA Indexer + GQA Sparse Attention for vLLM

## 概述

TransGQA 是一个在 vLLM 框架中实现的**两阶段稀疏注意力模型**，结合了：

- **MLA-based Indexer**：复用 DeepSeek V3.2 的 MLA 索引器结构和 DSA kernel，使用 FP8 量化的 K cache 进行高效的近似注意力打分，选出 top-k 最重要的 key 位置。
- **GQA Sparse Attention**：主注意力使用标准的 Grouped Query Attention (GQA) 投影，但只对 indexer 选出的 top-k 位置进行注意力计算，通过自定义 Triton kernel 实现 on-the-fly 从 paged KV cache 中 gather K/V。

整体架构基于 Qwen3 模型，额外加入了稀疏注意力 indexer 模块。

---

## 修改和新增的文件

### 1. `vllm/v1/attention/ops/sparse_gqa.py` （新建文件）

**用途**：fused sparse GQA attention 的 Triton kernel。

**核心设计**：
- 给定每个 token 的 top-k 个全局 cache slot 索引（由 indexer 产出），kernel 从 paged KV cache 中 on-the-fly gather K/V 并计算注意力，避免 materialize 完整的 gathered KV 张量。
- 使用 **online softmax** 保证数值稳定性，逐 tile 流式更新 `(m, l, acc)` 三元组。
- Grid 维度为 `(num_tokens, num_heads)`，每个 program 处理一个 `(token, head)` 对，在 top-k 维度上以 `BLOCK_TOPK` 为 tile 迭代。

**包含内容**：

| 组件 | 说明 |
|------|------|
| `_sparse_gqa_attention_kernel` | Triton JIT kernel，参数包括 Q/KC/VC/O/IDX 指针及各种 stride |
| `sparse_gqa_attention()` | Python 入口函数，计算 grid/block 参数并启动 kernel |

**关键参数**：
- `q`: `[num_tokens, num_heads, head_dim]` — query 张量 (bf16)
- `key_cache`: `[total_slots, num_kv_heads, head_dim]` — 展平的 paged K cache
- `value_cache`: `[total_slots, num_kv_heads, head_dim]` — 展平的 paged V cache
- `global_indices`: `[num_tokens, topk_k]` int32 — 全局 cache slot 索引，-1 表示无效
- `gqa_group_size`: `num_heads // num_kv_heads` — GQA 分组大小，用于将 Q head 映射到 KV head

---

### 2. `vllm/model_executor/models/transgqa.py` （新建文件）

**用途**：TransGQA 模型的完整实现，从模型结构到权重加载。

**文件结构**：

```
transgqa.py
├── TransGQAMLP                         # MLP 层（同 Qwen3）
├── TransGQAIndexerCache                # Indexer 的 FP8 K cache（AttentionLayerBase）
├── cp_gather_indexer_k_quant_cache()   # 从 paged cache 收集 FP8 K（CPU 参考实现）
├── transgqa_sparse_attn_indexer()      # Flat sparse indexer 自定义 op
├── transgqa_hierarchy_sparse_attn_indexer()  # 层级 sparse indexer 自定义 op
├── TransGQAIndexer                     # Indexer 模块（独立权重）
├── _build_req_id_per_token()           # 辅助：token→request 映射
├── _pad_topk_indices()                 # 辅助：padding topk indices
├── transgqa_sparse_gqa_forward()       # Sparse GQA forward 自定义 op
├── TransGQAAttention                   # 注意力层（Indexer + GQA）
├── TransGQADecoderLayer                # 解码器层
├── TransGQAModel                       # 模型主体（@support_torch_compile）
└── TransGQAForCausalLM                 # CausalLM 包装（SupportsPP）
```

#### 2.1 `TransGQAIndexerCache`（L120-152）

继承 `torch.nn.Module` + `AttentionLayerBase`，用于 indexer 的 FP8 K cache 管理。

| 方法 | 说明 |
|------|------|
| `__init__` | 将自身注册到 `compilation_config.static_forward_context`，使 vLLM 的 KV cache 管理系统能发现它 |
| `get_kv_cache_spec(vllm_config)` | 返回 `MLAAttentionSpec`，告诉 vLLM 为此层分配 `[num_blocks, block_size, head_dim]` 形状的 cache（`dtype=uint8`） |
| `get_attn_backend()` | 返回 `DeepseekV32IndexerBackend`，复用 DSA 的 indexer metadata builder |

#### 2.2 Indexer 自定义 Op（L216-555）

两个 vLLM 自定义 op，通过 `direct_register_custom_op` 注册以兼容 `torch.compile`：

**`transgqa_sparse_attn_indexer`** — Flat 模式：
1. 使用 `ops.indexer_k_quant_and_cache` 将新 K 写入 FP8 cache
2. Prefill：从 paged cache gather FP8 K，调用 `fp8_mqa_logits` 计算近似注意力分数，取 topk
3. Decode：调用 `fp8_paged_mqa_logits` 做 paged MQA logits，取 topk
4. 返回 `topk_indices_buffer: [max_tokens, topk_k]`

**`transgqa_hierarchy_sparse_attn_indexer`** — 层级模式 (HISA)：
1. 同样先写入 FP8 cache
2. 先在 block 级别粗筛（`fp8_hierarchy_mqa_logits` / `fp8_hierarchy_paged_mqa_logits`），选出 top-k blocks
3. 再在 block 内部精选 token 级别的 top-k
4. 返回 `topk_indices_buffer`

#### 2.3 `TransGQAIndexer`（L562-724）

独立权重的 MLA-style indexer 模块：

| 组件 | 说明 |
|------|------|
| `wq` | `ReplicatedLinear(hidden_size → head_dim * n_head)` — indexer 的 Q 投影 |
| `wk` | `ReplicatedLinear(hidden_size → head_dim)` — indexer 的 K 投影 |
| `k_norm` | `RMSNorm` — 对 K 的 nope 部分做归一化 |
| `weights_proj` | `ReplicatedLinear(hidden_size → n_head)` — 注意力权重投影 |
| `k_cache` | `TransGQAIndexerCache` 实例 |

`forward()` 流程：
```
hidden_states → wq → split(rope, nope) → rotary_emb → FP8 quant → q_fp8
hidden_states → wk → split(rope, nope) → k_norm → rotary_emb → k
hidden_states → weights_proj → abs importance weights
→ transgqa_sparse_attn_indexer / transgqa_hierarchy_sparse_attn_indexer
→ topk_indices_buffer
```

#### 2.4 `transgqa_sparse_gqa_forward` 自定义 Op（L763-861）

Sparse GQA attention 的核心自定义 op，六步流程：

```
Step 1: 通过 get_attention_context() 获取 attn_meta/attn_layer/kv_cache/slot_mapping
Step 2: 调用 attn_layer.impl.do_kv_cache_update() 更新 paged KV cache
Step 3: 展平 key_cache/value_cache 为 [total_slots, num_kv_heads, head_dim]
Step 4: 构建 req_id_per_token（token→request 映射）
Step 5: triton_convert_req_index_to_global_index() 将 request-local 索引转为全局 cache slot 索引
Step 6: 调用 sparse_gqa_attention() Triton kernel 计算稀疏注意力
```

#### 2.5 `TransGQAAttention`（L868-1012）

主注意力层，forward 流程：

```python
# Step 1: Indexer 选出 top-k
topk_indices = self.indexer(hidden_states, positions, self.rotary_emb)

# Step 2: 标准 QKV 投影 + RoPE
qkv → split(q, k, v) → q_norm, k_norm → rotary_emb

# Step 3: 分支
if has_valid_sparse:
    v_3d = v.view(-1, num_kv_heads, head_dim)   # ← 修复：reshape 为 3D
    → torch.ops.vllm.transgqa_sparse_gqa_forward(q, k, v_3d, topk_indices, ...)
else:
    → self.attn(q_flat, k_flat, v)  # 标准 Attention 作为 fallback
```

#### 2.6 `TransGQAModel` / `TransGQAForCausalLM`（L1108-1319）

- `TransGQAModel`：带 `@support_torch_compile` 的模型主体，创建共享的 `topk_indices_buffer`，在 `__init__` 中强制 `block_size >= 64`
- `TransGQAForCausalLM`：CausalLM 包装，实现 `load_weights()` 处理 stacked params 和 indexer 独立权重

---

### 3. `vllm/model_executor/models/registry.py` （修改）

**修改内容**：在 `_TEXT_GENERATION_MODELS` 字典中添加一行：

```python
"TransGQAForCausalLM": ("transgqa", "TransGQAForCausalLM"),
```

**作用**：当 HuggingFace config.json 中 `architectures` 包含 `"TransGQAForCausalLM"` 时，vLLM 的 `ModelRegistry.resolve_model_cls()` 能找到并懒加载 `vllm.model_executor.models.transgqa.TransGQAForCausalLM`。

---

## 不需要修改的文件及原因

| 文件 | 原因 |
|------|------|
| `transformers_utils/model_arch_config_convertor.py` | TransGQA 使用 `Qwen3Config`（`model_type="qwen3"`），不在 `is_deepseek_mla` 白名单中，`use_mla=False` 是正确的（主注意力是 GQA） |
| `model_executor/models/__init__.py` | 模型通过 `ModelRegistry` 懒加载，无需显式导出 |
| `v1/attention/selector.py` | Indexer 的 backend 通过 `TransGQAIndexerCache.get_attn_backend()` 逐层返回，不走全局 selector |
| `v1/attention/backends/registry.py` | `DeepseekV32IndexerBackend` 不在 `AttentionBackendEnum` 中（和 DSA 相同），通过逐层 `get_attn_backend()` 解析 |
| `v1/core/kv_cache_utils.py` | `MLAAttentionSpec` 继承自 `FullAttentionSpec`，`isinstance` 检查兼容，两种 spec 可以合并到同一个 `UniformTypeKVCacheSpecs` 组 |
| `config/model.py` | `use_mla=False` 不影响 TransGQA，block_size 在模型 `__init__` 中已强制 ≥ 64 |

---

## 关键设计决策

### 为什么不能直接复用 MLA 的 sparse attention？

MLA 的 sparse attention（`FlashMLASparseImpl`）针对 MLA 特有的**紧凑 KV cache**（单头，K 和 V 拼接为一个向量），使用 `flash_mla_with_kvcache` 或 `flash_mla_sparse_fwd` kernel。GQA 使用的是**标准分页 KV cache**（多 KV 头，K 和 V 分别存储），格式完全不同，无法复用 MLA 的 attention kernel。

### 为什么 Indexer 可以完全复用？

Indexer 只负责"选哪些 key"，它使用自己独立的 FP8 K cache（`MLAAttentionSpec`，单头），与主注意力的 KV cache 格式无关。Indexer 的所有 kernel（`indexer_k_quant_and_cache`、`fp8_mqa_logits`、`fp8_paged_mqa_logits`、`fp8_hierarchy_*`）都操作 indexer 自己的 cache，因此可以完全复用 DSA 的实现。

### KV Cache 共存方案

TransGQA 每层有两个 `AttentionLayerBase` 注册在 `static_forward_context` 中：

```
model.layers.0.self_attn.indexer.k_cache  → MLAAttentionSpec (FP8, uint8)
model.layers.0.self_attn.attn             → FullAttentionSpec (GQA, bf16)
```

- 两者都继承 `FullAttentionSpec`，`block_size=64`（已强制），所以被 `UniformTypeKVCacheSpecs.is_uniform_type()` 判定为同一类型
- 合并到**同一个 KV cache group**，共享 block pool 和 block table
- 各层的 cache 张量形状根据各自 spec 独立分配
- 后端分别通过 `get_attn_backend()` 解析：indexer → `DeepseekV32IndexerBackend`，GQA → `FlashAttentionBackend`

### block_size 约束

`DeepseekV32IndexerBackend.get_supported_kernel_block_sizes()` 在 CUDA 上返回 `[64]`。由于 `use_mla=False`，vLLM 的默认 block_size 对齐逻辑（`kernel_block_alignment_size=16`）不会强制到 64。因此在 `TransGQAModel.__init__` 中手动设置 `cache_config.block_size = max(block_size, 64)`。

---

## 使用方式

### config.json 示例

```json
{
  "architectures": ["TransGQAForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_hidden_layers": 32,
  "intermediate_size": 11008,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "max_position_embeddings": 131072,
  "rope_theta": 1000000.0,
  "vocab_size": 151936,

  "index_topk": 256,
  "freqfold": 2,
  "rank_of_wo": 4,

  "use_hisa": false,
  "hisa_k_block_size": 64,
  "hisa_block_topk": 8
}
```

### 启动命令

```bash
vllm serve /path/to/transgqa-model
```

vLLM 会根据 `architectures: ["TransGQAForCausalLM"]` 自动找到实现类，完成模型加载、双 KV cache 分配和推理。

---

## 数据流总览

```
                          hidden_states
                               │
                           qkv_proj
                               │
                          q, k, v  (local heads)
                               │
                        q_norm, k_norm
                               │
                    ┌──────────┼──────────┐
                    │ q_orig   │          │ k_orig
                    │ k_orig   │          │ (pre-RoPE)
                    │          │          │
                    │     rotary_emb      │
                    │          │          │
                    │       q, k (post-RoPE)
                    │          │
              TransGQAIndexer  │
    ┌─────────────────────┐    │
    │ Rope path:          │    │
    │   q,k → freq-fold   │    │
    │   → einsum(qk_proj) │    │
    │   → [:ff//2]        │    │
    │ Nope path:          │    │
    │   q_orig,k_orig     │    │
    │   → freq-fold       │    │
    │   → einsum(qk_proj) │    │
    │   → [ff//2:]        │    │
    │   → qk_nope_proj    │    │
    │ cat(rope, nope)     │    │
    │ → FP8 quant         │    │
    │ → DSA indexer       │    │
    │ → topk_indices ─────┼──→ │
    └─────────────────────┘    │
                          ┌────┴────┐
                        sparse   full_attn
                          │     (fallback)
               transgqa_sparse_gqa_forward
                          │
                          ↓
                   attn_output → o_proj → output
```
