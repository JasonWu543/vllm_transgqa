# TransGQA

基于本仓库（vLLM 扩展）的 **TransGQA**：在 Grouped Query Attention（GQA）上实现**稀疏注意力**，用紧凑的 **QK 索引**做 top-k 候选选择，再用自定义 kernel 对选中位置做稀疏注意力计算。

---

## 核心思路（索引与稀疏 Attention）

TransGQA 为每个 token 构造低维 **QK 索引**，使得原始注意力分数近似为索引内积：

\[
q^\top k \approx \mathrm{index}_Q^\top \cdot \mathrm{index}_K
\]

从而在稀疏 attention 中只需低维内积即可做 **top-k 块选择**，显著降低候选打分开销。

索引构建将注意力分解为：

| 部分 | 含义 | 压缩要点 |
|------|------|----------|
| **RoPE 分量** | 位置相关 | 对 RoPE 后的 K 做 **Frequency Folding**（按因子 \(F\) 折叠相邻频率），再对每个频率块做 **Block-wise PCA（SVD）**；取投影的前 \(F/2\) 维作为主成分方向，拼接得到 \(\mathrm{index}^{\mathrm{rope}}\)。实部与虚部在样本维拼接后联合 PCA，以保持 RoPE 内积结构。 |
| **Non-RoPE（NoPE）分量** | 语义、与位置解耦 | 对**未加 RoPE** 的原始 K 走同样的 folding 与块投影，取后半分量拼接后，再做一步**全局 PCA** 压到目标维度，得到 \(\mathrm{index}^{\mathrm{nope}}\)。 |
| **最终 QK 索引** | — | \(\mathrm{index} = [\mathrm{index}^{\mathrm{rope}};\, \mathrm{index}^{\mathrm{nope}}]\)，整体可视为块对角线性映射，RoPE 与 NoPE 互不混叠。 |
| **Value 方向** | 块重要性辅助 | 对每头的 \(W_o^{(h)}\) 做 SVD 取 rank-\(r\) 截断得 \(T_h\)，用 \(\|v_h T_h\|_2\) 等作为 value 影响力，与 QK 索引配合做筛选。 |

**Frequency Folding** 的动机：RoPE 相邻频率相似，将相邻 \(F\) 个频率跨 GQA 头拼接后联合 PCA，比逐频率独立 PCA 保留更多方差（与 TransMLA 类方法中方差论证一致）。

实现侧：索引器侧复用 **MLA 风格 FP8 K cache + DSA 类 indexer kernel** 做近似打分与 top-k；主注意力仍为 **标准 GQA + 分页 KV**，通过 **Triton sparse GQA** 仅对 top-k 槽位 gather 并计算。上文已概括索引构建主线；逐步公式推导见论文或内部技术附录。

---

## 本仓库中的实现

- **模型与数据流**：`model_executor/models/transgqa.py`（`TransGQAForCausalLM`、Indexer、稀疏 GQA forward 等）。
- **稀疏 GQA kernel**：`v1/attention/ops/sparse_gqa.py`（Triton）；可选 TileLang 路径见同目录 `sparse_gqa_tilelang.py`。

更完整的**文件说明、设计取舍、config 示例与数据流图**见 **[TransGQA_README.md](TransGQA_README.md)**。

---

## 环境、导入与运行

本树作为 `vllm` 包使用。若目录名不是 `vllm`，需通过父目录 + 软链接或 `PYTHONPATH` 挂载，详见 **[TRANSTGQA_TESTING.md](TRANSTGQA_TESTING.md)** 文首说明。

**分阶段测试（Phase 0–6）**（环境检查、sparse GQA、indexer 数学、checkpoint、e2e、benchmark）均写在 **TRANSTGQA_TESTING.md**；一键脚本在 `scripts/transgqa_testing/`。

**服务示例**（需合法权重与 GPU 环境）：

```bash
vllm serve /path/to/transgqa-model
```

`config.json` 中 `architectures` 含 `TransGQAForCausalLM` 时由 vLLM 注册表解析到上述实现；常用字段如 `index_topk`、`freqfold`、`rank_of_wo`、`use_hisa` 等见 TransGQA_README.md 中的示例。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [TransGQA_README.md](TransGQA_README.md) | 实现细节、模块清单、KV cache 与 block_size 约定 |
| [TRANSTGQA_TESTING.md](TRANSTGQA_TESTING.md) | 测试阶段、PYTHONPATH、各 Phase 命令 |

---

## 许可证

沿用上游 vLLM 仓库的许可证与条款；TransGQA 相关改动以本仓库实际文件为准。
