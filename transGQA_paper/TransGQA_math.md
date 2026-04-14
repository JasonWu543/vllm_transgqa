# TransGQA 索引构建方法：完整数学推导

## 符号定义

| 符号 | 含义 | 示例值 |
|------|------|--------|
| $d_{\text{model}}$ | 模型隐藏维度 | 4096 |
| $H$ | Query 头数 | 32 |
| $G$ | Key/Value 头数（GQA groups） | 8 |
| $D$ | 每头维度，$D = d_{\text{model}} / H$ | 128 |
| $D_f$ | 每头 RoPE 实部/虚部维度，$D_f = D/2$ | 64 |
| $F$ | 频率折叠因子（freqfold） | 4 |
| $M$ | 频率块数，$M = D_f / F$ | 16 |
| $B$ | batch size | — |
| $S$ | 序列长度 | — |
| $N$ | 展平后样本数，$N = BS$ | — |
| $r$ | Value 方向压缩的 rank | — |

---

## 总体目标

TransGQA 的目标是为 GQA 模型的每个 token 构建一个**紧凑的 QK 索引**，用于稀疏 attention 中的 top-k 块选择。

核心思路是：注意力分数 $q^\top k$ 可以分解为 RoPE 部分（携带位置信息）和 NoPE 部分（携带语义信息），两者分别压缩后拼接成索引，使得原始注意力分数可以近似为索引的内积：

$$q^\top k \approx \text{index}_Q^\top \cdot \text{index}_K$$

这样 top-k 块选择只需计算低维索引的内积，大幅降低计算量。

---

## Stage 0：原始 GQA 设置

### 基本计算

输入序列中第 $t$ 个 token 的特征 $x_t \in \mathbb{R}^{d_{\text{model}}}$，GQA 计算：

$$q_{t,i} = W^Q_i x_t \in \mathbb{R}^D, \quad i = 1, \ldots, H$$

$$k_{t,j} = W^K_j x_t \in \mathbb{R}^D, \quad j = 1, \ldots, G$$

$$v_{t,j} = W^V_j x_t \in \mathbb{R}^D, \quad j = 1, \ldots, G$$

### RoPE 的实部/虚部拆分

RoPE 将每个头的特征按维度两两配对，解释为复数的实部和虚部：

$$k_{t,j} = [k^{\text{real}}_{t,j},\ k^{\text{imag}}_{t,j}], \quad k^{\text{real}}_{t,j},\ k^{\text{imag}}_{t,j} \in \mathbb{R}^{D_f}$$

加 RoPE 后，第 $l$ 个频率（$l = 1, \ldots, D_f$）对应的 2D 旋转为：

$$\begin{bmatrix} \tilde{k}^{\text{real}}_{t,j,l} \\ \tilde{k}^{\text{imag}}_{t,j,l} \end{bmatrix} = \begin{bmatrix} \cos(t\theta_l) & -\sin(t\theta_l) \\ \sin(t\theta_l) & \cos(t\theta_l) \end{bmatrix} \begin{bmatrix} k^{\text{real}}_{t,j,l} \\ k^{\text{imag}}_{t,j,l} \end{bmatrix}$$

其中 $\theta_l = 10000^{-2(l-1)/D}$。

### 注意力分数的频率展开

对于 query head $i$ 和其对应的 KV group $j = \lceil i / (H/G) \rceil$，注意力分数为：

$$q_{t,i}^\top k_{s,j} = \sum_{l=1}^{D_f} \left[ \cos((t-s)\theta_l)(q^{\text{real}}_{i,l} k^{\text{real}}_{j,l} + q^{\text{imag}}_{i,l} k^{\text{imag}}_{j,l}) + \sin((t-s)\theta_l)(q^{\text{real}}_{i,l} k^{\text{imag}}_{j,l} - q^{\text{imag}}_{i,l} k^{\text{real}}_{j,l}) \right]$$

**关键观察：** 这个求和可以按频率 $l$ 分组。相邻频率 $\theta_l \approx \theta_{l+1}$（当 $D$ 较大时），因此可以把相邻 $F$ 个频率的贡献合并处理，这正是 Frequency Folding 的动机。

---

## Stage 1：Frequency Folding（频率折叠）

### 动机

RoPE 的 $D_f$ 个频率中，相邻频率极为相似：

$$\frac{\theta_{l+1}}{\theta_l} = 10000^{-2/D} \approx 1 \quad (D \text{ 较大时})$$

如果对每个频率 $l$ 分别做 PCA，则：
- 共需 $D_f$ 次独立 PCA，计算代价高
- 每次 PCA 的样本维度只有 $GF$（$G$ 个 KV 头在该频率上的分量），信息量有限

Frequency Folding 将相邻 $F$ 个频率的数据**拼接在一起**做联合 PCA，利用频率相似性，以更少的 PCA 次数（$M = D_f / F$）获得更好的主成分（参见 Proposition F.1）。

### 数据重排

将 $K^{\text{real}} \in \mathbb{R}^{B \times G \times S \times D_f}$ 按频率维度重排：

$$K^{\text{real}} \xrightarrow{\text{reshape}} \tilde{K}^{\text{real}} \in \mathbb{R}^{N \times M \times (GF)}$$

具体地，展平 batch 和序列维度得 $N = BS$，再将 $D_f$ 个频率维度拆分为 $M$ 个频率块，每块包含 $F$ 个相邻频率跨 $G$ 个头的分量：

$$\tilde{K}^{\text{real}}[n, m, :] = \left[ K^{\text{real}}[b, 0, s, mF:(m+1)F],\ \ldots,\ K^{\text{real}}[b, G-1, s, mF:(m+1)F] \right]$$

其中 $n = bS + s$，拼接后维度为 $GF$。虚部同理得到 $\tilde{K}^{\text{imag}} \in \mathbb{R}^{N \times M \times (GF)}$。

### 拼接实部与虚部

对每个频率块 $m = 1, \ldots, M$，将实部和虚部在样本维度上拼接：

$$X_m = \begin{bmatrix} \tilde{K}^{\text{real}}[:, m, :] \\ \tilde{K}^{\text{imag}}[:, m, :] \end{bmatrix} \in \mathbb{R}^{2N \times (GF)}$$

**为什么实部虚部要拼在一起？** 和 TransMLA 中 RoRoPE 的约束一致：RoPE 内积的不变性要求对实部和虚部施加**相同的线性变换**，分开处理会破坏 RoPE 的内积结构。因此必须联合建模。

> **与 TransMLA 的联系：** TransMLA 中 $U_l$ 的约束（实部虚部必须用同一旋转矩阵）和这里对 $X_m$ 联合建模的动机完全一致，都是为了保证 RoPE 内积不变性。

---

## Stage 2：Block-wise PCA（逐块主成分分析）

### 优化目标

对每个频率块 $m$，在 $X_m \in \mathbb{R}^{2N \times GF}$ 上做 SVD：

$$X_m = U_m \Sigma_m V_m^\top$$

其中 $U_m \in \mathbb{R}^{2N \times GF}$，$\Sigma_m = \mathrm{diag}(\sigma_1, \ldots, \sigma_{GF})$（奇异值降序），$V_m \in \mathbb{R}^{GF \times GF}$。

定义投影矩阵 $W_m = V_m \in \mathbb{R}^{GF \times GF}$（正交矩阵），对每个 token 的频率块特征 $\tilde{k}_m \in \mathbb{R}^{GF}$ 做投影：

$$z_m = \tilde{k}_m W_m \in \mathbb{R}^{GF}$$

**最优性：** $W_m = V_m$ 使得 $z_m$ 的前 $r$ 个分量最大化重建方差，即在所有正交变换中，$W_m$ 给出的前 $r$ 个主成分捕获了 $X_m$ 中最多的方差：

$$\max_{W_m^\top W_m = I} \sum_{i=1}^{r} \|X_m W_m[:, i]\|^2 = \sum_{i=1}^{r} \sigma_i^2$$

等号由 SVD 给出，$W_m$ 的第 $i$ 列为第 $i$ 右奇异向量。

### 堆叠得到全局投影矩阵

$$W_{\text{rope}} = [W_1, W_2, \ldots, W_M] \in \mathbb{R}^{M \times (GF) \times (GF)}$$

即对 $M$ 个频率块各有一个独立的 $GF \times GF$ 正交投影矩阵。

> **Proposition F.1（方差保证）：** 与对每个频率单独做 $G$ 维 PCA 相比，Frequency Folding 后在 $GF$ 维空间做联合 PCA，取同等数量主成分时保留的总方差更大。
>
> **证明思路（类比 TransMLA Proposition 2）：** 设 $F$ 个频率的数据矩阵分别为 $X_{m,1}, \ldots, X_{m,F}$，各自最大特征值为 $\lambda_{f,1}$。分别 PCA 取 1 个主成分的总方差为 $V_1 = \sum_f \lambda_{f,1}$。联合 PCA（即对 $[X_{m,1}, \ldots, X_{m,F}]$ 拼接）取 $F$ 个主成分的总方差 $V_2 = \sum_{i=1}^{F} \mu_i$（$\mu_i$ 为拼接协方差的前 $F$ 大特征值）。
>
> 构造特殊 $F$ 维子空间：令第 $f$ 个基向量只在第 $f$ 组块上非零，值为 $w_{f,1}$（第 $f$ 块的第一主成分）。该子空间捕获方差恰为 $V_1$。由于 $V_2$ 是最优 $F$ 维子空间的方差，故 $V_2 \geq V_1$。$\blacksquare$

---

## Stage 3：RoPE Index 构建

### 动机

经过 Block-wise PCA，$z_m \in \mathbb{R}^{GF}$ 的前几个分量集中了频率块 $m$ 的主要信息。

这里将 $GF$ 维输出分成两半：
- **前 $F/2$ 个分量**：方差最大，携带最多**位置相关**信息（RoPE 部分）
- **后 $GF - F/2$ 个分量**：方差较小，携带**位置无关**的语义信息（NoPE 部分）

> 这个分割和 TransMLA 中"第一个虚拟头保留 RoPE，其余头去掉 RoPE"的思路一致。区别在于 TransGQA 在投影后按方差大小分割，而 TransMLA 是对整个头做旋转后按头来分割。

### 构造 RoPE Index

对每个 token，取每个频率块投影的前 $F/2$ 个分量：

$$z^{\text{rope}}_m = z_m[:F/2] \in \mathbb{R}^{F/2}, \quad m = 1, \ldots, M$$

拼接所有频率块：

$$\text{index}^{\text{rope}}_K = \mathrm{concat}_{m=1}^{M}\, z^{\text{rope}}_m \in \mathbb{R}^{M \cdot F/2 = D_f/2}$$

> 注意维度：$M \times F/2 = (D_f/F) \times (F/2) = D_f/2$，恰好是原始每头 RoPE 维度的一半。

Query 侧完全对称：对 $q_{t,i}$ 做相同的 Frequency Folding 和投影，取前 $F/2$ 个分量：

$$\text{index}^{\text{rope}}_Q \in \mathbb{R}^{D_f/2}$$

### RoPE Index 保留的内积近似

**命题：** 对 RoPE 部分，$\text{index}^{\text{rope}}_Q \cdot \text{index}^{\text{rope}}_K \approx$ 原始注意力分数中 RoPE 贡献的部分。

**直觉：** 原始 $q^\top k$ 的 RoPE 部分是 $\sum_{l} S_l$（各频率的贡献之和）。Frequency Folding 把相邻 $F$ 个频率折叠后做 PCA，PCA 的前 $F/2$ 个主成分捕获了这 $F$ 个频率联合信息的主要方差，因此前 $F/2$ 个投影分量的内积近似了原来 $F$ 个频率的联合贡献。

---

## Stage 4：Non-RoPE（NoPE）Index 构建

### 两步压缩

NoPE 部分需要两步压缩：

**第一步：** 从 Block-wise PCA 的后半输出中取 NoPE 分量。

对原始（未加 RoPE 的）key $K^{\text{orig}}$ 做同样的 Frequency Folding，得到 $\tilde{k}^{\text{orig}}_m \in \mathbb{R}^{GF}$，然后投影：

$$z_m = \tilde{k}^{\text{orig}}_m W_m \in \mathbb{R}^{GF}$$

取后半输出（位置信息少的分量）：

$$z^{\text{nope}}_m = z_m[F/2:] \in \mathbb{R}^{GF - F/2}, \quad m = 1, \ldots, M$$

拼接所有频率块：

$$Z^{\text{nope}} = \mathrm{concat}_{m=1}^{M}\, z^{\text{nope}}_m \in \mathbb{R}^{M(GF - F/2)}$$

**为什么用未加 RoPE 的原始 key？** NoPE 分量的目的是捕获语义信息（内容相关性），与位置无关。使用原始 key 避免了 RoPE 旋转对语义方向的干扰，让第二步 PCA 能更干净地找到语义主方向。

**第二步：** 对 $Z^{\text{nope}}$ 再做一次 PCA，进一步压缩到 $D_f/2$ 维：

$$Z^{\text{nope}} \approx U \Sigma V^\top, \quad W_{\text{nope}} = V^\top \in \mathbb{R}^{D_f/2 \times M(GF-F/2)}$$

> 注意：第二步 PCA 是在**所有 token 的** $Z^{\text{nope}}$ 上做的，找到跨频率块的全局语义主方向。

最终：

$$\text{index}^{\text{nope}}_K = Z^{\text{nope}}\, W_{\text{nope}}^\top \in \mathbb{R}^{D_f/2}$$

Query 侧对称构造 $\text{index}^{\text{nope}}_Q \in \mathbb{R}^{D_f/2}$。

### 为什么需要两步压缩

| | 第一步（Block-wise PCA 后半） | 第二步（全局 PCA） |
|--|------|------|
| 输入维度 | $M(GF - F/2)$（较大） | $D_f/2$（目标维度） |
| 目的 | 分离 RoPE 和 NoPE 信息 | 跨频率块找全局语义主方向 |
| 作用范围 | 每个频率块独立 | 所有频率块联合 |

直接对原始 key 做一次 PCA 会混淆 RoPE 方向和语义方向；两步分离确保索引的 NoPE 部分真正是位置无关的语义信息。

---

## Stage 5：最终 QK 索引

### 拼接

将 RoPE 和 NoPE 两部分拼接：

$$\text{index}_K = \begin{bmatrix} \text{index}^{\text{rope}}_K \\ \text{index}^{\text{nope}}_K \end{bmatrix} \in \mathbb{R}^{D_f/2 + D_f/2 = D_f}$$

$$\text{index}_Q = \begin{bmatrix} \text{index}^{\text{rope}}_Q \\ \text{index}^{\text{nope}}_Q \end{bmatrix} \in \mathbb{R}^{D_f}$$

### 线性映射的块对角结构

整个索引构建过程可以写成一个分块线性映射 $P$：

$$\text{index}(k) = P\, k, \quad P = \begin{bmatrix} P_{\text{rope}} & 0 \\ 0 & P_{\text{nope}} \end{bmatrix} \in \mathbb{R}^{D_f \times D}$$

其中：
- $P_{\text{rope}} \in \mathbb{R}^{D_f/2 \times D_f}$：对 RoPE 后的 key 实部虚部做 Frequency Folding + Block PCA 前半
- $P_{\text{nope}} \in \mathbb{R}^{D_f/2 \times D_f}$：对原始 key 做 Frequency Folding + Block PCA 后半 + 全局 PCA

**块对角性的意义：** RoPE 部分和 NoPE 部分互不干扰，各自独立压缩，与各自的 query 索引做内积。这保证了两种信息的压缩互相正交，不会因为混合而相互污染。

### 近似质量

总体近似误差来源于两处截断：
1. Block-wise PCA 取 $F/2$ 个主成分（丢弃了 $GF - F/2$ 个较小奇异值方向）
2. 全局 NoPE PCA 截断到 $D_f/2$ 维

由于 PCA 在 Frobenius 范数意义下是最优低秩近似（Eckart-Young 定理），两步截断分别在各自子空间内是最优的：

$$\min_{\text{rank-}r\ \hat{X}} \|X - \hat{X}\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}$$

---

## Stage 6：Value 方向压缩

### 动机

在稀疏 attention 中，选出 top-k 块后需要计算 attention 输出：

$$o_{t,i} = \sum_{j \in \text{top-k}} \text{softmax}(\cdot)\, v_j$$

Value 向量 $v_j \in \mathbb{R}^D$ 维度较高。如果能找到 $v_j$ 的低维表示，可以进一步加速块重要性的估计（不是存储压缩，而是用低维 value 近似计算输出范数来评估块的重要性）。

### SVD 压缩

对输出投影矩阵 $W_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，按 head 拆分为：

$$W_o^{(h)} \in \mathbb{R}^{D \times d_{\text{model}}}, \quad h = 1, \ldots, H$$

对每个 head 做 SVD：

$$W_o^{(h)} = U_h \Sigma_h V_h^\top$$

取 rank-$r$ 截断：

$$T_h = U_h[:, :r]\, \Sigma_h[:r,:r] \in \mathbb{R}^{D \times r}$$

**最优性（Eckart-Young 定理）：** $T_h$ 是 $W_o^{(h)}$ 的最优 rank-$r$ 近似（在 Frobenius 范数意义下），即在所有 rank $\leq r$ 的矩阵中，$\|W_o^{(h)} - T_h V_h^\top\|_F$ 最小。

### 压缩后的 Value 表示

定义压缩后的 value：

$$v'_h = v_h T_h \in \mathbb{R}^r$$

这等价于将输出 $W_o^{(h)} v_h \approx T_h (T_h^\top v_h) = T_h v'_h$，用 $r$ 维向量 $v'_h$ 近似表示 head $h$ 对输出的贡献方向。

### 块重要性评分

对于 group $g$ 内的一个 KV 块，定义其重要性为压缩后 value 的 L2 范数：

$$\alpha_{b,s,g} = \|v'_{b,s,g}\|_2 = \|v_{b,s,g}\, T_g\|_2$$

**直觉：** $\|v'_h\|_2$ 越大，说明该 token 的 value 在输出投影的主方向上投影越大，对最终输出的贡献越显著。因此 $\alpha$ 可以作为评估一个 KV 块是否值得计算的重要性分数，配合 QK 索引共同决定 top-k 块选择。

> **与 QK 索引的分工：**
> - **QK 索引**：估计 query 和 key 的注意力分数（相关性），决定哪些位置的 key 和 query 相似
> - **Value 评分 $\alpha$**：估计 value 本身的"影响力"，决定哪些位置的 value 对输出贡献大
> 两者结合，才能在稀疏 attention 中准确定位最重要的 KV 块

---

## 整体流程总结

```
原始 GQA Key/Query
  每头维度 D，实部+虚部各 D_f = D/2 维
        │
        │ Stage 1：Frequency Folding（折叠因子 F）
        │   将 D_f 个频率按 F 个一组折叠
        │   实部虚部拼接：X_m ∈ R^{2N × GF}，共 M = D_f/F 个块
        ▼
频率块表示 {X_m}，m = 1,...,M
        │
        │ Stage 2：Block-wise PCA
        │   对每个 X_m 做 SVD，得投影矩阵 W_m ∈ R^{GF × GF}
        │   投影：z_m = k̃_m W_m ∈ R^{GF}
        ▼
投影后分量 {z_m}，前 F/2 个分量方差大（RoPE），后 GF-F/2 个分量方差小（NoPE）
        │
        ├──────────────────────────────────────┐
        │ Stage 3：RoPE Index                  │ Stage 4：NoPE Index
        │   取 z_m 前 F/2 维                   │   取 z_m（原始key）后 GF-F/2 维
        │   拼接：index^rope ∈ R^{D_f/2}       │   拼接后再做全局 PCA
        │                                      │   index^nope ∈ R^{D_f/2}
        └──────────────────┬───────────────────┘
                           │ Stage 5：拼接
                           ▼
            index_K = [index^rope ; index^nope] ∈ R^{D_f = D/2}
            index_Q 对称构造

            稀疏 Attention Top-k 选择：
            score = index_Q^T · index_K  （低维内积近似原始注意力分数）
        │
        │ Stage 6：Value 方向压缩
        │   W_o^(h) 做 SVD，取 rank-r 截断得 T_h
        │   v'_h = v_h T_h ∈ R^r
        │   重要性：α = ||v'||_2
        ▼
最终稀疏 Attention：QK 索引选 top-k 块 + Value 评分辅助筛选
```

---

## 三个关键设计的核心作用

| 设计 | 解决的问题 | 手段 |
|------|-----------|------|
| **Frequency Folding** | 相邻频率独立 PCA 信息利用不充分 | 将相邻 $F$ 个频率联合建模，联合 PCA 方差保留更多 |
| **RoPE/NoPE 分离** | 位置信息和语义信息混淆，索引质量差 | Block PCA 后按方差大小分割，前半做 RoPE index，后半做 NoPE index |
| **Value 方向压缩** | 单靠 QK 相似度不能准确衡量块重要性 | SVD 提取 $W_o$ 主方向，用 $\|v T_h\|_2$ 作为 value 影响力评分 |
