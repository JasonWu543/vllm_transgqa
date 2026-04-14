# TransGQA 索引构建方法

## Setup

设隐藏维度 $d_{\text{model}}$，attention 头数 $H$，key/value 头数 $G$，每头维度：

$$D = \frac{d_{\text{model}}}{H}, \quad \text{Groups} = \frac{H}{G}, \quad D_f = \frac{D}{2}$$

RoPE 将每个 head 特征拆分为实部与虚部：

$$x = [x^{\text{real}},\ x^{\text{imag}}], \quad x^{\text{real}},\ x^{\text{imag}} \in \mathbb{R}^{D_f}$$

---

## Frequency Folding

给定折叠因子 $F$（freqfold），要求 $D_f = MF$。

对 key 的实部进行重排：

$$K^{\text{real}} \in \mathbb{R}^{B \times G \times S \times D_f} \Rightarrow \tilde{K}^{\text{real}} \in \mathbb{R}^{N \times M \times (GF)}$$

其中 $N = BS$。虚部同理得到 $\tilde{K}^{\text{imag}}$。

对每个频率块 $m$：

$$X_m = \begin{bmatrix} \tilde{K}^{\text{real}}[:, m, :] \\ \tilde{K}^{\text{imag}}[:, m, :] \end{bmatrix} \in \mathbb{R}^{2N \times (GF)}$$

---

## Block-wise PCA

对每个 $X_m$ 做低秩分解：

$$X_m \approx U_m \Sigma_m V_m^\top$$

定义投影矩阵：

$$W_m = V_m \in \mathbb{R}^{(GF) \times (GF)}$$

堆叠得到：

$$W_{\text{rope}} \in \mathbb{R}^{M \times (GF) \times (GF)}$$

---

## RoPE Index

在线阶段，对每个 token 的频率块表示 $\tilde{k}_m \in \mathbb{R}^{GF}$：

$$z_m = \tilde{k}_m W_m$$

仅保留前 $F/2$ 个主成分：

$$z^{\text{rope}}_m = z_m[:, :F/2]$$

拼接所有频率块：

$$\text{index}^{\text{rope}}_K = \text{concat}_{m=1}^{M}\ z^{\text{rope}}_m \in \mathbb{R}^{D_f}$$

Query 同理：

$$\text{index}^{\text{rope}}_Q \in \mathbb{R}^{D_f}$$

---

## Non-RoPE Component

对未旋转的 $K^{\text{orig}}$ 进行同样的 folding 与投影：

$$z^{\text{nope}}_m = \tilde{k}^{\text{orig}}_m W_m$$

保留后半主成分：

$$z^{\text{nope,rot}}_m = z_m[:, F/2:]$$

拼接后得 $Z^{\text{nope}} \in \mathbb{R}^{D_f}$。

再做一次 PCA：

$$Z^{\text{nope}} \approx U \Sigma V^\top, \quad W_{\text{nope}} = V^\top$$

最终：

$$\text{index}^{\text{nope}}_K = Z^{\text{nope}} W_{\text{nope}} \in \mathbb{R}^{D_f}$$

---

## Final QK Index

$$\text{index}_K = \begin{bmatrix} \text{index}^{\text{rope}}_K,\ \text{index}^{\text{nope}}_K \end{bmatrix} \in \mathbb{R}^{D}$$

$$\text{index}_Q = \begin{bmatrix} \text{index}^{\text{rope}}_Q,\ \text{index}^{\text{nope}}_Q \end{bmatrix}$$

因此存在分块线性映射：

$$P = \begin{bmatrix} P_{\text{rope}} & 0 \\ 0 & P_{\text{nope}} \end{bmatrix}, \quad \text{index}(x) = Px$$

---

## Value Direction Compression

输出投影：$W_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$。

按 head 重排：$W_o^{(h)} \in \mathbb{R}^{D \times d_{\text{model}}}$。

做 SVD：

$$W_o^{(h)} = U_h \Sigma_h V_h^\top$$

取 rank-$r$：

$$T_h = U_h[:, :r]\, \Sigma_h[:r] \in \mathbb{R}^{D \times r}$$

定义变换：

$$v'_h = v_h T_h$$

得到 group-level 权重：

$$\alpha_{b,s,g} = \|v'_{b,s,g}\|_2$$
