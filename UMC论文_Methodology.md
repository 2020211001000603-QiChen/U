# UMC论文 - Methodology（方法论）

## 3. Methodology

### 3.1 问题定义

给定一个包含 $N$ 个多模态话语样本的无标签数据集 $\mathcal{D} = \{(x_i^t, x_i^a, x_i^v)\}_{i=1}^N$，其中 $x_i^t \in \mathbb{R}^{L_t \times d_t}$、$x_i^a \in \mathbb{R}^{L_a \times d_a}$ 和 $x_i^v \in \mathbb{R}^{L_v \times d_v}$ 分别表示第 $i$ 个样本的文本、音频和视频特征序列，$L_t$、$L_a$、$L_v$ 为序列长度，$d_t$、$d_a$、$d_v$ 为特征维度。

无监督多模态聚类的目标是学习一个映射函数 $f: \mathcal{D} \rightarrow \mathcal{C}$，将样本映射到 $K$ 个潜在的语义簇 $\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$，其中 $K$ 通常通过数据集先验知识或启发式方法确定。我们的目标是学习一个多模态特征编码器 $f_\theta$，使得同一簇内的样本在特征空间中距离较近，不同簇间的样本距离较远。

### 3.2 UMC框架概述

UMC框架的整体架构如图1所示（引用框架图），包含四个主要阶段：

**阶段一：特征提取与投影。** 使用预训练的BERT、WavLM和Swin Transformer分别提取文本、音频和视频的初始特征，然后通过线性投影层将不同模态的特征投影到统一的维度空间 $d = 256$。

**阶段二：ConFEDE双投影增强。** 对视频和音频特征分别应用双投影机制，通过相似性投影和相异性投影分别提取主要信息和环境信息，然后通过融合层和残差连接得到增强特征。

**阶段三：文本引导多模态融合。** 以文本特征为锚点，通过交叉注意力机制引导视频和音频特征的注意力计算，然后通过文本引导注意力和自注意力层进一步细化特征交互，最后通过门控融合机制生成最终的多模态表示。

**阶段四：自适应渐进式聚类学习。** 使用自适应阈值动态选择高质量样本进行监督学习，低质量样本进行无监督对比学习，通过联合优化聚类损失和对比损失，学习具有判别性的特征表示。

### 3.3 创新点一：ConFEDE双投影机制

#### 3.3.1 核心思想

传统的多模态融合方法通常将视频和音频特征作为单一向量处理，忽略了特征内部的层次性。具体而言，视频特征既包含与语义相关的主要信息（如人物动作、表情），也包含与语义无关的环境信息（如背景、场景）；音频特征既包含主要信息（如语音内容、语调、情感），也包含环境信息（如背景噪音、环境音、音质）。

ConFEDE（Contextual Feature Extraction via Dual Projection）机制通过双投影网络分别提取这两种信息，然后通过融合层和残差连接得到增强特征，显著提升了特征表示的丰富性和判别能力。

#### 3.3.2 技术实现

对于视频模态，我们定义相似性投影 $P_{\text{simi}}^v$ 和相异性投影 $P_{\text{dissi}}^v$：

$$\mathbf{z}_{\text{simi}}^v = P_{\text{simi}}^v(\mathbf{x}^v) = \text{GELU}(\text{LN}(\mathbf{x}^v) \mathbf{W}_{\text{simi}}^v + \mathbf{b}_{\text{simi}}^v)$$

$$\mathbf{z}_{\text{dissi}}^v = P_{\text{dissi}}^v(\mathbf{x}^v) = \text{GELU}(\text{LN}(\mathbf{x}^v) \mathbf{W}_{\text{dissi}}^v + \mathbf{b}_{\text{dissi}}^v)$$

其中 $\mathbf{x}^v \in \mathbb{R}^{L \times d}$ 为输入视频特征，$\text{LN}$ 为LayerNorm，$\mathbf{W}_{\text{simi}}^v, \mathbf{W}_{\text{dissi}}^v \in \mathbb{R}^{d \times d}$ 为可学习参数，$\text{GELU}$ 为激活函数。

然后，我们将双投影结果拼接并通过融合层得到增强特征：

$$\mathbf{z}_{\text{dual}}^v = [\mathbf{z}_{\text{simi}}^v; \mathbf{z}_{\text{dissi}}^v] \in \mathbb{R}^{L \times 2d}$$

$$\hat{\mathbf{x}}^v = \text{Fusion}(\mathbf{z}_{\text{dual}}^v) + \mathbf{x}^v = \mathbf{z}_{\text{dual}}^v \mathbf{W}_{\text{fusion}}^v + \mathbf{x}^v$$

其中 $\mathbf{W}_{\text{fusion}}^v \in \mathbb{R}^{2d \times d}$ 为融合层参数，残差连接 $\mathbf{x}^v$ 确保原始信息不丢失。

对于音频模态，我们采用相同的双投影机制：

$$\hat{\mathbf{x}}^a = \text{Fusion}([P_{\text{simi}}^a(\mathbf{x}^a); P_{\text{dissi}}^a(\mathbf{x}^a)]) + \mathbf{x}^a$$

#### 3.3.3 设计动机

通过分离主要信息和环境信息，ConFEDE机制能够：
1. **提升特征判别性**：主要信息更直接地与语义相关，有助于区分不同类别
2. **增强鲁棒性**：环境信息可能包含噪声，通过分离可以降低其对聚类的影响
3. **丰富特征表示**：双投影提供了更丰富的特征表示空间，有助于学习更复杂的语义结构

### 3.4 创新点二：文本引导多模态注意力融合

#### 3.4.1 核心思想

在多模态对话场景中，文本信息通常比其他模态更丰富和可靠，因为文本直接表达了语义内容。因此，我们提出以文本为锚点，引导视频和音频特征的注意力计算，确保多模态融合过程中的语义一致性。

文本引导多模态注意力融合包含三个阶段：
1. **交叉注意力阶段**：以文本特征为查询，引导视频和音频特征的注意力计算
2. **文本引导注意力阶段**：进一步细化文本与视频/音频的交互
3. **自注意力阶段**：对拼接后的多模态特征进行全局交互

#### 3.4.2 交叉注意力机制与多头注意力

UMC在所有注意力计算中使用**多头注意力（Multi-Head Attention）**机制，这是Transformer架构的核心组件。多头注意力通过并行计算多个注意力头，能够从不同角度捕获特征间的复杂关系。

**多头注意力的数学定义**：给定查询 $\mathbf{Q} \in \mathbb{R}^{L \times B \times d}$、键 $\mathbf{K} \in \mathbb{R}^{L \times B \times d}$ 和值 $\mathbf{V} \in \mathbb{R}^{L \times B \times d}$，多头注意力计算如下：

1. **线性投影到多个头**：
   $$\mathbf{Q}_h = \mathbf{Q} \mathbf{W}_Q^h, \quad \mathbf{K}_h = \mathbf{K} \mathbf{W}_K^h, \quad \mathbf{V}_h = \mathbf{V} \mathbf{W}_V^h$$
   其中 $h = 1, \ldots, H$，$H = 8$ 为注意力头数，$\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h \in \mathbb{R}^{d \times d_h}$，$d_h = d/H = 32$ 为每个头的维度。

2. **缩放点积注意力**：
   $$\text{Attn}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_h}}\right) \mathbf{V}_h$$

3. **多头拼接和输出投影**：
   $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{Attn}_1, \ldots, \text{Attn}_H) \mathbf{W}_O$$
   其中 $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ 为输出投影矩阵。

**交叉注意力应用**：首先，我们将文本、视频和音频特征转换为序列格式：$\mathbf{T} \in \mathbb{R}^{L_t \times B \times d}$，$\mathbf{V} \in \mathbb{R}^{L_v \times B \times d}$，$\mathbf{A} \in \mathbb{R}^{L_a \times B \times d}$，其中 $B$ 为批次大小。

对于视频模态，我们使用文本特征作为查询，视频特征作为键值，通过8头注意力计算：

$$\text{Attn}_v = \text{MultiHead}(\mathbf{T}, \mathbf{V}, \mathbf{V})$$

对于音频模态，我们采用相同的8头交叉注意力机制：

$$\text{Attn}_a = \text{MultiHead}(\mathbf{T}, \mathbf{A}, \mathbf{A})$$

**设计优势**：多头注意力能够从8个不同子空间捕获特征关系，相比单头注意力具有更强的表达能力，同时可以并行计算，提升效率。

#### 3.4.3 文本引导注意力机制

在交叉注意力之后，我们进一步使用文本特征引导注意力计算，同样采用8头多头注意力：

$$\mathbf{H}_v = \text{MultiHead}(\mathbf{T}, \text{Attn}_v, \text{Attn}_v)$$

$$\mathbf{H}_a = \text{MultiHead}(\mathbf{T}, \text{Attn}_a, \text{Attn}_a)$$

其中 $\text{MultiHead}$ 为8头多头注意力机制，以文本特征为查询，交叉注意力结果为键值。8个注意力头能够从不同角度细化文本与视频/音频的交互关系。

#### 3.4.4 自注意力层

为了进一步建模多模态特征间的全局交互，我们将文本、视频和音频特征拼接后进行自注意力计算，使用2层8头多头注意力：

$$\mathbf{H}_{\text{combined}} = [\mathbf{T}; \mathbf{H}_v; \mathbf{H}_a] \in \mathbb{R}^{(L_t + L_v + L_a) \times B \times d}$$

$$\mathbf{H}_{\text{attended}}^{(1)} = \text{MultiHead}(\mathbf{H}_{\text{combined}}, \mathbf{H}_{\text{combined}}, \mathbf{H}_{\text{combined}}) + \mathbf{H}_{\text{combined}}$$

$$\mathbf{H}_{\text{attended}}^{(2)} = \text{MultiHead}(\mathbf{H}_{\text{attended}}^{(1)}, \mathbf{H}_{\text{attended}}^{(1)}, \mathbf{H}_{\text{attended}}^{(1)}) + \mathbf{H}_{\text{attended}}^{(1)}$$

其中每层都使用8头多头注意力，2层堆叠能够建模更复杂的多模态特征交互，残差连接确保原始信息不丢失。

然后，我们将 $\mathbf{H}_{\text{attended}}$ 分离回三个模态：

$$\mathbf{T}_{\text{enhanced}} = \mathbf{H}_{\text{attended}}[:L_t]$$
$$\mathbf{V}_{\text{enhanced}} = \mathbf{H}_{\text{attended}}[L_t:L_t+L_v]$$
$$\mathbf{A}_{\text{enhanced}} = \mathbf{H}_{\text{attended}}[L_t+L_v:]$$

#### 3.4.5 门控融合机制

最后，我们使用门控机制自适应地加权不同模态的贡献。首先，我们对每个模态进行池化操作：

$$\mathbf{t} = \text{Pool}(\mathbf{T}_{\text{enhanced}}) \in \mathbb{R}^{B \times d}$$
$$\mathbf{v} = \text{Pool}(\mathbf{V}_{\text{enhanced}}) \in \mathbb{R}^{B \times d}$$
$$\mathbf{a} = \text{Pool}(\mathbf{A}_{\text{enhanced}}) \in \mathbb{R}^{B \times d}$$

然后，我们计算门控权重：

$$\mathbf{g}_t = \text{sigmoid}(\mathbf{t} \mathbf{W}_g^t + \mathbf{b}_g^t)$$
$$\mathbf{g}_v = \text{sigmoid}(\mathbf{v} \mathbf{W}_g^v + \mathbf{b}_g^v)$$
$$\mathbf{g}_a = \text{sigmoid}(\mathbf{a} \mathbf{W}_g^a + \mathbf{b}_g^a)$$

$$\mathbf{g} = \text{softmax}([\mathbf{g}_t; \mathbf{g}_v; \mathbf{g}_a]) \in \mathbb{R}^{B \times 3}$$

最终的多模态表示为：

$$\mathbf{h} = \mathbf{g}_t \odot \mathbf{t} + \mathbf{g}_v \odot \mathbf{v} + \mathbf{g}_a \odot \mathbf{a}$$

其中 $\odot$ 为逐元素乘法。

### 3.5 创新点三：自适应渐进式学习策略

#### 3.5.1 核心思想

传统的无监督聚类方法通常使用固定的训练策略，如固定的样本选择阈值或简单的线性增长策略。然而，聚类学习是一个动态过程，不同训练阶段需要不同的策略。因此，我们提出自适应渐进式学习策略，通过多维度动态调整训练参数，实现更高效的聚类学习。

自适应渐进式学习策略包含四个核心组件：
1. **S型曲线阈值增长**：根据训练进度动态调整样本选择阈值
2. **性能自适应调整**：根据近期性能趋势动态调整阈值
3. **损失自适应调整**：根据损失变化情况调整策略
4. **稳定性调整**：监控训练稳定性并做出相应调整

#### 3.5.2 S型曲线阈值增长

我们设计了一个S型曲线函数，根据训练进度 $p = \frac{\text{epoch}}{\text{total\_epochs}}$ 动态计算基础阈值：

$$\tau_{\text{base}}(p) = \begin{cases}
\tau_{\text{min}} + \Delta \tau \cdot 0.2 \cdot p^2 & \text{if } p < 0.2 \\
\tau_{\text{min}} + \Delta \tau \cdot (0.2 + 0.5 \cdot \frac{p-0.2}{0.4}) & \text{if } 0.2 \leq p < 0.6 \\
\tau_{\text{min}} + \Delta \tau \cdot (0.7 + 0.25 \cdot \frac{\log(1+3(p-0.6))}{\log(4)}) & \text{if } 0.6 \leq p < 0.9 \\
\tau_{\text{min}} + \Delta \tau \cdot (0.95 + 0.05 \cdot \frac{p-0.9}{0.1}) & \text{if } p \geq 0.9
\end{cases}$$

其中 $\tau_{\text{min}} = 0.05$ 为最小阈值，$\Delta \tau = 0.45$ 为阈值范围，$p$ 为训练进度。

S型曲线设计符合聚类学习的自然规律：
- **早期阶段（0-20%）**：缓慢增长，避免选择低质量样本
- **中期阶段（20-60%）**：快速增长，充分利用已学习的特征
- **后期阶段（60-90%）**：稳定增长，逐步优化聚类质量
- **最终阶段（90-100%）**：微调增长，精细优化

#### 3.5.3 性能自适应调整

我们监控最近 $W = 3$ 个epoch的性能趋势，根据性能变化动态调整阈值：

$$\Delta \tau_{\text{perf}} = \begin{cases}
+0.03 & \text{if } \text{trend} = \text{improving} \text{ and } \text{rate} > 0.02 \\
+0.02 & \text{if } \text{trend} = \text{improving} \text{ and } 0.01 < \text{rate} \leq 0.02 \\
+0.01 & \text{if } \text{trend} = \text{improving} \text{ and } \text{rate} \leq 0.01 \\
-0.02 & \text{if } \text{trend} = \text{declining} \text{ and } \text{rate} > 0.02 \\
-0.01 & \text{if } \text{trend} = \text{declining} \text{ and } \text{rate} \leq 0.02 \\
0 & \text{otherwise}
\end{cases}$$

其中 $\text{rate} = \frac{|\text{perf}_t - \text{perf}_{t-W}|}{W}$ 为性能变化率。

#### 3.5.4 损失自适应调整

我们监控最近3个epoch的损失变化，根据损失趋势调整阈值：

$$\Delta \tau_{\text{loss}} = \begin{cases}
+0.015 & \text{if } \Delta \mathcal{L} < -0.05 \\
+0.01 & \text{if } -0.05 \leq \Delta \mathcal{L} < -0.01 \\
-0.02 & \text{if } \Delta \mathcal{L} > 0.05 \\
-0.01 & \text{if } 0.01 < \Delta \mathcal{L} \leq 0.05 \\
0 & \text{otherwise}
\end{cases}$$

其中 $\Delta \mathcal{L} = \frac{\mathcal{L}_t - \mathcal{L}_{t-3}}{3}$ 为损失变化率。

#### 3.5.5 稳定性调整

我们计算最近5个epoch的性能变异系数（Coefficient of Variation），根据稳定性调整阈值：

$$\text{CV} = \frac{\sigma(\{\text{perf}_{t-4}, \ldots, \text{perf}_t\})}{\mu(\{\text{perf}_{t-4}, \ldots, \text{perf}_t\})}$$

$$\Delta \tau_{\text{stab}} = \begin{cases}
-0.01 & \text{if } \text{CV} > 0.1 \\
+0.005 & \text{if } \text{CV} < 0.02 \\
0 & \text{otherwise}
\end{cases}$$

#### 3.5.6 综合阈值计算

最终的自适应阈值为：

$$\tau_{\text{adaptive}} = \text{clip}(\tau_{\text{base}} + \Delta \tau_{\text{perf}} + \Delta \tau_{\text{loss}} + \Delta \tau_{\text{stab}}, \tau_{\text{min}}, \tau_{\text{max}})$$

其中 $\tau_{\text{max}} = 0.5$ 为最大阈值，$\text{clip}$ 为截断函数。

#### 3.5.7 K-means++聚类初始化与Warm Start策略

UMC使用**K-means++算法**初始化聚类中心，并创新性地引入了**Warm Start策略**，这是K-means算法的改进版本，能够提供更好的初始聚类中心，从而提升聚类质量和训练稳定性。

**K-means++算法原理**：K-means++通过概率分布选择初始聚类中心，而不是随机选择。具体步骤如下：

1. **第一个中心**：随机选择一个样本作为第一个聚类中心 $\mathbf{c}_1$。

2. **后续中心**：对于 $k = 2, \ldots, K$，按照以下概率分布选择第 $k$ 个聚类中心：
   $$P(\mathbf{x}_i \text{ 被选为 } \mathbf{c}_k) = \frac{D(\mathbf{x}_i)^2}{\sum_{j=1}^N D(\mathbf{x}_j)^2}$$
   其中 $D(\mathbf{x}_i) = \min_{l=1}^{k-1} \|\mathbf{x}_i - \mathbf{c}_l\|_2$ 为样本 $\mathbf{x}_i$ 到最近已选中心的距离。

3. **K-means迭代**：使用选定的初始中心，运行标准K-means算法直到收敛。

**UMC的Warm Start创新**：
- **第一个epoch**：使用K-means++初始化聚类中心 $\mathbf{C}^{(0)} = \text{K-means++}(\mathcal{D})$
- **后续epoch**：使用上一轮训练得到的聚类中心作为初始值（Warm Start）：
  $$\mathbf{C}^{(t)} = \text{K-means}(\mathcal{D}, \text{init} = \mathbf{C}^{(t-1)}), \quad t \geq 1$$

**设计优势**：
1. **更好的初始中心**：K-means++选择的初始中心分布更均匀，避免陷入局部最优
2. **计算效率**：Warm Start避免了重复的K-means++初始化，显著减少计算时间
3. **训练稳定性**：使用上一轮的聚类中心作为初始值，训练过程更稳定，减少训练波动
4. **收敛速度**：相比随机初始化，K-means++能够更快收敛到好的聚类结果

#### 3.5.8 基于密度的自适应样本选择

UMC创新性地提出了**基于局部密度的自适应样本选择机制**，相比传统的基于距离的简单选择，能够更准确地识别高质量样本。

##### 3.5.8.1 局部密度计算

对于每个簇 $C_k$ 内的样本，UMC计算其局部密度来评估样本质量。对于样本 $\mathbf{x}_i \in C_k$，其局部密度定义为：

$$\rho_k(\mathbf{x}_i) = \frac{1}{D_k(\mathbf{x}_i)} = \frac{k}{\sum_{j=1}^k \|\mathbf{x}_i - \mathbf{x}_{i_j}\|_2}$$

其中：
- $D_k(\mathbf{x}_i) = \frac{1}{k} \sum_{j=1}^k \|\mathbf{x}_i - \mathbf{x}_{i_j}\|_2$ 为样本 $\mathbf{x}_i$ 到其 $k$ 个最近邻的平均距离（可达距离）
- $\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_k}$ 为样本 $\mathbf{x}_i$ 在簇 $C_k$ 内的 $k$ 个最近邻

**设计动机**：密度高的样本更接近聚类中心，更可能属于该簇，因此更可靠。相比简单的距离度量，局部密度能够更好地反映样本在簇内的相对位置。

##### 3.5.8.2 自适应k值选择

UMC不是使用固定的k值计算密度，而是通过**网格搜索**选择最优的k值，这是UMC的另一个创新点。

**候选k值范围**：
$$K_{\text{candidates}} = \{k_{\text{cand}} \cdot |C_k| | k_{\text{cand}} \in [0.1, 0.32], \text{step} = 0.02\}$$

**评估指标**：对于每个候选k值 $k_{\text{cand}}$，计算选中样本的**类内紧密度**作为评估分数：

$$\text{Score}(k_{\text{cand}}) = \frac{1}{|\mathcal{S}_k^{(k_{\text{cand}})}|} \sum_{\mathbf{x}_i \in \mathcal{S}_k^{(k_{\text{cand}})}} \min_{j \neq i} \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

其中 $\mathcal{S}_k^{(k_{\text{cand}})}$ 为使用 $k_{\text{cand}}$ 计算密度后选择的前 $N_{\text{cutoff}}$ 个样本。

**选择策略**：
$$k^* = \arg\max_{k_{\text{cand}} \in K_{\text{candidates}}} \text{Score}(k_{\text{cand}})$$

选择评估分数最高的k值（类内紧密度越大，样本质量越高）。

##### 3.5.8.3 类别不平衡处理

为了处理真实数据集中的类别不平衡问题，UMC设计了**最小样本数保证机制**和**动态调整策略**。

**最小样本数计算**：
$$N_{\text{min}} = \max(5, \lfloor 0.1 \cdot \frac{|\mathcal{D}|}{K} \rfloor)$$

其中 $|\mathcal{D}|$ 为总样本数，$K$ 为类别数。该公式采用**双重保证机制**：
- **绝对最小值**：至少5个样本（防止极端情况）
- **相对最小值**：总数的10%除以类别数（适应数据集规模）
- **取两者中的较大值**，确保小数据集和大数据集都有合适的保证

**实际例子**：
- MIntRec数据集（$|\mathcal{D}|=10,000$, $K=20$）：$N_{\text{min}} = \max(5, 50) = 50$
- IEMOCAP-DA数据集（$|\mathcal{D}|=6,000$, $K=9$）：$N_{\text{min}} = \max(5, 66) = 66$
- 极端小数据集（$|\mathcal{D}|=500$, $K=10$）：$N_{\text{min}} = \max(5, 5) = 5$

**每个簇的选择数量（动态调整）**：
$$\text{cutoff}_k = \begin{cases}
|C_k| & \text{if } |C_k| \leq 5 \\
\max(\lfloor |C_k| \cdot \tau_{\text{adaptive}} \rfloor, \min(N_{\text{min}}, |C_k|)) & \text{otherwise}
\end{cases}$$

**动态调整的两个层面**：

1. **自适应阈值 $\tau_{\text{adaptive}}$ 的动态调整**（每个epoch）：
   - 早期epoch（0-20%）：$\tau \approx 0.05-0.15$（保守，选择少）
   - 中期epoch（20-60%）：$\tau \approx 0.15-0.35$（快速增长）
   - 后期epoch（60-90%）：$\tau \approx 0.35-0.45$（稳定增长）
   - 最终epoch（90-100%）：$\tau \approx 0.45-0.5$（微调）
   - 根据性能、损失、稳定性反馈进一步调整

2. **选择数量 $\text{cutoff}_k$ 的动态调整**（跟随阈值变化）：
   - 阈值小时：$\text{cutoff}_k = \max(\text{阈值计算值}, N_{\text{min}})$（至少保证最小样本数）
   - 阈值大时：$\text{cutoff}_k = \max(\text{阈值计算值}, N_{\text{min}})$（可能选择更多）

**设计优势**：
1. **保证覆盖**：即使阈值很小，也保证每个类别至少有 $N_{\text{min}}$ 个样本用于训练
2. **处理极端不平衡**：对于极少数类（少于5个样本），选择所有样本避免信息丢失
3. **动态适应**：根据训练进度和性能反馈动态调整选择数量，既保证数量又保证质量
4. **双重保护**：最小样本数保证 + 极少数类完全保护，确保所有类别都有足够样本

##### 3.5.8.4 最终样本选择

对于簇 $C_k$，最终的样本选择过程为：

1. **使用最优k值计算密度**：$\rho_{k^*}(\mathbf{x}_i) = 1 / D_{k^*}(\mathbf{x}_i)$
2. **按密度排序**：$\text{sort}(\{\rho_{k^*}(\mathbf{x}_i)\}_{i \in C_k})$
3. **选择前 $N_{\text{cutoff}}$ 个样本**：$\mathcal{S}_k = \text{TopN}(\text{sort}(\{\rho_{k^*}(\mathbf{x}_i)\}_{i \in C_k}), \text{cutoff}_k)$

然后，根据自适应阈值和密度选择结果，将样本分为两类：

$$\mathcal{S}_{\text{high}} = \bigcup_{k=1}^K \mathcal{S}_k$$

$$\mathcal{S}_{\text{low}} = \{i | i \notin \mathcal{S}_{\text{high}}\}$$

**与自适应阈值的协同**：
- **自适应阈值**控制"选多少"：$\text{cutoff}_k = \max(|C_k| \cdot \tau_{\text{adaptive}}, N_{\text{min}})$
- **密度选择**控制"选哪些"：在给定阈值下，选择密度最高的样本
- **两者协同**：实现更智能的样本选择，既保证数量又保证质量

#### 3.5.8 智能早停机制

我们设计了多维度早停机制，避免无效训练：

1. **性能早停**：如果最近 $P = 5$ 个epoch的性能都低于最佳性能的99.5%，则早停
2. **损失早停**：如果最近 $P$ 个epoch的损失持续上升，则早停
3. **退化早停**：如果总退化次数超过3次且训练轮数超过10，则早停

### 3.6 损失函数设计（创新点四：联合损失优化策略）

UMC采用**多损失联合优化策略**，将监督学习、对比学习和聚类优化有机结合，这是UMC的第四个创新点。与现有方法通常只使用单一损失函数不同，UMC设计了分样本类型的损失函数，针对高质量样本和低质量样本采用不同的优化策略。

#### 3.6.1 监督对比学习损失（SupConLoss）

对于高质量样本 $\mathcal{S}_{\text{high}}$，我们使用**监督对比学习损失（Supervised Contrastive Learning Loss）**，这是对比学习在监督场景下的扩展。给定样本特征 $\mathbf{h}_i$ 和伪标签 $y_i$，SupConLoss定义为：

$$\mathcal{L}_{\text{supcon}} = -\frac{1}{|\mathcal{S}_{\text{high}}|} \sum_{i \in \mathcal{S}_{\text{high}}} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_p) / \tau_{\text{sup}})}{\sum_{a \in A(i)} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_a) / \tau_{\text{sup}})}$$

其中：
- $P(i) = \{p \in \mathcal{S}_{\text{high}} | y_p = y_i, p \neq i\}$ 为样本 $i$ 的正样本集合（同类别样本）
- $A(i) = \{a \in \mathcal{S}_{\text{high}} | a \neq i\}$ 为样本 $i$ 的所有其他样本集合
- $\text{sim}(\mathbf{h}_i, \mathbf{h}_j) = \frac{\mathbf{h}_i \cdot \mathbf{h}_j}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|}$ 为余弦相似度
- $\tau_{\text{sup}} = 1.4$ 为温度参数（监督学习场景）

**设计动机**：SupConLoss能够充分利用伪标签信息，将同类别样本拉近，不同类别样本推远，同时避免了交叉熵损失对硬标签的过度依赖，提升了模型的鲁棒性。

#### 3.6.2 无监督对比学习损失（ContrastiveLoss）

对于低质量样本 $\mathcal{S}_{\text{low}}$，我们使用**无监督对比学习损失**，通过数据增强生成正样本对：

$$\mathcal{L}_{\text{con}} = -\frac{1}{|\mathcal{S}_{\text{low}}|} \sum_{i \in \mathcal{S}_{\text{low}}} \log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau_{\text{unsup}})}{\sum_{j \neq i} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau_{\text{unsup}})}$$

其中：
- $\mathbf{h}_i^+$ 为样本 $i$ 的增强版本（通过随机dropout或特征扰动生成）
- $\tau_{\text{unsup}} = 1.0$ 为无监督学习的温度参数

**设计动机**：低质量样本的伪标签不可靠，因此使用无监督对比学习，通过最大化样本与其增强版本的相似度，学习具有判别性的特征表示。

#### 3.6.3 聚类紧密度损失（Compactness Loss）

为了优化聚类质量，我们设计了**紧密度损失**，鼓励同类样本聚集在聚类中心周围：

$$\mathcal{L}_{\text{compact}} = \frac{1}{|\mathcal{S}_{\text{high}}|} \sum_{i \in \mathcal{S}_{\text{high}}} \max(0, \|\mathbf{h}_i - \mathbf{c}_{y_i}\|_2^2 - \delta_{\text{compact}})^2$$

其中：
- $\mathbf{c}_{y_i}$ 为样本 $i$ 所属簇的中心
- $\delta_{\text{compact}} = 0.3$ 为紧密度阈值（margin）

**设计动机**：紧密度损失确保高质量样本与其聚类中心距离足够近，提升类内紧密度。

#### 3.6.4 聚类分离度损失（Separation Loss）

为了优化聚类质量，我们设计了**分离度损失**，鼓励不同聚类中心之间保持足够距离：

$$\mathcal{L}_{\text{separate}} = \frac{1}{K(K-1)} \sum_{k=1}^K \sum_{l \neq k} \max(0, \delta_{\text{separate}} - \|\mathbf{c}_k - \mathbf{c}_l\|_2)^2$$

其中：
- $\delta_{\text{separate}} = 0.8$ 为分离度阈值（margin）

**设计动机**：分离度损失确保不同聚类中心之间距离足够远，提升类间分离度。

#### 3.6.5 总损失函数

最终的总损失为四个损失的加权组合：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{supcon}} \mathcal{L}_{\text{supcon}} + \lambda_{\text{con}} \mathcal{L}_{\text{con}} + \lambda_{\text{compact}} \mathcal{L}_{\text{compact}} + \lambda_{\text{separate}} \mathcal{L}_{\text{separate}}$$

其中：
- $\lambda_{\text{supcon}} = 1.0$：监督对比学习损失权重
- $\lambda_{\text{con}} = 0.5$：无监督对比学习损失权重
- $\lambda_{\text{compact}} = 0.3$：紧密度损失权重
- $\lambda_{\text{separate}} = 0.2$：分离度损失权重

**设计动机**：通过加权组合不同损失，UMC能够同时优化特征表示质量和聚类结构，实现更有效的无监督学习。这种联合损失优化策略是UMC的第四个创新点，与三个架构创新点协同工作，共同提升聚类性能。

### 3.7 训练过程

UMC的训练过程分为两个阶段：

**阶段一：预训练阶段。** 使用对比学习预训练多模态编码器，学习基础的多模态表示。预训练使用InfoNCE损失，温度参数 $\tau_{\text{pre}} = 0.2$。

**阶段二：主训练阶段。** 使用自适应渐进式学习策略，联合优化监督学习损失和对比学习损失。在每个epoch：
1. 计算自适应阈值 $\tau_{\text{adaptive}}$
2. 使用K-means++初始化聚类中心
3. 根据阈值选择高质量和低质量样本
4. 计算总损失并反向传播
5. 更新模型参数
6. 检查早停条件

训练使用Adam优化器，学习率 $lr = 3 \times 10^{-4}$，批次大小 $B = 128$，最大训练轮数 $E = 100$。

#### 3.5.9 伪标签生成和使用策略

UMC的伪标签生成和使用策略是其聚类学习的关键创新，通过分样本类型的训练策略，充分利用所有样本的同时避免伪标签污染。

##### 3.5.9.1 伪标签生成

伪标签直接来源于K-means聚类结果：

$$\text{pseudo\_label}_i = \arg\min_{k=1,\ldots,K} \|\mathbf{h}_i - \mathbf{c}_k\|_2$$

即每个样本被分配到最近的聚类中心对应的簇。

##### 3.5.9.2 样本分类策略

基于自适应阈值和密度选择，UMC将样本分为两类：

**高质量样本** $\mathcal{S}_{\text{high}}$：
- **特征**：密度高（接近聚类中心）、距离聚类中心近
- **伪标签可靠性**：高（伪标签更可能正确）
- **训练策略**：使用**监督学习**
  - 损失函数：$\mathcal{L}_{\text{supcon}} + \mathcal{L}_{\text{compact}} + \mathcal{L}_{\text{separate}}$
  - 目标：充分利用可靠的伪标签进行监督学习，提升特征表示质量

**低质量样本** $\mathcal{S}_{\text{low}}$：
- **特征**：密度低（远离聚类中心）、距离聚类中心远
- **伪标签可靠性**：低（伪标签可能错误）
- **训练策略**：使用**无监督对比学习**
  - 损失函数：$\mathcal{L}_{\text{con}}$
  - 目标：通过对比学习提升特征表示质量，不使用不可靠的伪标签

##### 3.5.9.3 设计优势

1. **分而治之**：针对不同质量的样本采用不同的学习策略，最大化学习效果
2. **避免伪标签污染**：低质量样本不使用不可靠的伪标签，避免错误信息传播
3. **充分利用数据**：所有样本都参与训练，但采用不同策略，避免数据浪费
4. **动态调整**：随着训练进行，自适应阈值动态调整，更多样本逐渐被标记为高质量

##### 3.5.9.4 与K-means++和密度选择的协同

**完整的协同机制**：
1. **K-means++初始化**：提供好的初始聚类中心
2. **Warm Start策略**：使用上一轮中心初始化，提升稳定性
3. **密度计算**：评估样本质量，识别可靠样本
4. **自适应k值选择**：优化密度计算，提升选择准确性
5. **自适应阈值**：动态调整选择数量，适应训练进程
6. **分样本类型训练**：高质量样本监督学习，低质量样本无监督学习

这种多层次的协同机制是UMC在聚类学习中的核心创新，显著提升了无监督聚类的效果和稳定性。

---

## 写作要点说明

### 1. 结构组织（7个小节）
- **3.1 问题定义**：形式化定义问题
- **3.2 UMC框架概述**：整体架构介绍
- **3.3-3.5**：三个创新点的详细描述
- **3.6 损失函数设计**：各种损失的数学定义
- **3.7 训练过程**：训练流程说明

### 2. 数学公式要求
- 所有关键操作都要有数学公式
- 公式要清晰、规范
- 变量要明确定义

### 3. 如何修改
- **根据实际实现调整公式**：如果代码实现与公式有差异，以代码为准
- **添加伪代码**：如果需要，可以在3.7节添加算法伪代码
- **扩展技术细节**：如果期刊要求，可以添加更多实现细节
- **调整长度**：根据期刊要求（通常6-8页），可以适当精简或扩展

### 4. 关键要点
- **创新点要突出**：每个创新点都要有清晰的动机、实现和效果
- **公式要准确**：所有公式都要与代码实现一致
- **流程要清晰**：训练过程要易于理解和复现

