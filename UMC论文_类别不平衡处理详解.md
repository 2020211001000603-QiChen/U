# UMC类别不平衡处理机制详解

## 🎯 核心机制

UMC通过**最小样本数保证机制**和**动态阈值调整**相结合，有效处理类别不平衡问题。

---

## 1. 最小样本数的计算

### 1.1 计算公式

```python
min_samples_per_class = max(5, int(len(feats) / self.num_labels * 0.1))
```

**数学公式**：
$$N_{\text{min}} = \max(5, \lfloor 0.1 \cdot \frac{|\mathcal{D}|}{K} \rfloor)$$

其中：
- $|\mathcal{D}|$ 为总样本数（`len(feats)`）
- $K$ 为类别数（`self.num_labels`）
- $0.1$ 为比例系数（10%）
- $5$ 为最小保证值

### 1.2 计算逻辑

**双重保证机制**：
1. **绝对最小值**：至少5个样本（防止极端情况）
2. **相对最小值**：总数的10%除以类别数（适应数据集规模）

**取两者中的较大值**，确保：
- 小数据集：至少每个类别5个样本
- 大数据集：每个类别至少有总数的10%/K个样本

### 1.3 实际例子

**例子1：MIntRec数据集**
- 总样本数：$|\mathcal{D}| = 10,000$
- 类别数：$K = 20$
- 计算：$N_{\text{min}} = \max(5, \lfloor 0.1 \times \frac{10000}{20} \rfloor) = \max(5, 50) = 50$
- **结果**：每个类别至少选择50个样本

**例子2：IEMOCAP-DA数据集（小数据集）**
- 总样本数：$|\mathcal{D}| = 6,000$
- 类别数：$K = 9$
- 计算：$N_{\text{min}} = \max(5, \lfloor 0.1 \times \frac{6000}{9} \rfloor) = \max(5, 66) = 66$
- **结果**：每个类别至少选择66个样本

**例子3：极端小数据集**
- 总样本数：$|\mathcal{D}| = 500$
- 类别数：$K = 10$
- 计算：$N_{\text{min}} = \max(5, \lfloor 0.1 \times \frac{500}{10} \rfloor) = \max(5, 5) = 5$
- **结果**：每个类别至少选择5个样本（使用绝对最小值）

---

## 2. 动态调整机制

### 2.1 每个簇的选择数量计算

```python
cutoff = max(
    int(len(cluster_samples) * threshold), 
    min(min_samples_per_class, len(cluster_samples))
)
```

**数学公式**：
$$\text{cutoff}_k = \max(\lfloor |C_k| \cdot \tau_{\text{adaptive}} \rfloor, \min(N_{\text{min}}, |C_k|))$$

其中：
- $|C_k|$ 为簇 $C_k$ 的样本数（`len(cluster_samples)`）
- $\tau_{\text{adaptive}}$ 为自适应阈值（`threshold`）
- $N_{\text{min}}$ 为最小样本数（`min_samples_per_class`）

### 2.2 动态调整的两个层面

#### 层面1：自适应阈值 $\tau_{\text{adaptive}}$ 的动态调整

**每个epoch都会重新计算阈值**：

```python
# 在训练循环中
threshold = self.progressive_learner.compute_threshold(
    epoch, args.num_train_epochs, current_performance, current_loss
)
```

**阈值的变化**：
- **早期epoch**（0-20%）：$\tau \approx 0.05-0.15$（保守，选择少）
- **中期epoch**（20-60%）：$\tau \approx 0.15-0.35$（快速增长）
- **后期epoch**（60-90%）：$\tau \approx 0.35-0.45$（稳定增长）
- **最终epoch**（90-100%）：$\tau \approx 0.45-0.5$（微调）

**自适应调整**：
- 性能提升时：$\tau$ 增加（选择更多样本）
- 性能下降时：$\tau$ 减少（选择更少样本）
- 损失下降时：$\tau$ 增加
- 损失上升时：$\tau$ 减少

#### 层面2：选择数量 $\text{cutoff}_k$ 的动态调整

**由于阈值动态变化，每个簇的选择数量也会动态变化**：

$$\text{cutoff}_k^{(t)} = \max(\lfloor |C_k| \cdot \tau_{\text{adaptive}}^{(t)} \rfloor, \min(N_{\text{min}}, |C_k|))$$

其中 $t$ 表示epoch。

### 2.3 动态调整的完整流程

```
Epoch 0:
  threshold = 0.05 (初始阈值)
  cutoff_k = max(|C_k| * 0.05, min(50, |C_k|))
  → 选择较少的样本（保守策略）

Epoch 10:
  threshold = 0.20 (S型曲线增长 + 性能调整)
  cutoff_k = max(|C_k| * 0.20, min(50, |C_k|))
  → 选择更多样本（如果性能提升）

Epoch 50:
  threshold = 0.40 (继续增长)
  cutoff_k = max(|C_k| * 0.40, min(50, |C_k|))
  → 选择更多样本（充分利用已学习的特征）

Epoch 90:
  threshold = 0.48 (接近最大值)
  cutoff_k = max(|C_k| * 0.48, min(50, |C_k|))
  → 选择最多样本（精细优化）
```

---

## 3. 特殊情况处理

### 3.1 极少数类处理

```python
if len(cluster_samples) <= 5:
    cutoff = len(cluster_samples)
    # 选择所有样本
```

**处理逻辑**：
- 如果簇的样本数 $\leq 5$，直接选择所有样本
- **原因**：极少数类如果还按比例选择，可能只有1-2个样本，无法有效训练

**数学描述**：
$$\text{cutoff}_k = \begin{cases}
|C_k| & \text{if } |C_k| \leq 5 \\
\max(\lfloor |C_k| \cdot \tau_{\text{adaptive}} \rfloor, \min(N_{\text{min}}, |C_k|)) & \text{otherwise}
\end{cases}$$

### 3.2 最小样本数保证

**关键机制**：即使阈值很小，也保证每个类别至少有 $N_{\text{min}}$ 个样本

**例子**：
- 簇大小：$|C_k| = 100$
- 阈值：$\tau = 0.05$（早期epoch，很小）
- 计算：$\lfloor 100 \times 0.05 \rfloor = 5$
- 最小样本数：$N_{\text{min}} = 50$
- **最终cutoff**：$\max(5, \min(50, 100)) = \max(5, 50) = 50$
- **结果**：即使阈值很小，也选择50个样本（保证最小样本数）

---

## 4. 完整的选择流程

### 4.1 算法流程

```
输入：特征表示 {h_i}, 自适应阈值 τ, 类别数 K, 总样本数 |D|

1. 计算最小样本数
   N_min = max(5, floor(0.1 * |D| / K))

2. 对每个簇 C_k:
   a. 如果 |C_k| <= 5:
      cutoff_k = |C_k|  // 选择所有样本
   
   b. 否则:
      // 计算基于阈值的数量
      threshold_based = floor(|C_k| * τ)
      
      // 计算最小保证数量
      min_guaranteed = min(N_min, |C_k|)
      
      // 取两者中的较大值
      cutoff_k = max(threshold_based, min_guaranteed)
   
   c. 使用密度选择前cutoff_k个样本

3. 返回选中的样本
```

### 4.2 实际例子

**场景：MIntRec数据集，Epoch 20**

- 总样本数：$|\mathcal{D}| = 10,000$
- 类别数：$K = 20$
- 最小样本数：$N_{\text{min}} = 50$
- 当前阈值：$\tau = 0.25$（中期epoch）

**簇1（大簇）**：
- 簇大小：$|C_1| = 800$
- 阈值计算：$\lfloor 800 \times 0.25 \rfloor = 200$
- 最小保证：$\min(50, 800) = 50$
- **最终cutoff**：$\max(200, 50) = 200$
- **结果**：选择200个样本（由阈值决定）

**簇2（中等簇）**：
- 簇大小：$|C_2| = 300$
- 阈值计算：$\lfloor 300 \times 0.25 \rfloor = 75$
- 最小保证：$\min(50, 300) = 50$
- **最终cutoff**：$\max(75, 50) = 75$
- **结果**：选择75个样本（由阈值决定）

**簇3（小簇）**：
- 簇大小：$|C_3| = 50$
- 阈值计算：$\lfloor 50 \times 0.25 \rfloor = 12$
- 最小保证：$\min(50, 50) = 50$
- **最终cutoff**：$\max(12, 50) = 50$
- **结果**：选择50个样本（由最小保证决定，避免选择过少）

**簇4（极小簇）**：
- 簇大小：$|C_4| = 3$
- **特殊处理**：$|C_4| \leq 5$
- **最终cutoff**：$3$
- **结果**：选择所有3个样本（极少数类保护）

---

## 5. 动态调整的优势

### 5.1 适应训练进程

- **早期**：阈值小，选择少，避免低质量样本污染
- **中期**：阈值增长，选择增多，充分利用已学习的特征
- **后期**：阈值大，选择多，精细优化聚类质量

### 5.2 保证类别覆盖

- **最小样本数保证**：即使阈值很小，也保证每个类别有足够样本
- **极少数类保护**：极少数类选择所有样本，避免信息丢失

### 5.3 处理类别不平衡

- **大簇**：按阈值比例选择（可能选择很多）
- **小簇**：至少选择最小样本数（保证覆盖）
- **极小簇**：选择所有样本（完全保护）

---

## 6. 在论文中的描述

### 6.1 最小样本数计算

**位置**：Methodology 3.5.8.3节

**描述**：
> UMC通过最小样本数保证机制处理类别不平衡问题。最小样本数 $N_{\text{min}}$ 的计算公式为：
> $$N_{\text{min}} = \max(5, \lfloor 0.1 \cdot \frac{|\mathcal{D}|}{K} \rfloor)$$
> 其中 $|\mathcal{D}|$ 为总样本数，$K$ 为类别数。该公式采用双重保证机制：绝对最小值5个样本（防止极端情况），相对最小值总数的10%除以类别数（适应数据集规模），取两者中的较大值。

### 6.2 动态调整机制

**位置**：Methodology 3.5.8.4节

**描述**：
> 每个簇的选择数量 $\text{cutoff}_k$ 通过以下公式动态计算：
> $$\text{cutoff}_k = \max(\lfloor |C_k| \cdot \tau_{\text{adaptive}} \rfloor, \min(N_{\text{min}}, |C_k|))$$
> 其中 $\tau_{\text{adaptive}}$ 为自适应阈值，在每个epoch根据S型曲线增长和多维度自适应调整动态变化。这种动态调整机制具有两个层面：第一，自适应阈值根据训练进度和性能反馈动态调整；第二，选择数量根据阈值和簇大小动态调整，既保证数量又保证质量。

---

## 7. 关键参数总结

| 参数 | 符号 | 计算公式 | 说明 |
|------|------|----------|------|
| 最小样本数 | $N_{\text{min}}$ | $\max(5, \lfloor 0.1 \cdot \frac{|\mathcal{D}|}{K} \rfloor)$ | 每个类别至少选择的样本数 |
| 自适应阈值 | $\tau_{\text{adaptive}}$ | S型曲线 + 多维度调整 | 动态变化的阈值（0.05-0.5） |
| 选择数量 | $\text{cutoff}_k$ | $\max(\lfloor |C_k| \cdot \tau \rfloor, \min(N_{\text{min}}, |C_k|))$ | 每个簇实际选择的样本数 |
| 极少数类阈值 | - | $|C_k| \leq 5$ | 如果簇大小≤5，选择所有样本 |

---

## ✅ 总结

### 最小样本数
- **计算公式**：$N_{\text{min}} = \max(5, \lfloor 0.1 \cdot \frac{|\mathcal{D}|}{K} \rfloor)$
- **实际值**：通常在5-100之间，取决于数据集规模
- **作用**：保证每个类别至少有足够的样本用于训练

### 动态调整
- **两个层面**：
  1. 自适应阈值 $\tau$ 的动态调整（每个epoch）
  2. 选择数量 $\text{cutoff}_k$ 的动态调整（跟随阈值变化）
- **调整依据**：
  - S型曲线增长（训练进度）
  - 性能反馈（性能提升/下降）
  - 损失反馈（损失下降/上升）
  - 稳定性反馈（训练稳定性）
- **效果**：既适应训练进程，又保证类别覆盖

