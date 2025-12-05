## MIntRec 数据集超参数实验方案

> 目的：围绕 UMC 项目的三个创新点（双投影模态分解、文本引导多模态融合、S 型渐进式阈值）设计可直接执行、可写入论文的超参数实验，并确认其可行性。

---

### 0. 实验环境与公共设置
- 数据集：`MIntRec`
- 模型：启用三大创新点的 UMC 改进版
- 统一训练超参：`lr=1e-4`、`batch_size=64`、`epochs=100`、`optimizer=AdamW (betas=0.9/0.999)`、`warmup_ratio=0.1`
- 指标：`NMI / ARI / ACC`，核心配置均跑 `3` 个随机种子，报告 `mean ± std`
- 记录：所有实验写入 `logs/mintrec_hparam/`；命名示例 `dualProj_dim256_seed0`

---

## 1. 创新点一：双投影模态分解

### 1.1 机制对比（dual_projection vs simple_linear）
| 超参 | 取值 |
|------|------|
| `dual_projection_strategy` | `dual_projection` / `simple_linear` |
| 其它 | `projection_dim=256`, `dropout=0.1` |

- **理由**：直接回答“显式语义/干扰解耦是否优于黑盒线性映射”，是论文主表（表 1.a）。
- **实现**：使用 `python run_dual_projection_compare.py --dataset MIntRec --gpu 0 --seeds 0 1 2 --train --save_results`，脚本会依次运行 `dual_projection_full` 与 `dual_projection_simple_linear` 两个配置。
- **呈现**：表格形式，列出 `mean±std`；附一句分析。
- **预期**：`dual_projection` 在 NMI/ARI/ACC 上比 `simple_linear` 至少高 1~3 点。

### 1.2 子空间维度敏感性
| 超参 | 取值 |
|------|------|
| `projection_dim` | `128 / 256 / 384` |
| `dual_projection_strategy` | `dual_projection` 与 `simple_linear` 均执行 |

- **理由**：说明性能不是偶然出现在某个维度；展示方法鲁棒性。
- **呈现**：表 1.b（六行），并绘制折线图（NMI vs dim）。
- **预期**：`dual_projection` 曲线整体高于 `simple_linear`，维度变化时波动更小。

### 1.3 投影层 dropout 稳健性
| 超参 | 取值 |
|------|------|
| `dropout` | `0.0 / 0.1 / 0.3` |
| `dual_projection_strategy` | 固定 `dual_projection` |

- **理由**：验证机制对正则化不敏感，回应“是否脆弱”。
- **呈现**：表 1.c。
- **预期**：`dropout=0.1` 最佳，0/0.3 差距小 → 机制稳健。

---

## 2. 创新点二：文本引导多模态融合

### 2.1 融合策略对比（text_guided vs direct_concat）
| 超参 | 取值 |
|------|------|
| `fusion_strategy` | `text_guided` / `direct_concat` |
| 统一 | `nheads=8`, `num_layers=2`, `dropout=0.1` |

- **理由**：证明“深度文本引导”优于“简单拼接”，论文主表（表 2.a）。
- **呈现**：表格 + 文字点评。
- **预期**：`text_guided` 全面优于 `direct_concat`。

### 2.2 注意力容量扫描
| 实验 | 固定 | 扫描 |
|------|------|------|
| 2.2.a | `num_layers=2` | `nheads=4/8/12` |
| 2.2.b | `nheads=8` | `num_layers=1/2/3` |

- **理由**：回应“是否仅靠加深/加宽”；展示合理容量即可取得收益。
- **呈现**：表 2.b/2.c + 两张折线图。
- **预期**：性能随容量增加先升后平，证明机制本身贡献大。

### 2.3 文本锚点权重（若有 `alpha_text`）
| 超参 | 取值 |
|------|------|
| `alpha_text` | `0.3 / 0.5 / 0.7` |

- **理由**：展示文本锚点强度对性能的影响，证明合理权重范围。
- **呈现**：表 2.d + 折线。
- **预期**：`alpha_text=0.5` 最优，过低/过高略降，但整体保持领先。

---

## 3. 创新点三：S 型渐进阈值 + 软重加权

### 3.1 调度策略对比（S-curve vs linear）
| 超参 | 取值 |
|------|------|
| `threshold_scheduling` | `s_curve` / `linear` |
| 统一 | `initial=0.5`, `max=0.9`, `sample_selection_strategy=soft_reweighting` |

- **理由**：直接对标 UMC 原法，证明 S 曲线更契合学习；论文主表（表 3.a）。
- **呈现**：表格 + 收敛曲线（epoch vs NMI）。
- **预期**：S 曲线收敛更平稳、最终指标更高。

### 3.2 曲线形状参数（steepness）
| 超参 | 取值 |
|------|------|
| `steepness` | `2 / 4 / 6` |

- **理由**：证明不是依赖某个“魔法曲线”，形状可调。
- **呈现**：表 3.b + “高置信样本比例”曲线。
- **预期**：中等 steepness 最佳，整体差距小 → 方案鲁棒。

### 3.3 样本选择策略（soft vs hard）
| 超参 | 取值 |
|------|------|
| `sample_selection_strategy` | `soft_reweighting` / `hard_threshold` |
| `threshold_scheduling` | 固定 `s_curve` |

- **理由**：证明软重加权优于硬筛选，可充分利用边界样本。
- **呈现**：表 3.c，附 `SelectedRatio@epoch50`。
- **预期**：软重加权指标更高、方差更小；硬阈值选样比例低且不稳定。

---

## 4. 执行顺序与论文呈现

1. **基础配置验证**：先用默认创新点组合在 MIntRec 上跑通，确认日志路径与评估脚本。
2. **主对比三张表**：1.1 + 2.1 + 3.1 → 形成论文主表，展示“我们的机制 vs 简单/原方案”。
3. **敏感性研究**：1.2/1.3, 2.2/2.3, 3.2/3.3 → 每个创新点至少一张附图/附表说明鲁棒性。
4. **最终组合配置**：选出最佳超参组合（例如 `dual_projection + text_guided + s_curve + soft_reweighting`，`projection_dim=256`, `nheads=8`, `steepness=4`），跑 3 seeds，作为 MIntRec 最终结果与原版 UMC 对比。

---

## 5. 可行性确认
- 所有建议实验只涉及现有配置文件中的开关或简单数值修改，不需要新数据。  
- 训练成本：单次 MIntRec 约 3~4 小时（视算力），总实验可分阶段并行。  
- 记录方式：结合 `run_ablation_experiments.py`、`quick_ablation.py`，或在 `configs/umc_ablation_param.py` 中新增相应 `experiment_name`。  
- 结果呈现：  
  - 主文：三张对比表 + 1~2 张关键曲线（收敛曲线、维度敏感性）。  
  - 附录：完整超参数扫描表格。  

综上，这套实验方案完全基于现有代码与数据即可执行，覆盖“机制对比 + 超参敏感性 + 稳健性”三个层次，满足撰写论文所需的论据。

