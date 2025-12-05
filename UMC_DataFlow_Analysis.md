# UMC项目详细数据流图分析

## 1. 整体数据流架构

```
输入数据 → 特征提取 → 多模态融合 → 聚类学习 → 结果输出
    ↓         ↓         ↓         ↓         ↓
  原始数据   编码特征   融合特征   聚类标签   评估指标
```

## 2. 详细数据流图

### 2.1 数据输入层 (Data Input Layer)

```
原始数据输入:
├── 文本数据 (Text Data)
│   ├── train.tsv/dev.tsv/test.tsv
│   ├── BERT Tokenization
│   └── 输入格式: [input_ids, attention_mask, token_type_ids]
│
├── 视频数据 (Video Data)  
│   ├── swin_feats.pkl
│   ├── Swin Transformer特征
│   └── 维度: [batch_size, seq_len, 256]
│
└── 音频数据 (Audio Data)
    ├── wavlm_feats.pkl  
    ├── WavLM特征
    └── 维度: [batch_size, seq_len, 768]
```

### 2.2 特征提取层 (Feature Extraction Layer)

```
特征提取流程:
├── 文本特征提取
│   ├── BERTEncoder
│   ├── 输入: [batch_size, 3, seq_len] (input_ids, attention_mask, token_type_ids)
│   ├── 输出: [batch_size, seq_len, 768]
│   └── 投影: Linear(768 → 256)
│
├── 视频特征提取
│   ├── 预提取的Swin特征
│   ├── 输入: [batch_size, seq_len, 256]
│   ├── Transformer编码器处理
│   └── 输出: [batch_size, seq_len, 256]
│
└── 音频特征提取
    ├── 预提取的WavLM特征
    ├── 输入: [batch_size, seq_len, 768]  
    ├── Transformer编码器处理
    └── 投影: Linear(768 → 256)
```

### 2.3 多模态融合层 (Multimodal Fusion Layer)

```
融合架构 (UMC.py):
├── 基础特征投影
│   ├── text_layer: Linear(768 → 256)
│   ├── video_layer: Linear(256 → 256)  
│   └── audio_layer: Linear(768 → 256)
│
├── ConFEDE机制 (可选)
│   ├── VideoDualProjector
│   │   ├── simi_proj: 主要信息投影
│   │   ├── dissimi_proj: 环境信息投影
│   │   └── fusion: 双投影融合
│   │
│   └── AudioDualProjector
│       ├── simi_proj: 语音内容投影
│       ├── dissimi_proj: 环境音投影
│       └── fusion: 双投影融合
│
├── 注意力机制
│   ├── 文本引导交叉注意力
│   │   ├── text_guided_video_attn
│   │   └── text_guided_audio_attn
│   │
│   ├── 自注意力层 (可选)
│   │   └── self_attention_layers (2层)
│   │
│   └── 特征交互层
│       └── feature_interaction
│
├── 门控融合机制
│   ├── text_weight_gate: Sigmoid(256 → 1)
│   ├── video_weight_gate: Sigmoid(256 → 1)
│   ├── audio_weight_gate: Sigmoid(256 → 1)
│   └── 加权融合: w1*text + w2*video + w3*audio
│
└── 最终融合
    ├── shared_embedding_layer: GELU + Dropout + Linear
    ├── fusion_layer: GELU + Dropout + Linear  
    └── LayerNorm归一化
```

### 2.4 聚类学习层 (Clustering Learning Layer)

```
聚类学习流程:
├── 高质量样本选择 (ConvexSampler)
│   ├── K-means++初始化聚类中心
│   ├── 计算样本到中心的距离
│   ├── 动态阈值选择高质量样本
│   └── 输出: select_ids, pseudo_labels
│
├── 对比学习损失
│   ├── InstanceLoss: 实例级对比
│   │   ├── 正样本: 同一实例的不同视图
│   │   ├── 负样本: 不同实例
│   │   └── 温度参数: 0.07
│   │
│   ├── ClusterLoss: 聚类级对比
│   │   ├── 正样本: 同一聚类的不同视图
│   │   ├── 负样本: 不同聚类
│   │   └── 熵正则化
│   │
│   └── SupConLoss: 监督对比学习
│       ├── 温度参数: 1.4 (监督), 1.0 (无监督)
│       └── 多视图对比
│
├── 聚类优化损失 (可选)
│   ├── CompactnessLoss: 紧密度损失
│   │   ├── 计算类内距离
│   │   └── Margin: 0.3
│   │
│   ├── SeparationLoss: 分离度损失
│   │   ├── 计算类间距离
│   │   └── Margin: 0.8
│   │
│   └── 总聚类损失: compactness + separation
│
└── 渐进式学习优化
    ├── AdaptiveProgressiveLearning
    ├── 自适应阈值调整
    ├── 性能监控和早停
    └── S型曲线阈值增长
```

### 2.5 训练流程 (Training Pipeline)

```
训练阶段:
├── 预训练阶段 (可选)
│   ├── 冻结BERT参数 (除最后2层)
│   ├── 对比学习预训练
│   ├── 温度参数: 0.2
│   └── 学习率: 2e-5
│
├── 主训练阶段
│   ├── 冻结BERT参数 (除最后2层)
│   ├── 学习率: 3e-4
│   ├── 训练轮数: 100
│   ├── 批次大小: 128
│   │
│   ├── 每个epoch:
│   │   ├── 聚类初始化 (K-means++)
│   │   ├── 高质量样本选择
│   │   ├── 伪标签生成
│   │   ├── 监督学习 (高质量样本)
│   │   ├── 无监督学习 (低质量样本)
│   │   └── 损失计算和反向传播
│   │
│   └── 早停机制
│       ├── 性能窗口: 3
│       ├── 耐心值: 5
│       └── 自适应阈值调整
│
└── 测试阶段
    ├── 特征提取
    ├── K-means聚类
    ├── 评估指标计算
    └── 结果保存
```

### 2.6 损失函数组合 (Loss Function Composition)

```
总损失函数:
├── 主要损失
│   ├── 监督对比损失 (SupConLoss)
│   │   ├── 权重: 1.0
│   │   ├── 温度: 1.4
│   │   └── 多视图对比
│   │
│   └── 无监督对比损失 (InstanceLoss)
│       ├── 权重: 1.0  
│       ├── 温度: 1.0
│       └── 实例级对比
│
├── 辅助损失 (可选)
│   ├── 聚类损失
│   │   ├── 紧密度损失: margin=0.3
│   │   ├── 分离度损失: margin=0.8
│   │   └── 权重: 0.1
│   │
│   └── 对比学习损失
│       ├── 温度: 0.07
│       └── 权重: 0.5
│
└── 总损失计算
    ├── 监督损失: SupConLoss
    ├── 无监督损失: InstanceLoss  
    ├── 聚类损失: CompactnessLoss + SeparationLoss
    └── 加权求和: w1*sup + w2*unsup + w3*cluster
```

### 2.7 评估指标 (Evaluation Metrics)

```
评估流程:
├── 聚类结果生成
│   ├── K-means聚类
│   ├── 聚类中心初始化
│   └── 样本分配
│
├── 指标计算
│   ├── NMI (Normalized Mutual Information)
│   │   └── 范围: [0, 1], 越高越好
│   │
│   ├── ARI (Adjusted Rand Index)  
│   │   └── 范围: [-1, 1], 越高越好
│   │
│   ├── ACC (Clustering Accuracy)
│   │   ├── 匈牙利算法对齐
│   │   └── 范围: [0, 1], 越高越好
│   │
│   └── FMI (Fowlkes-Mallows Index)
│       └── 范围: [0, 1], 越高越好
│
└── 结果保存
    ├── CSV格式保存
    ├── 多次运行平均
    └── 统计分析
```

## 3. 关键技术细节

### 3.1 数据维度变化

```
输入维度:
├── 文本: [batch_size, 3, seq_len] → [batch_size, seq_len, 768] → [batch_size, seq_len, 256]
├── 视频: [batch_size, seq_len, 256] → [batch_size, seq_len, 256]
└── 音频: [batch_size, seq_len, 768] → [batch_size, seq_len, 256]

融合后维度:
└── 多模态特征: [batch_size, seq_len, 256]

最终输出维度:
├── 特征表示: [batch_size, 256]
└── MLP输出: [batch_size, 256]
```

### 3.2 关键超参数

```
训练参数:
├── 学习率: 3e-4 (主训练), 2e-5 (预训练)
├── 批次大小: 128
├── 训练轮数: 100
├── 温度参数: 1.4 (监督), 1.0 (无监督), 0.07 (对比学习)
├── 权重衰减: 0.01
└── Dropout: 0.1

聚类参数:
├── 聚类数量: 数据集标签数量
├── 阈值范围: [0.05, 0.5]
├── 增量步长: 0.05
├── TopK: 5
└── Margin: 0.3 (紧密度), 0.8 (分离度)

模型参数:
├── 基础维度: 256
├── 注意力头数: 8
├── Transformer层数: 1
├── 自注意力层数: 2
└── 隐藏维度: 768
```

### 3.3 内存和计算复杂度

```
计算复杂度:
├── BERT编码: O(seq_len² × hidden_size)
├── Transformer编码: O(seq_len² × base_dim)
├── 注意力计算: O(seq_len² × base_dim × num_heads)
├── K-means聚类: O(n × k × iterations)
└── 对比学习: O(batch_size² × feature_dim)

内存使用:
├── 模型参数: ~100M (BERT + 融合层)
├── 特征缓存: ~1GB (预提取特征)
├── 梯度缓存: ~200MB
└── 总内存需求: ~2-4GB
```

## 4. 数据流优化策略

### 4.1 训练优化

```
优化策略:
├── 梯度裁剪: -1.0 (禁用)
├── 学习率调度: 线性warmup + 余弦退火
├── 参数冻结: BERT参数冻结 (除最后2层)
├── 混合精度: 支持FP16训练
└── 数据并行: 支持多GPU训练
```

### 4.2 推理优化

```
推理优化:
├── 特征缓存: 预计算和缓存特征
├── 模型量化: 支持INT8量化
├── 批处理: 批量推理提升效率
└── 内存管理: 动态内存分配
```

这个详细的数据流图展示了UMC项目从数据输入到最终结果输出的完整技术流程，包括每个阶段的具体实现细节、数据维度变化、关键算法和优化策略。
