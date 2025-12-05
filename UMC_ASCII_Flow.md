# UMC模型ASCII流程图

## 完整数据流程

```
输入数据
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   预处理阶段                                │
│  text_feats (B,L,text_dim)  →  float32 → GPU              │
│  video_feats (B,L,video_dim) →  float32 → GPU             │
│  audio_feats (B,L,audio_dim) →  float32 → GPU             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                 基础特征提取                                │
│                                                             │
│  text_feats → BERT → text_bert (B,L,t_dim)                 │
│                ↓                                           │
│              Linear → text_feat (B,L,base_dim)             │
│                ↓                                           │
│            LayerNorm → 标准化文本特征                       │
│                                                             │
│  video_feats → Linear → video_seq (B,L,base_dim)          │
│                ↓                                           │
│            LayerNorm → 标准化视频特征                       │
│                                                             │
│  audio_feats → Linear → audio_seq (B,L,base_dim)          │
│                ↓                                           │
│            LayerNorm → 标准化音频特征                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│              ConFEDE双投影处理 (创新点1)                    │
│                                                             │
│  video_seq → VideoDualProjector                            │
│    ├── simi_proj: 主要信息 (内容、动作)                    │
│    └── dissimi_proj: 环境信息 (光照、背景)                 │
│    ↓ 融合 → 增强视频特征                                    │
│                                                             │
│  audio_seq → AudioDualProjector                            │
│    ├── simi_proj: 语音内容 (情感、语义)                    │
│    └── dissimi_proj: 环境信息 (噪音、音质)                 │
│    ↓ 融合 → 增强音频特征                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│               多层注意力机制                                │
│                                                             │
│  转置: (B,L,base_dim) → (L,B,base_dim)                    │
│                                                             │
│  交叉注意力:                                                │
│  text_feat_t ←→ video_seq_t → x_video                      │
│  text_feat_t ←→ audio_seq_t → x_audio                      │
│                                                             │
│  文本引导注意力:                                            │
│  text_feat_t → text_guided_video_attn → text_guided_video  │
│  text_feat_t → text_guided_audio_attn → text_guided_audio  │
│                                                             │
│  自注意力融合:                                              │
│  [text_feat_t, text_guided_video, text_guided_audio]      │
│    ↓ 自注意力层                                             │
│  attended_features → 特征分离                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│             特征池化和交互                                  │
│                                                             │
│  BERT CLS: text_bert_cls (B,t_dim)                         │
│    ↓ 投影                                                   │
│  text_bert_proj (B,base_dim)                               │
│                                                             │
│  注意力池化:                                                │
│  enhanced_video_feat + text_bert_proj → text_video_enh_pooled│
│  enhanced_audio_feat + text_bert_proj → text_audio_enh_pooled│
│                                                             │
│  特征交互:                                                  │
│  [text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled]│
│    ↓ feature_interaction                                    │
│  交互后特征                                                 │
│                                                             │
│  门控融合:                                                  │
│  交互后特征 → gated_fusion → enhanced_text                  │
│  enhanced_text → LayerNorm → 最终融合特征                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│           聚类优化路径 (创新点2)                            │
│                                                             │
│  全局特征提取:                                              │
│  ├── text_global: text_bert_cls (B,t_dim)                  │
│  ├── video_global: video_feats.mean(dim=1) (B,video_dim)   │
│  └── audio_global: audio_feats.mean(dim=1) (B,audio_dim)   │
│                                                             │
│  聚类投影:                                                  │
│  ├── text_global → ClusteringProjector → text_clustered     │
│  ├── video_global → ClusteringProjector → video_clustered   │
│  └── audio_global → ClusteringProjector → audio_clustered   │
│                                                             │
│  聚类融合:                                                  │
│  [text_clustered, video_clustered, audio_clustered]        │
│    ↓ ClusteringFusion                                       │
│  clustering_features (B,base_dim)                          │
│                                                             │
│  特征融合:                                                  │
│  features + clustering_features × weight → 最终特征         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   损失计算                                  │
│                                                             │
│  对比学习:                                                  │
│  features → contrastive_proj → contrastive_features        │
│  contrastive_features + labels → contrastive_loss          │
│                                                             │
│  聚类损失:                                                  │
│  clustering_features + labels → enhanced_clustering_loss   │
│    ├── compactness_loss: 类内紧密度                        │
│    └── separation_loss: 类间分离度                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   输出结果                                  │
│                                                             │
│  根据mode返回:                                              │
│  ├── 'features': features (B,base_dim)                     │
│  ├── 'clustering-features': clustering_features (B,base_dim)│
│  ├── 'train-mm': features, mlp_output, losses...           │
│  └── 'pretrain-mm': mlp_output, losses...                  │
└─────────────────────────────────────────────────────────────┘
```

## 训练流程

```
训练开始
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   聚类过程 (K-means++)                       │
│                                                             │
│  特征提取 → K-means++初始化 → 聚类中心计算 → 伪标签生成     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   训练循环                                  │
│                                                             │
│  Epoch 0: K-means++初始化聚类中心                          │
│  Epoch 1+: 使用上一轮聚类中心                              │
│                                                             │
│  每个Epoch:                                                 │
│  ├── 聚类 → 伪标签                                         │
│  ├── 监督学习 (高质量样本)                                  │
│  ├── 无监督学习 (低质量样本)                               │
│  └── 损失更新                                              │
└─────────────────────────────────────────────────────────────┘
```

## 关键创新点总结

```
创新点1: ConFEDE机制
├── 目的: 从单一模态提取相似和相异信息
├── 实现: VideoDualProjector + AudioDualProjector
└── 效果: 增强特征表达能力

创新点2: 聚类优化路径  
├── 目的: 专门优化聚类任务的特征
├── 实现: ClusteringProjector + ClusteringFusion
└── 效果: 提升聚类质量

创新点3: 多层注意力融合
├── 目的: 深度理解多模态交互
├── 实现: 交叉注意力 + 文本引导注意力 + 自注意力
└── 效果: 更好的多模态融合
```
