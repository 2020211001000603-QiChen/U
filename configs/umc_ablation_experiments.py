"""
UMC消融实验配置文件
基于3个主要创新点的消融实验设计
"""

class UMCAblationConfig:
    """UMC消融实验配置类"""
    
    def __init__(self):
        self.base_config = self._get_base_config()
        self.experiments = self._get_ablation_experiments()
    
    def _get_base_config(self):
        """基础配置参数"""
        return {
            # 基础模型参数
            'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
            'pretrain_batch_size': 128,
            'train_batch_size': 128,
            'eval_batch_size': 128,
            'test_batch_size': 128,
            'num_pretrain_epochs': 50,
            'num_train_epochs': 50,
            'pretrain': [True],
            'aligned_method': 'ctc',
            'need_aligned': False,
            'freeze_pretrain_bert_parameters': [True],
            'freeze_train_bert_parameters': [True],
            'pretrain_temperature': [0.2],
            'train_temperature_sup': [1.4],
            'train_temperature_unsup': [1],
            'activation': 'tanh',
            'lr_pre': 2e-5,
            'lr': [3e-4],
            'weight_decay': 0.01,
            'feat_dim': 768,
            'hidden_size': 768,
            'grad_clip': -1.0,
            'warmup_proportion': 0.1,
            'hidden_dropout_prob': 0.1,
            'weight': 1.0,
            'loss_mode': 'rdrop',
            'base_dim': 256,
            'nheads': 8,
            'attn_dropout': 0.1,
            'relu_dropout': 0.1,
            'embed_dropout': 0.1,
            'res_dropout': 0.0,
            'attn_mask': True,
            'encoder_layers_1': 1,
            'fusion_act': 'tanh',
            
            # 数据集特定参数
            'text_feat_dim': 768,
            'video_feat_dim': 256,
            'audio_feat_dim': 768,
            
            # 损失权重
            'clustering_weight': 1.0,
            'contrastive_weight': 0.5,
            'compactness_weight': 0.3,
            'separation_weight': 0.2,
            'contrastive_temperature': 0.07,
        }
    
    def _get_ablation_experiments(self):
        """消融实验配置"""
        return {
            # ==================== 实验1: ConFEDE多模态特征增强机制消融 ====================
            
            # 1.1 基础对比实验
            'baseline_traditional': {
                'name': 'Baseline (Traditional Fusion)',
                'description': '使用传统的简单拼接融合，关闭所有ConFEDE机制',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'confede_only': {
                'name': 'ConFEDE Only',
                'description': '仅启用ConFEDE双投影机制',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'text_guided_only': {
                'name': 'Text-Guided Attention Only',
                'description': '仅启用文本引导注意力机制',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'gated_fusion_only': {
                'name': 'Gated Fusion Only',
                'description': '仅启用门控融合机制',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            # 1.2 渐进组合实验
            'confede_text_guided': {
                'name': 'ConFEDE + Text-Guided',
                'description': 'ConFEDE双投影 + 文本引导注意力',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'confede_gated': {
                'name': 'ConFEDE + Gated-Fusion',
                'description': 'ConFEDE双投影 + 门控融合',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'full_confede': {
                'name': 'Full ConFEDE Mechanism',
                'description': '完整的ConFEDE特征增强机制',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            # ==================== 实验2: 渐进式聚类学习策略消融 ====================
            
            # 2.1 策略对比实验
            'fixed_strategy': {
                'name': 'Fixed Strategy',
                'description': '使用固定的阈值和传统损失',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'enable_enhanced_loss': False,
                    'enable_kmeans_plus': False,
                    'thres': [0.25],  # 固定阈值
                    'delta': [0.0],   # 无增长
                }
            },
            
            'progressive_only': {
                'name': 'Progressive Strategy Only',
                'description': '仅启用渐进策略',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': False,
                    'enable_kmeans_plus': False,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            'enhanced_loss_only': {
                'name': 'Enhanced Loss Only',
                'description': '仅启用增强的聚类损失',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': False,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': False,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                }
            },
            
            'kmeans_plus_only': {
                'name': 'K-means++ Only',
                'description': '仅启用K-means++初始化',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'enable_enhanced_loss': False,
                    'enable_kmeans_plus': True,
                }
            },
            
            # 2.2 策略组合实验
            'progressive_enhanced_loss': {
                'name': 'Progressive + Enhanced Loss',
                'description': '渐进策略 + 增强损失',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': False,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            'progressive_kmeans_plus': {
                'name': 'Progressive + K-means++',
                'description': '渐进策略 + K-means++',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': False,
                    'enable_kmeans_plus': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            'full_progressive': {
                'name': 'Full Progressive Strategy',
                'description': '完整的渐进式聚类学习策略',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            # ==================== 实验3: 聚类优化架构消融 ====================
            
            # 3.1 架构对比实验
            'standard_architecture': {
                'name': 'Standard Architecture',
                'description': '使用标准的Transformer架构',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                }
            },
            
            'clustering_projector_only': {
                'name': 'Clustering Projector Only',
                'description': '仅启用聚类投影器',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                }
            },
            
            'clustering_fusion_only': {
                'name': 'Clustering Fusion Only',
                'description': '仅启用聚类融合器',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                }
            },
            
            'attention_pooling_only': {
                'name': 'Attention Pooling Only',
                'description': '仅启用注意力池化',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': True,
                    'use_layer_norm': False,
                }
            },
            
            # 3.2 架构组合实验
            'clustering_optimized': {
                'name': 'Clustering-Optimized Architecture',
                'description': '聚类投影器 + 聚类融合器',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                }
            },
            
            'full_optimized_architecture': {
                'name': 'Full Optimized Architecture',
                'description': '完整的聚类优化架构',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                }
            },
            
            # ==================== 实验4: 创新点协同效果验证 ====================
            
            # 4.1 单创新点验证
            'only_confede': {
                'name': 'Only ConFEDE',
                'description': '仅使用ConFEDE特征增强机制',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                }
            },
            
            'only_progressive': {
                'name': 'Only Progressive Strategy',
                'description': '仅使用渐进式聚类学习策略',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            'only_architecture': {
                'name': 'Only Optimized Architecture',
                'description': '仅使用聚类优化架构',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                }
            },
            
            # 4.2 双创新点协同
            'confede_progressive': {
                'name': 'ConFEDE + Progressive',
                'description': 'ConFEDE + 渐进策略',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                }
            },
            
            'confede_architecture': {
                'name': 'ConFEDE + Architecture',
                'description': 'ConFEDE + 优化架构',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                }
            },
            
            'progressive_architecture': {
                'name': 'Progressive + Architecture',
                'description': '渐进策略 + 优化架构',
                'config': {
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    'enable_gated_fusion': False,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                }
            },
            
            # 4.3 完整模型
            'full_umc_model': {
                'name': 'Full UMC Model',
                'description': '完整的UMC模型（所有创新点）',
                'config': {
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    'enable_gated_fusion': True,
                    'enable_clustering_optimization': True,
                    'enable_progressive_learning': True,
                    'enable_enhanced_loss': True,
                    'enable_kmeans_plus': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'max_threshold': 0.5,
                    'min_threshold': 0.05,
                    'performance_window': 3,
                    'patience': 5,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                }
            },
        }
    
    def get_experiment_config(self, experiment_name):
        """获取特定实验的完整配置"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        # 合并基础配置和实验特定配置
        experiment_config = self.base_config.copy()
        experiment_config.update(self.experiments[experiment_name]['config'])
        
        return {
            'name': self.experiments[experiment_name]['name'],
            'description': self.experiments[experiment_name]['description'],
            'config': experiment_config
        }
    
    def get_all_experiments(self):
        """获取所有实验的配置"""
        return {name: self.get_experiment_config(name) for name in self.experiments.keys()}
    
    def get_experiment_groups(self):
        """获取实验分组"""
        return {
            'confede_ablation': [
                'baseline_traditional', 'confede_only', 'text_guided_only', 'gated_fusion_only',
                'confede_text_guided', 'confede_gated', 'full_confede'
            ],
            'progressive_ablation': [
                'fixed_strategy', 'progressive_only', 'enhanced_loss_only', 'kmeans_plus_only',
                'progressive_enhanced_loss', 'progressive_kmeans_plus', 'full_progressive'
            ],
            'architecture_ablation': [
                'standard_architecture', 'clustering_projector_only', 'clustering_fusion_only',
                'attention_pooling_only', 'clustering_optimized', 'full_optimized_architecture'
            ],
            'synergy_ablation': [
                'only_confede', 'only_progressive', 'only_architecture',
                'confede_progressive', 'confede_architecture', 'progressive_architecture',
                'full_umc_model'
            ]
        }


# 使用示例
if __name__ == "__main__":
    ablation_config = UMCAblationConfig()
    
    # 获取特定实验配置
    confede_experiment = ablation_config.get_experiment_config('full_confede')
    print(f"实验名称: {confede_experiment['name']}")
    print(f"实验描述: {confede_experiment['description']}")
    
    # 获取实验分组
    groups = ablation_config.get_experiment_groups()
    print(f"ConFEDE消融实验: {groups['confede_ablation']}")
    print(f"渐进策略消融实验: {groups['progressive_ablation']}")
    print(f"架构消融实验: {groups['architecture_ablation']}")
    print(f"协同效果实验: {groups['synergy_ablation']}")
