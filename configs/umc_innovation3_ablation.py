"""
创新点三：聚类优化的多模态架构设计 - 消融实验配置
"""

class Innovation3AblationConfig:
    """创新点三消融实验配置类"""
    
    def __init__(self):
        self.base_config = self._get_base_config()
        self.experiments = self._get_innovation3_experiments()
    
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
            
            # 创新点三相关参数
            'enable_clustering_optimization': True,
            'clustering_weight': 1.0,
            'contrastive_weight': 0.5,
            'compactness_weight': 1.0,
            'separation_weight': 1.0,
            'contrastive_temperature': 0.07,
        }
    
    def _get_innovation3_experiments(self):
        """创新点三消融实验配置"""
        return {
            # ==================== 创新点三：聚类优化的多模态架构设计消融 ====================
            
            # 3.1 基础对比实验
            'baseline_no_clustering': {
                'name': 'Baseline (No Clustering Optimization)',
                'description': '关闭所有聚类优化功能，使用标准架构',
                'config': {
                    'enable_clustering_optimization': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'clustering_projector_only': {
                'name': 'Clustering Projector Only',
                'description': '仅启用聚类投影器',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'clustering_fusion_only': {
                'name': 'Clustering Fusion Only',
                'description': '仅启用聚类融合器',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'attention_pooling_only': {
                'name': 'Attention Pooling Only',
                'description': '仅启用注意力池化',
                'config': {
                    'enable_clustering_optimization': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': True,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'layer_norm_only': {
                'name': 'Layer Norm Only',
                'description': '仅启用LayerNorm优化',
                'config': {
                    'enable_clustering_optimization': False,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': True,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            # 3.2 损失函数消融
            'clustering_loss_only': {
                'name': 'Clustering Loss Only',
                'description': '仅启用聚类损失函数',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'contrastive_loss_only': {
                'name': 'Contrastive Loss Only',
                'description': '仅启用对比学习损失',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.5,
                }
            },
            
            'both_losses': {
                'name': 'Both Clustering and Contrastive Losses',
                'description': '同时启用聚类损失和对比学习损失',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 0.5,
                }
            },
            
            # 3.3 架构组件组合实验
            'projector_fusion': {
                'name': 'Projector + Fusion',
                'description': '聚类投影器 + 聚类融合器',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': False,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'projector_pooling': {
                'name': 'Projector + Attention Pooling',
                'description': '聚类投影器 + 注意力池化',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': False,
                    'use_attention_pooling': True,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'fusion_pooling': {
                'name': 'Fusion + Attention Pooling',
                'description': '聚类融合器 + 注意力池化',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': False,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'projector_fusion_pooling': {
                'name': 'Projector + Fusion + Pooling',
                'description': '聚类投影器 + 聚类融合器 + 注意力池化',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': False,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            # 3.4 完整架构实验
            'full_architecture_no_loss': {
                'name': 'Full Architecture (No Loss)',
                'description': '完整聚类优化架构，但不使用聚类损失',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'full_architecture_clustering_loss': {
                'name': 'Full Architecture + Clustering Loss',
                'description': '完整聚类优化架构 + 聚类损失',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': False,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 0.0,
                }
            },
            
            'full_architecture_contrastive_loss': {
                'name': 'Full Architecture + Contrastive Loss',
                'description': '完整聚类优化架构 + 对比学习损失',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 0.0,
                    'contrastive_weight': 0.5,
                }
            },
            
            'full_innovation3': {
                'name': 'Full Innovation 3',
                'description': '完整的创新点三：聚类优化的多模态架构设计',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 0.5,
                }
            },
            
            # 3.5 权重消融实验
            'weight_ablation_clustering_0.5': {
                'name': 'Clustering Weight 0.5',
                'description': '聚类损失权重设为0.5',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 0.5,
                    'contrastive_weight': 0.5,
                }
            },
            
            'weight_ablation_clustering_2.0': {
                'name': 'Clustering Weight 2.0',
                'description': '聚类损失权重设为2.0',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 2.0,
                    'contrastive_weight': 0.5,
                }
            },
            
            'weight_ablation_contrastive_0.1': {
                'name': 'Contrastive Weight 0.1',
                'description': '对比学习权重设为0.1',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 0.1,
                }
            },
            
            'weight_ablation_contrastive_1.0': {
                'name': 'Contrastive Weight 1.0',
                'description': '对比学习权重设为1.0',
                'config': {
                    'enable_clustering_optimization': True,
                    'use_clustering_projector': True,
                    'use_clustering_fusion': True,
                    'use_attention_pooling': True,
                    'use_layer_norm': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'clustering_weight': 1.0,
                    'contrastive_weight': 1.0,
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
            'architecture_components': [
                'baseline_no_clustering', 'clustering_projector_only', 'clustering_fusion_only',
                'attention_pooling_only', 'layer_norm_only'
            ],
            'loss_functions': [
                'clustering_loss_only', 'contrastive_loss_only', 'both_losses'
            ],
            'component_combinations': [
                'projector_fusion', 'projector_pooling', 'fusion_pooling', 'projector_fusion_pooling'
            ],
            'full_architecture': [
                'full_architecture_no_loss', 'full_architecture_clustering_loss',
                'full_architecture_contrastive_loss', 'full_innovation3'
            ],
            'weight_ablation': [
                'weight_ablation_clustering_0.5', 'weight_ablation_clustering_2.0',
                'weight_ablation_contrastive_0.1', 'weight_ablation_contrastive_1.0'
            ]
        }


# 使用示例
if __name__ == "__main__":
    ablation_config = Innovation3AblationConfig()
    
    # 获取特定实验配置
    full_experiment = ablation_config.get_experiment_config('full_innovation3')
    print(f"实验名称: {full_experiment['name']}")
    print(f"实验描述: {full_experiment['description']}")
    
    # 获取实验分组
    groups = ablation_config.get_experiment_groups()
    print(f"架构组件消融: {groups['architecture_components']}")
    print(f"损失函数消融: {groups['loss_functions']}")
    print(f"组件组合消融: {groups['component_combinations']}")
    print(f"完整架构消融: {groups['full_architecture']}")
    print(f"权重消融: {groups['weight_ablation']}")
