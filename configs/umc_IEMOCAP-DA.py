
class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        
        # 处理消融实验配置
        ablation_config = self._get_ablation_config(args)
        
        if args.multimodal_method == 'umc':
            hyper_parameters = {
                'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
                'pretrain_batch_size': 64,        # 48GB GPU可以用更大的批处理
                'train_batch_size': 64,          # 48GB GPU可以用更大的批处理
                'eval_batch_size': 64,            # 48GB GPU可以用更大的批处理
                'test_batch_size': 64,            # 48GB GPU可以用更大的批处理
                'num_pretrain_epochs': [150],        # 从100增加到150，更多预训练（针对信息量不足）
                'num_train_epochs': [100],
                'pretrain': [True],               # 启用预训练（与MELD-DA一致）
                'aligned_method': 'ctc',
                'need_aligned': False,
                'freeze_pretrain_bert_parameters': True,
                'freeze_train_bert_parameters': True,
                'pretrain_temperature': [0.2],
                'train_temperature_sup': [20],      # 从10提高到20，学习更平滑的分布（诊断建议）
                'train_temperature_unsup': [20],    # IEMOCAP-DA特有参数
                'activation': 'tanh',
                'lr_pre': [2e-5],
                'lr': [2e-4],                      # 从5e-4降低到2e-4，提高训练稳定性（诊断建议）
                'delta': [0.01],                   # 从0.02进一步降低到0.01，更慢的阈值增长（针对极端不平衡）
                'thres': [0.03],                   # 从0.05进一步降低到0.03，更保守的初始阈值（针对极端不平衡）
                'topk': [5],
                'weight_decay': 0.01,
                'feat_dim': 768,
                'hidden_size': 768,
                'grad_clip': [-1.0],
                'warmup_proportion': 0.1,
                'hidden_dropout_prob': 0.1,
                'weight': 1.0,
                'loss_mode': 'rdrop',
                'base_dim': [256],                 # 从128增加到256，增强特征表示能力（针对文本信息量不足）
                'nheads': 8,
                'attn_dropout': 0.1,
                'relu_dropout': 0.1,
                'embed_dropout': 0.1,
                'res_dropout': 0.0,
                'attn_mask': True,
                'encoder_layers_1': 1,
                'fusion_act': 'tanh',
                
                # 创新点一：双投影机制
                'enable_dual_projection': ablation_config.get('enable_dual_projection', True),
                'enable_video_dual': ablation_config.get('enable_video_dual', True),
                'enable_audio_dual': ablation_config.get('enable_audio_dual', True),

                # 创新点二：文本引导注意力
                'enable_text_guided_attention': ablation_config.get('enable_text_guided_attention', True),
                'enable_self_attention': ablation_config.get('enable_self_attention', True),
                'self_attention_layers': ablation_config.get('self_attention_layers', 2),

                # 创新点三：聚类/对比学习
                'enable_clustering_optimization': ablation_config.get('enable_clustering_optimization', True),
                'enable_clustering_loss': ablation_config.get('enable_clustering_loss', True),
                'enable_contrastive_loss': ablation_config.get('enable_contrastive_loss', True),
                'enable_compactness_loss': ablation_config.get('enable_compactness_loss', True),
                'enable_separation_loss': ablation_config.get('enable_separation_loss', True),

                # 对比学习/聚类权重
                'contrastive_weight': ablation_config.get('contrastive_weight', 1.0),
                'contrastive_temperature': 0.07,
                'clustering_weight': ablation_config.get('clustering_weight', 2.0),
                'compactness_weight': ablation_config.get('compactness_weight', 1.5),
                'separation_weight': ablation_config.get('separation_weight', 1.5),

                # 渐进式学习优化参数（针对极端类别不平衡，使用更保守的阈值范围）
                'enable_progressive_learning': ablation_config.get('enable_progressive_learning', True),
                'enable_adaptive_threshold': ablation_config.get('enable_adaptive_threshold', True),
                'enable_performance_monitoring': ablation_config.get('enable_performance_monitoring', True),
                'enable_early_stop': ablation_config.get('enable_early_stop', True),
                'max_threshold': 0.3,
                'min_threshold': 0.03,
                'performance_window': 3,
                'patience': 5,

                # 消融实验配置（保留以支持消融实验）
                'ablation_config': ablation_config,
                'fixed_train_epochs': ablation_config.get('fixed_train_epochs', 25),
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters
    
    def _get_ablation_config(self, args):
        """获取消融实验配置"""
        # 默认配置
        default_config = {
            # 创新点一：双投影机制 (ConFEDE)
            'enable_dual_projection': True,
            'enable_video_dual': True,
            'enable_audio_dual': True,
            
            # 创新点二：文本引导注意力和自注意力机制
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            
            # 创新点三：聚类优化架构（包含聚类损失）
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            
            # 渐进式学习
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_performance_monitoring': True,
            'enable_early_stop': True,
            
            # 注意力机制（已移到创新点二）
            
            # 聚类优化
            'enable_clustering_optimization': True,
            
            # 损失权重
            'clustering_weight': 1.0,
            'contrastive_weight': 0.5,
            'compactness_weight': 0.3,
            'separation_weight': 0.2,
            
            # 训练参数
            'fixed_train_epochs': 25,
        }
        
        # 根据消融实验名称调整配置
        if hasattr(args, 'ablation_experiment') and args.ablation_experiment:
            experiment_name = args.ablation_experiment.lower()
            
            if experiment_name == 'full_confede':
                # full_confede: 启用所有ConFEDE相关功能
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                })
            elif experiment_name == 'no_dual_projection':
                # 禁用双投影机制 - 测试第一个创新点的贡献
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    # 保留其他创新点
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                })
            elif experiment_name == 'no_clustering_loss':
                # 禁用聚类损失
                config = default_config.copy()
                config.update({
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'enable_compactness_loss': False,
                    'enable_separation_loss': False,
                })
            elif experiment_name == 'no_text_guided_attention':
                # 禁用创新点二：文本引导注意力和自注意力机制
                # 关闭创新点二，保留创新点一和创新点三
                config = default_config.copy()
                config.update({
                    # 创新点一：保留双投影机制
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,

                    # 创新点二：关闭文本引导注意力和自注意力机制
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,

                    # 创新点三：保留聚类和渐进式学习
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                })
            elif experiment_name == 'no_progressive_learning':
                # 禁用创新点三：聚类优化 + 渐进式学习
                # 关闭创新点三，保留创新点一和创新点二
                config = default_config.copy()
                config.update({
                    # 创新点一：保留双投影机制
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,

                    # 创新点二：保留文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,

                    # 创新点三：关闭聚类和渐进式学习相关组件
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'enable_compactness_loss': False,
                    'enable_separation_loss': False,
                    'enable_clustering_optimization': False,
                    'enable_progressive_learning': False,
                    'enable_adaptive_threshold': False,
                    'enable_performance_monitoring': False,
                    'enable_early_stop': False,
                })
            # ==================== 替代性消融实验 ====================
            
            # 创新点一的替代实验
            elif experiment_name == 'dual_projection_simple_linear':
                # 替代方案A：简单线性映射替代双投影
                config = default_config.copy()
                config.update({
                    'dual_projection_strategy': 'simple_linear',  # 使用简单线性映射
                    'enable_video_dual': True,  # 仍然启用，但使用简单线性映射
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'enable_progressive_learning': True,
                })
            elif experiment_name == 'dual_projection_denoising_ae':
                # 替代方案B：标准自动编码器去噪替代双投影
                config = default_config.copy()
                config.update({
                    'dual_projection_strategy': 'denoising_ae',  # 使用去噪自编码器
                    'enable_video_dual': True,  # 仍然启用，但使用去噪AE
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'enable_progressive_learning': True,
                })
            
            # 创新点二的替代实验
            elif experiment_name == 'fusion_direct_concat':
                # 替代方案A：直接拼接替代文本引导注意力
                config = default_config.copy()
                config.update({
                    'fusion_strategy': 'direct_concat',  # 使用直接拼接
                    'enable_text_guided_attention': False,  # 关闭文本引导注意力
                    'enable_self_attention': False,  # 关闭自注意力
                    'enable_dual_projection': True,  # 保留创新点一
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_progressive_learning': True,  # 保留创新点三
                })
            elif experiment_name == 'fusion_all_to_all':
                # 替代方案B：全模态自注意力替代文本引导注意力
                config = default_config.copy()
                config.update({
                    'fusion_strategy': 'all_to_all',  # 使用全模态自注意力
                    'enable_text_guided_attention': False,  # 关闭文本引导注意力
                    'enable_self_attention': True,  # 保留自注意力（但改为全模态）
                    'enable_dual_projection': True,  # 保留创新点一
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_progressive_learning': True,  # 保留创新点三
                })
            
            # 创新点三的替代实验
            elif experiment_name == 'progressive_linear_scheduling':
                # 替代方案A：线性阈值调度替代S型曲线
                config = default_config.copy()
                config.update({
                    'threshold_scheduling': 'linear',  # 使用线性调度（UMC论文方法）
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_dual_projection': True,  # 保留创新点一
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,  # 保留创新点二
                    'enable_self_attention': True,
                })
            elif experiment_name == 'progressive_hard_threshold':
                # 替代方案B：硬阈值筛选替代软性重加权
                config = default_config.copy()
                config.update({
                    'sample_selection_strategy': 'hard_threshold',  # 使用硬阈值筛选
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_dual_projection': True,  # 保留创新点一
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_text_guided_attention': True,  # 保留创新点二
                    'enable_self_attention': True,
                })
            
            else:
                # 未知的消融实验，使用默认配置
                config = default_config.copy()
        else:
            # 没有指定消融实验，使用默认配置
            config = default_config.copy()
            
        return config
    