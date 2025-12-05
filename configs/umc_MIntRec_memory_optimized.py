# UMC内存优化配置文件

class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        内存优化版本的UMC配置
        """
        if args.multimodal_method == 'umc':
            hyper_parameters = {
                'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
                
                # 🔧 内存优化：减小批次大小
                'pretrain_batch_size': 32,      # 从128减少到32
                'train_batch_size': 32,         # 从128减少到32
                'eval_batch_size': 16,          # 从128减少到16
                'test_batch_size': 16,          # 从128减少到16
                
                # 🔧 训练轮数保持不变
                'num_pretrain_epochs': 100,
                'num_train_epochs': 100,
                'pretrain': [True],
                
                # 🔧 基础配置
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
                'delta': [0.05],
                'thres': [0.1],
                'topk': [5],
                'weight_decay': 0.01,
                'feat_dim': 768,
                'hidden_size': 768,
                'grad_clip': -1.0,
                'warmup_proportion': 0.1,
                'hidden_dropout_prob': 0.1,
                'weight': 1.0,
                'loss_mode': 'rdrop',
                
                # 🔧 模型架构优化：减小模型复杂度
                'base_dim': 128,                # 从256减少到128
                'nheads': 4,                    # 从8减少到4
                'attn_dropout': 0.1,
                'relu_dropout': 0.1,
                'embed_dropout': 0.1,
                'res_dropout': 0.0,
                'attn_mask': True,
                'encoder_layers_1': 1,
                'fusion_act': 'tanh',
                
                # 🔧 功能开关：选择性启用功能以减少内存使用
                'enable_video_dual': False,     # 禁用视频双投影以节省内存
                'enable_audio_dual': False,     # 禁用音频双投影以节省内存
                'enable_text_guided_attention': True,  # 保留文本引导注意力
                'enable_self_attention': False,         # 禁用自注意力层
                'self_attention_layers': 1,             # 减少自注意力层数
                
                # 🔧 对比学习参数优化
                'contrastive_weight': 0.05,     # 减少对比学习权重
                'contrastive_temperature': 0.07,
                'enable_contrastive': True,
                
                # 🔧 聚类损失参数优化
                'enable_clustering_optimization': False,  # 禁用聚类优化以节省内存
                'clustering_weight': 0.05,               # 减少聚类损失权重
                'clustering_feature_weight': 0.05,
                'compactness_weight': 0.5,               # 减少紧密度权重
                'separation_weight': 0.5,                # 减少分离度权重
                'enable_clustering_loss': False,         # 禁用聚类损失
                
                # 🔧 渐进式学习参数
                'max_threshold': 0.3,                   # 减少最大阈值
                'min_threshold': 0.05,
                'performance_window': 3,
                'patience': 5,
                'enable_early_stop': True,
                'enable_adaptive_threshold': True,
                
                # 🔧 新增内存优化参数
                'gradient_accumulation_steps': 4,       # 梯度累积步数
                'max_grad_norm': 1.0,                   # 梯度裁剪
                'use_mixed_precision': True,            # 混合精度训练
                'pin_memory': False,                    # 禁用pin_memory
                'num_workers': 2,                       # 减少数据加载进程数
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters
