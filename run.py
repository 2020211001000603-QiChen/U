from configs.base import ParamManager, add_config_param
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed, save_results, set_output_path

import argparse
import logging
import os
import datetime
import itertools
import warnings
import copy

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logger_name', type=str, default='Multimodal Semantics Discovery', help="Logger name for multimodal semantics discovery.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The selected dataset.")

    parser.add_argument('--multimodal_method', type=str, default='umc', help="which method to use.")

    parser.add_argument('--method', type=str, default='umc', help="which method to use.")

    parser.add_argument("--text_backbone", type=str, default='bert', help="which backbone to use for text modality")

    parser.add_argument('--seed', type=int, default=0, help="The selected person id.")

    parser.add_argument('--num_workers', type=int, default=16, help="The number of workers to load data.")
    
    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")
    
    parser.add_argument('--gpu_id', type=str, default='0', help="The selected person id.")

    parser.add_argument("--data_path", default = '', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument("--train", action="store_true", help="Whether to train the model.")

    parser.add_argument("--tune", action="store_true", help="Whether to tune the model with a series of hyper-parameters.")
    
    parser.add_argument("--save_model", action="store_true", help="save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true", help="save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")
    
    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")   

    parser.add_argument('--video_data_path', type=str, default='video_data', help="The directory of the video data.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data.")

    parser.add_argument('--video_feats_path', type=str, default='video_feats.pkl', help="The directory of the video features.")
    
    parser.add_argument('--audio_feats_path', type=str, default='audio_feats.pkl', help="The directory of the audio features.")

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", default='', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_path", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--config_file_name", type=str, default='umc_meld.py', help = "The name of the config file.")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")
    
    # 消融实验参数
    parser.add_argument("--ablation_experiment", type=str, default=None, 
                       help="消融实验名称，可选: baseline_traditional, confede_only, no_dual_projection, no_clustering_loss, no_progressive_learning, text_guided_only, gated_fusion_only, full_confede, fixed_strategy, progressive_only, enhanced_loss_only, full_progressive, standard_architecture, clustering_projector_only, full_optimized_architecture, only_confede, only_progressive, only_architecture, full_umc_model")
    
    parser.add_argument("--run_all_ablation", action="store_true", 
                       help="运行所有消融实验")
    
    args = parser.parse_args()

    return args

def set_logger(args):
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name =  f"{args.method}_{args.multimodal_method}_{args.text_backbone}_{args.dataset}_{args.seed}"
    args.log_id = f"{args.logger_name}_{time}"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

def apply_ablation_config(args, experiment_name):
    """应用消融实验配置"""
    
    # 将实验名称转换为小写以确保匹配
    experiment_name = experiment_name.lower() if experiment_name else None
    
    ablation_configs = {
        # ConFEDE消融实验
        'baseline_traditional': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'confede_only': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'no_dual_projection': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'enable_gated_fusion': False,  # 关闭门控融合
            'enable_cross_attention': False,  # 关闭多层Cross-Attention
            'enable_feature_interaction': False,  # 关闭特征交互层
            'enable_contrastive_loss': False,  # 关闭对比学习损失
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_clustering_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        'only_innovation1': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'enable_adaptive_threshold': False,
            'enable_clustering_loss': False,
            'enable_contrastive_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'only_innovation2': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'enable_adaptive_threshold': False,
            'enable_clustering_loss': False,
            'enable_contrastive_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'only_innovation3': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        'no_dual_projection_strict_ablation': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'enable_gated_fusion': False,
            'enable_cross_attention': False,
            'enable_feature_interaction': False,
            'enable_contrastive_loss': False,  # 关闭对比学习损失
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_clustering_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        'no_text_guided_attention': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        'no_clustering_loss': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': False,
            'enable_clustering_loss': False,
            'enable_progressive_learning': True,
            'clustering_weight': 0.0,
            'contrastive_weight': 0.0,
            'thres': [0.05],
            'delta': [0.02],
        },
        'no_progressive_learning': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'enable_gated_fusion': True,
            # 关闭创新点三：聚类优化和渐进式学习
            'enable_clustering_optimization': False,
            'enable_clustering_loss': False,
            'enable_contrastive_loss': False,
            'enable_compactness_loss': False,
            'enable_separation_loss': False,
            'enable_progressive_learning': False,
            'enable_adaptive_threshold': False,
            'enable_performance_monitoring': False,
            'enable_early_stop': False,
            'thres': [0.25],  # 固定阈值
            'delta': [0.0],   # 固定步长
        },
        'text_guided_only': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': True,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'gated_fusion_only': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'full_confede': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        # ==================== 创新点三：聚类优化架构消融实验 ====================
        
        'baseline_no_clustering': {
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
            'enable_clustering_loss': False,
            'enable_contrastive_loss': False,
            'enable_compactness_loss': False,
            'enable_separation_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_projector_only': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': True,
            'use_clustering_fusion': False,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': False,
            'enable_compactness_loss': False,
            'enable_separation_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_fusion_only': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': False,
            'use_clustering_fusion': True,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': False,
            'enable_compactness_loss': False,
            'enable_separation_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_loss_only': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': False,
            'use_clustering_fusion': False,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_projector_fusion': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': True,
            'use_clustering_fusion': True,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': False,
            'enable_compactness_loss': False,
            'enable_separation_loss': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_projector_loss': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': True,
            'use_clustering_fusion': False,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'clustering_fusion_loss': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': False,
            'use_clustering_fusion': True,
            'use_attention_pooling': False,
            'use_layer_norm': False,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        'full_clustering_optimization': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': False,
            'use_clustering_projector': True,
            'use_clustering_fusion': True,
            'use_attention_pooling': True,
            'use_layer_norm': True,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.25],
            'delta': [0.0],
        },
        
        # 渐进策略消融实验
        'fixed_strategy': {
            'enable_video_dual': False,
            'enable_audio_dual': False,
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'enable_enhanced_loss': False,
            'enable_kmeans_plus': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'progressive_only': {
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
        },
        'enhanced_loss_only': {
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
            'thres': [0.25],
            'delta': [0.0],
        },
        'full_progressive': {
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
        },
        
        # 架构消融实验
        'standard_architecture': {
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
            'thres': [0.25],
            'delta': [0.0],
        },
        'clustering_projector_only': {
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
            'thres': [0.25],
            'delta': [0.0],
        },
        'full_optimized_architecture': {
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
            'thres': [0.25],
            'delta': [0.0],
        },
        
        # 协同效果实验
        'only_confede': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': False,
            'enable_progressive_learning': False,
            'thres': [0.25],
            'delta': [0.0],
        },
        'only_progressive': {
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
        },
        'only_architecture': {
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
            'thres': [0.25],
            'delta': [0.0],
        },
        'full_umc_model': {
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
        },
        
        # ==================== 替代性消融实验 ====================
        
        # 创新点一替代实验：简单线性映射替代双投影机制
        # 关闭双投影，使用简单线性映射作为替代方案
        'dual_projection_simple_linear': {
            'enable_video_dual': False,  # 关闭双投影
            'enable_audio_dual': False,  # 关闭双投影
            'use_simple_linear_projection': True,  # 使用简单线性映射替代
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        
        # 创新点二替代实验：直接拼接替代文本引导注意力
        'fusion_direct_concat': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'fusion_strategy': 'direct_concat',  # 使用直接拼接
            'enable_text_guided_attention': False,
            'enable_self_attention': False,
            'enable_gated_fusion': False,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
        
        # 创新点三替代实验：线性阈值调度替代S型曲线
        'progressive_linear_scheduling': {
            'enable_video_dual': True,
            'enable_audio_dual': True,
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            'enable_gated_fusion': True,
            'enable_clustering_optimization': True,
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'threshold_scheduling': 'linear',  # 使用线性调度
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            'thres': [0.05],
            'delta': [0.02],
        },
    }
    
    if experiment_name in ablation_configs:
        config = ablation_configs[experiment_name]
        for key, value in config.items():
            setattr(args, key, value)
        print(f"已应用消融实验配置: {experiment_name}")
        return True
    else:
        print(f"未知的消融实验: {experiment_name}")
        return False

def set_up(args):

    save_model_name = f"{args.method}_{args.multimodal_method}_{args.dataset}_{args.text_backbone}_{args.seed}"
    
    # 如果是消融实验，修改保存名称
    if hasattr(args, 'ablation_experiment') and args.ablation_experiment:
        save_model_name = f"ablation_{args.ablation_experiment}_{args.dataset}_{args.seed}"
    
    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)
    
    set_torch_seed(args.seed)
    
    return args
    
def work(args, data, logger, debug_args=None):
    
    set_torch_seed(args.seed)
    
    method_manager = method_map[args.method]
    model = ModelManager(args)
    method = method_manager(args, data, model)
    
    logger.info('Multimodal Intent Recognition begins...')

    if args.train:

        logger.info('Training begins...')
        method._train(args)
        logger.info('Training is finished...')

    logger.info('Testing begins...')
    outputs = method._test(args)
    logger.info('Testing is finished...')
    logger.info('Multimodal intent recognition is finished...')
    
    if args.save_results:
        
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)
        
def run(args, data, logger, ind_args = None):
    debug_args = {}

    for k,v in args.items():
        if isinstance(v, list):
            debug_args[k] = v
        
    for result in itertools.product(*debug_args.values()):
        for i, key in enumerate(debug_args.keys()):
            args[key] = result[i]         
        
        work(args, data, logger, debug_args)

def run_all_ablation_experiments(args):
    """运行所有消融实验"""
    
    ablation_experiments = [
        'baseline_traditional', 'confede_only', 'text_guided_only', 'gated_fusion_only', 'full_confede',
        'fixed_strategy', 'progressive_only', 'enhanced_loss_only', 'full_progressive',
        'standard_architecture', 'clustering_projector_only', 'full_optimized_architecture',
        'only_confede', 'only_progressive', 'only_architecture', 'full_umc_model'
    ]
    
    print(f"开始运行所有消融实验，共 {len(ablation_experiments)} 个实验")
    
    results = {}
    
    for i, experiment_name in enumerate(ablation_experiments, 1):
        print(f"\n{'='*50}")
        print(f"运行实验 {i}/{len(ablation_experiments)}: {experiment_name}")
        print(f"{'='*50}")
        
        # 创建实验特定的args副本
        exp_args = copy.deepcopy(args)
        exp_args.ablation_experiment = experiment_name
        
        # 应用消融实验配置
        if apply_ablation_config(exp_args, experiment_name):
            try:
                # 初始化参数管理器
                param = ParamManager(exp_args)
                exp_args = param.args
                
                # 初始化数据管理器
                data = DataManager(exp_args)
                logger = set_logger(exp_args)
                
                # 添加配置文件参数
                exp_args = add_config_param(exp_args, exp_args.config_file_name)
                exp_args = set_up(exp_args)
                
                # 记录实验参数
                logger.info("="*30+" 消融实验参数 "+"="*30)
                logger.info(f"实验名称: {experiment_name}")
                logger.info(f"数据集: {exp_args.dataset}")
                logger.info(f"方法: {exp_args.method}")
                logger.info(f"随机种子: {exp_args.seed}")
                
                # 记录关键消融参数
                ablation_keys = [
                    'enable_video_dual', 'enable_audio_dual', 'enable_text_guided_attention',
                    'enable_self_attention', 'enable_gated_fusion', 'enable_clustering_optimization',
                    'enable_progressive_learning', 'use_clustering_projector', 'use_clustering_fusion',
                    'use_attention_pooling', 'use_layer_norm'
                ]
                
                logger.info("消融实验关键参数:")
                for key in ablation_keys:
                    if hasattr(exp_args, key):
                        logger.info(f"  {key}: {getattr(exp_args, key)}")
                
                logger.info("="*30+" 参数记录完成 "+"="*30)
                
                # 执行训练和测试
                work(exp_args, data, logger)
                
                results[experiment_name] = 'success'
                print(f"✅ 实验 {experiment_name} 成功完成")
                
            except Exception as e:
                print(f"❌ 实验 {experiment_name} 失败: {str(e)}")
                results[experiment_name] = f'failed: {str(e)}'
        else:
            print(f"❌ 实验 {experiment_name} 配置失败")
            results[experiment_name] = 'config_failed'
    
    # 打印总结
    print(f"\n{'='*50}")
    print("消融实验总结")
    print(f"{'='*50}")
    
    success_count = sum(1 for result in results.values() if result == 'success')
    total_count = len(results)
    
    print(f"总实验数: {total_count}")
    print(f"成功实验数: {success_count}")
    print(f"失败实验数: {total_count - success_count}")
    
    print(f"\n详细结果:")
    for exp_name, result in results.items():
        status = "✅" if result == 'success' else "❌"
        print(f"{status} {exp_name}: {result}")

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # 检查是否运行所有消融实验
    if args.run_all_ablation:
        run_all_ablation_experiments(args)
    else:
        # 检查是否运行单个消融实验
        if args.ablation_experiment:
            print(f"运行消融实验: {args.ablation_experiment}")
            
            # 应用消融实验配置（如果不在run.py的字典中，尝试使用配置文件中的配置）
            experiment_found = apply_ablation_config(args, args.ablation_experiment)
            
            # 如果apply_ablation_config返回False，仍然继续，让配置文件处理
            if not experiment_found:
                print(f"注意: 实验 {args.ablation_experiment} 不在run.py的配置中，将使用配置文件中的配置")
            
            param = ParamManager(args)
            args = param.args

            data = DataManager(args)
            logger = set_logger(args)
            
            args = add_config_param(args, args.config_file_name)
            args = set_up(args)
            
            logger.info("="*30+" 消融实验参数 "+"="*30)
            logger.info(f"实验名称: {args.ablation_experiment}")
            logger.info(f"数据集: {args.dataset}")
            logger.info(f"方法: {args.method}")
            logger.info(f"随机种子: {args.seed}")
            
            # 记录关键消融参数
            ablation_keys = [
                'enable_video_dual', 'enable_audio_dual', 'enable_text_guided_attention',
                'enable_self_attention', 'enable_gated_fusion', 'enable_clustering_optimization',
                'enable_progressive_learning', 'use_clustering_projector', 'use_clustering_fusion',
                'use_attention_pooling', 'use_layer_norm'
            ]
            
            logger.info("消融实验关键参数:")
            for key in ablation_keys:
                if hasattr(args, key):
                    logger.info(f"  {key}: {getattr(args, key)}")
            
            logger.info("="*30+" 参数记录完成 "+"="*30)
            
            run(args, data, logger)
            
            print(f"消融实验完成: {args.ablation_experiment}")
        else:
            # 正常运行
            param = ParamManager(args)
            args = param.args

            data = DataManager(args)
            logger = set_logger(args)
            
            args = add_config_param(args, args.config_file_name)
            args = set_up(args)
            
            logger.info("="*30+" Params "+"="*30)
            for k in args.keys():
                logger.info(f"{k}: {args[k]}")
            logger.info("="*30+" End Params "+"="*30)
            
            run(args, data, logger)
    

