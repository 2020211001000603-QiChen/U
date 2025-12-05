#!/usr/bin/env python3
"""
运行单个UMC消融实验
"""

import os
import sys
import argparse
import json
import warnings
import datetime
import itertools
import copy
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.base import ParamManager, add_config_param
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed, save_results, set_output_path
from configs.umc_ablation_param import UMCAblationParam

def set_logger(args):
    """设置日志"""
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name = f"ablation_{args.experiment_name}_{args.dataset}_{args.seed}"
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

def set_up(args):
    """设置输出路径和随机种子"""
    # 兼容原 run.py：如果没有 output_path，则使用 output_dir 作为根输出目录
    if not hasattr(args, 'output_path') or not args.output_path:
        # output_dir 在本脚本的 argparse 中默认存在
        args.output_path = getattr(args, 'output_dir', 'ablation_results')

    save_model_name = f"ablation_{args.experiment_name}_{args.dataset}_{args.seed}"
    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)
    set_torch_seed(args.seed)
    return args

def _flatten_singleton_lists(args):
    """
    将配置中的单元素 list/tuple 展开为标量，避免后续数值计算中出现 list 与 float/int 运算错误。
    例如：['0.05'] -> 0.05, [True] -> True。
    """
    for k, v in vars(args).items():
        if isinstance(v, (list, tuple)) and len(v) == 1:
            setattr(args, k, v[0])
    return args

def work(args, data, logger):
    """执行训练和测试"""
    set_torch_seed(args.seed)
    
    method_manager = method_map[args.method]
    model = ModelManager(args)
    method = method_manager(args, data, model)
    
    logger.info(f'消融实验 {args.experiment_name} 开始...')

    if args.train:
        logger.info('训练开始...')
        method._train(args)
        logger.info('训练完成...')

    logger.info('测试开始...')
    outputs = method._test(args)
    logger.info('测试完成...')
    logger.info(f'消融实验 {args.experiment_name} 完成...')
    
    if args.save_results:
        logger.info('结果已保存')
        save_results(args, outputs)
    
    return outputs

def run_single_ablation_experiment(experiment_name, args):
    """运行单个消融实验"""
    
    print(f"开始运行消融实验: {experiment_name}")
    print(f"数据集: {args.dataset}")
    print(f"GPU: {args.gpu}")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 创建实验参数
    ablation_param = UMCAblationParam(experiment_name, args)
    
    # 打印实验配置
    print(f"\n实验配置:")
    print(f"实验名称: {experiment_name}")
    
    # 初始化参数管理器
    param = ParamManager(args)
    args = param.args

    # 添加消融实验配置
    for key, value in ablation_param.hyper_param.items():
        args[key] = value

    # 设置数据集路径
    args.data_path = args.data_path or f"data/{args.dataset}"

    # 确保多模态特征路径参数存在（与 examples/run_umc.sh 中保持一致）
    if not hasattr(args, 'video_data_path'):
        args.video_data_path = 'video_data'
    if not hasattr(args, 'video_feats_path'):
        args.video_feats_path = 'swin_feats.pkl'
    if not hasattr(args, 'audio_data_path'):
        args.audio_data_path = 'audio_data'
    if not hasattr(args, 'audio_feats_path'):
        args.audio_feats_path = 'wavlm_feats.pkl'

    # 先创建日志器，确保 args.logger_name 已设置，DataManager 可以安全使用
    logger = set_logger(args)

    # 初始化数据管理器（依赖 logger_name 以及上述路径参数）
    data = DataManager(args)
    
    # 添加配置文件参数
    # 将 experiment_name 传递给配置解析，便于 _get_ablation_config 命中对应实验
    args.ablation_experiment = experiment_name
    args = add_config_param(args, args.config_file_name)
    # 将单元素 list/tuple 展开为标量，避免后续数值计算出错
    args = _flatten_singleton_lists(args)
    args = set_up(args)
    
    # 记录实验参数
    logger.info("="*30+" 消融实验参数 "+"="*30)
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"方法: {args.method}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"base_dim: {getattr(args, 'base_dim', 'n/a')}")
    logger.info(f"dual_projection_strategy: {getattr(args, 'dual_projection_strategy', 'n/a')}")
    logger.info(f"use_simple_linear_projection: {getattr(args, 'use_simple_linear_projection', 'n/a')}")
    logger.info(f"enable_dual_projection: {getattr(args, 'enable_dual_projection', 'n/a')}")
    
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
    
    # 执行训练和测试
    try:
        outputs = work(args, data, logger)
        
        # 提取结果
        result = {
            'experiment_name': experiment_name,
            'dataset': args.dataset,
            'config': ablation_param.hyper_param,
            'metrics': {
                'nmi': outputs.get('nmi', 0.0),
                'ari': outputs.get('ari', 0.0),
                'acc': outputs.get('acc', 0.0),
                'f1': outputs.get('f1', 0.0),
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        print(f"\n实验结果:")
        print(f"NMI: {result['metrics']['nmi']:.4f}")
        print(f"ARI: {result['metrics']['ari']:.4f}")
        print(f"ACC: {result['metrics']['acc']:.4f}")
        print(f"F1: {result['metrics']['f1']:.4f}")
        
        # 保存结果
        if args.save_results:
            save_experiment_result(result, args.output_dir)
        
        return result
        
    except Exception as e:
        print(f"实验失败: {str(e)}")
        logger.error(f"实验失败: {str(e)}")
        return {
            'experiment_name': experiment_name,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }

def save_experiment_result(result, output_dir):
    """保存实验结果"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    result_file = os.path.join(output_dir, f"{result['experiment_name']}_result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {result_file}")

def main():
    parser = argparse.ArgumentParser(description='运行单个UMC消融实验')
    
    # 消融实验参数
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='实验名称')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='MIntRec',
                       help='数据集名称')
    
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU设备')
    
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')
    
    parser.add_argument('--method', type=str, default='umc',
                       help='方法名称')
    
    parser.add_argument('--multimodal_method', type=str, default='umc',
                       help='多模态方法')
    
    parser.add_argument('--text_backbone', type=str, default='bert',
                       help='文本骨干网络')
    
    # 训练参数
    parser.add_argument('--train', action='store_true',
                       help='是否训练模型')
    
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存实验结果')
    
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    
    # 路径参数
    parser.add_argument('--data_path', type=str, default='',
                       help='数据路径')
    
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='结果保存目录')
    
    parser.add_argument('--log_path', type=str, default='logs',
                       help='日志路径')
    
    parser.add_argument('--results_path', type=str, default='results',
                       help='结果路径')
    
    parser.add_argument('--model_path', type=str, default='models',
                       help='模型路径')
    
    parser.add_argument('--config_file_name', type=str, default='umc_MIntRec.py',
                       help='配置文件名称')
    
    parser.add_argument('--results_file_name', type=str, default='results.csv',
                       help='结果文件名')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=16,
                       help='数据加载工作进程数')
    
    args = parser.parse_args()
    
    # 设置警告过滤
    warnings.filterwarnings('ignore')
    
    # 运行实验
    result = run_single_ablation_experiment(args.experiment_name, args)
    
    print(f"\n实验完成！")
    
    if 'error' in result:
        print(f"实验失败: {result['error']}")
        sys.exit(1)
    else:
        print(f"实验成功完成！")

if __name__ == '__main__':
    main()
