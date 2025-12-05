#!/usr/bin/env python3
"""
创新点三：聚类优化的多模态架构设计 - 消融实验运行脚本
"""

import os
import sys
import argparse
import json
import warnings
import datetime
import subprocess
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.umc_innovation3_ablation import Innovation3AblationConfig

def run_innovation3_experiment(experiment_name, dataset='MIntRec', gpu='0', train=True, save_results=True):
    """运行创新点三消融实验"""
    
    print(f"运行创新点三消融实验: {experiment_name}")
    print(f"数据集: {dataset}")
    print(f"GPU: {gpu}")
    print(f"训练: {train}")
    print(f"保存结果: {save_results}")
    
    # 构建命令
    cmd = [
        'python', 'run_single_ablation.py',
        '--experiment_name', experiment_name,
        '--dataset', dataset,
        '--gpu', gpu,
        '--config_file_name', 'umc_MIntRec.py'
    ]
    
    if train:
        cmd.append('--train')
    
    if save_results:
        cmd.append('--save_results')
        cmd.extend(['--output_dir', 'innovation3_results'])
    
    # 运行命令
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"实验成功完成！耗时: {end_time - start_time:.2f}秒")
        print("="*50)
        print("实验输出:")
        print(result.stdout)
        print("="*50)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def run_experiment_group(group_name, dataset='MIntRec', gpu='0', train=True):
    """运行实验组"""
    
    config = Innovation3AblationConfig()
    groups = config.get_experiment_groups()
    
    if group_name not in groups:
        print(f"未知的实验组: {group_name}")
        print(f"可用的实验组: {list(groups.keys())}")
        return False
    
    experiments = groups[group_name]
    print(f"运行实验组: {group_name}")
    print(f"包含实验: {experiments}")
    
    results = {}
    
    for i, experiment_name in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"运行实验 {i}/{len(experiments)}: {experiment_name}")
        print(f"{'='*60}")
        
        success = run_innovation3_experiment(experiment_name, dataset, gpu, train)
        results[experiment_name] = success
        
        if success:
            print(f"✅ 实验 {experiment_name} 成功完成")
        else:
            print(f"❌ 实验 {experiment_name} 失败")
    
    # 保存组结果
    group_result_file = f"innovation3_results/{group_name}_group_results.json"
    os.makedirs("innovation3_results", exist_ok=True)
    
    with open(group_result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'group_name': group_name,
            'dataset': dataset,
            'gpu': gpu,
            'train': train,
            'timestamp': datetime.datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n组实验结果已保存到: {group_result_file}")
    
    # 统计结果
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n组实验完成统计:")
    print(f"成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    
    return success_count == total_count

def run_all_innovation3_experiments(dataset='MIntRec', gpu='0', train=True):
    """运行所有创新点三消融实验"""
    
    config = Innovation3AblationConfig()
    groups = config.get_experiment_groups()
    
    print("运行所有创新点三消融实验")
    print(f"数据集: {dataset}")
    print(f"GPU: {gpu}")
    print(f"训练: {train}")
    
    all_results = {}
    
    for group_name, experiments in groups.items():
        print(f"\n{'='*80}")
        print(f"开始运行实验组: {group_name}")
        print(f"{'='*80}")
        
        group_success = run_experiment_group(group_name, dataset, gpu, train)
        all_results[group_name] = group_success
    
    # 保存所有结果
    all_result_file = "innovation3_results/all_innovation3_results.json"
    os.makedirs("innovation3_results", exist_ok=True)
    
    with open(all_result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': dataset,
            'gpu': gpu,
            'train': train,
            'timestamp': datetime.datetime.now().isoformat(),
            'group_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n所有实验结果已保存到: {all_result_file}")
    
    # 统计结果
    success_groups = sum(all_results.values())
    total_groups = len(all_results)
    print(f"\n总体实验完成统计:")
    print(f"成功组: {success_groups}/{total_groups}")
    print(f"失败组: {total_groups - success_groups}/{total_groups}")
    
    return success_groups == total_groups

def interactive_mode():
    """交互式模式"""
    
    config = Innovation3AblationConfig()
    groups = config.get_experiment_groups()
    
    print("创新点三：聚类优化的多模态架构设计 - 消融实验")
    print("="*60)
    
    # 显示实验组
    print("可用的实验组:")
    for i, (group_name, experiments) in enumerate(groups.items(), 1):
        print(f"{i:2d}. {group_name} ({len(experiments)}个实验)")
        for exp in experiments[:3]:  # 只显示前3个
            print(f"    - {exp}")
        if len(experiments) > 3:
            print(f"    - ... 还有{len(experiments)-3}个实验")
        print()
    
    # 显示所有实验
    all_experiments = []
    for experiments in groups.values():
        all_experiments.extend(experiments)
    
    print(f"所有实验 ({len(all_experiments)}个):")
    for i, exp in enumerate(all_experiments, 1):
        print(f"{i:2d}. {exp}")
    
    try:
        choice = input(f"\n请选择运行模式:\n1. 运行单个实验\n2. 运行实验组\n3. 运行所有实验\n选择 (1-3): ").strip()
        
        if choice == '1':
            # 单个实验
            exp_choice = input(f"请选择实验编号 (1-{len(all_experiments)}): ")
            exp_index = int(exp_choice) - 1
            
            if 0 <= exp_index < len(all_experiments):
                experiment_name = all_experiments[exp_index]
            else:
                print("无效的选择！")
                return
            
        elif choice == '2':
            # 实验组
            group_choice = input(f"请选择实验组编号 (1-{len(groups)}): ")
            group_index = int(group_choice) - 1
            
            if 0 <= group_index < len(groups):
                group_name = list(groups.keys())[group_index]
            else:
                print("无效的选择！")
                return
            
        elif choice == '3':
            # 所有实验
            group_name = 'all'
            
        else:
            print("无效的选择！")
            return
        
        # 获取其他参数
        dataset = input("数据集 (默认: MIntRec): ").strip() or 'MIntRec'
        gpu = input("GPU设备 (默认: 0): ").strip() or '0'
        train_choice = input("是否训练模型? (y/n, 默认: y): ").strip().lower()
        train = train_choice != 'n'
        
        # 运行实验
        if choice == '1':
            success = run_innovation3_experiment(experiment_name, dataset, gpu, train)
            if success:
                print(f"\n✅ 实验 {experiment_name} 成功完成！")
            else:
                print(f"\n❌ 实验 {experiment_name} 失败！")
                
        elif choice == '2':
            success = run_experiment_group(group_name, dataset, gpu, train)
            if success:
                print(f"\n✅ 实验组 {group_name} 成功完成！")
            else:
                print(f"\n❌ 实验组 {group_name} 部分失败！")
                
        elif choice == '3':
            success = run_all_innovation3_experiments(dataset, gpu, train)
            if success:
                print(f"\n✅ 所有创新点三消融实验成功完成！")
            else:
                print(f"\n❌ 部分创新点三消融实验失败！")
                
    except KeyboardInterrupt:
        print("\n用户取消实验")
    except Exception as e:
        print(f"发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='创新点三消融实验运行脚本')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['single', 'group', 'all', 'interactive'], 
                       default='interactive', help='运行模式')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, help='实验名称（single模式）')
    parser.add_argument('--group_name', type=str, help='实验组名称（group模式）')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='MIntRec', help='数据集名称')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备')
    parser.add_argument('--train', action='store_true', help='是否训练模型')
    parser.add_argument('--no_save_results', action='store_true', help='不保存结果')
    
    args = parser.parse_args()
    
    # 设置警告过滤
    warnings.filterwarnings('ignore')
    
    # 根据模式运行
    if args.mode == 'single':
        if not args.experiment_name:
            print("single模式需要指定--experiment_name")
            return
        
        success = run_innovation3_experiment(
            args.experiment_name, 
            args.dataset, 
            args.gpu, 
            args.train,
            not args.no_save_results
        )
        
    elif args.mode == 'group':
        if not args.group_name:
            print("group模式需要指定--group_name")
            return
        
        success = run_experiment_group(
            args.group_name,
            args.dataset,
            args.gpu,
            args.train
        )
        
    elif args.mode == 'all':
        success = run_all_innovation3_experiments(
            args.dataset,
            args.gpu,
            args.train
        )
        
    elif args.mode == 'interactive':
        interactive_mode()
        return
    
    if args.mode != 'interactive':
        if success:
            print(f"\n✅ {args.mode}模式实验成功完成！")
        else:
            print(f"\n❌ {args.mode}模式实验失败！")
            sys.exit(1)

if __name__ == '__main__':
    main()
