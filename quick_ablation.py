#!/usr/bin/env python3
"""
快速运行UMC消融实验的简化脚本
"""

import os
import sys
import subprocess

def run_ablation_experiment(experiment_name, dataset='MIntRec', gpu='0', train=True):
    """运行消融实验"""
    
    print(f"运行消融实验: {experiment_name}")
    print(f"数据集: {dataset}")
    print(f"GPU: {gpu}")
    print(f"训练: {train}")
    
    # 构建命令
    cmd = [
        'python', 'run_single_ablation.py',
        '--experiment_name', experiment_name,
        '--dataset', dataset,
        '--gpu', gpu,
        '--save_results',
        '--config_file_name', 'umc_MIntRec.py'
    ]
    
    if train:
        cmd.append('--train')
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("实验成功完成！")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    """主函数"""
    
    # 可用的实验名称
    available_experiments = [
        'baseline_traditional',
        'confede_only', 
        'text_guided_only',
        'gated_fusion_only',
        'full_confede',
        'fixed_strategy',
        'progressive_only',
        'enhanced_loss_only',
        'full_progressive',
        'standard_architecture',
        'clustering_projector_only',
        'full_optimized_architecture',
        'only_confede',
        'only_progressive',
        'only_architecture',
        'full_umc_model'
    ]
    
    print("可用的消融实验:")
    for i, exp in enumerate(available_experiments, 1):
        print(f"{i:2d}. {exp}")
    
    # 获取用户输入
    try:
        choice = input("\n请选择实验编号 (1-16): ")
        exp_index = int(choice) - 1
        
        if 0 <= exp_index < len(available_experiments):
            experiment_name = available_experiments[exp_index]
        else:
            print("无效的选择！")
            return
        
        # 获取其他参数
        dataset = input("数据集 (默认: MIntRec): ").strip() or 'MIntRec'
        gpu = input("GPU设备 (默认: 0): ").strip() or '0'
        train_choice = input("是否训练模型? (y/n, 默认: y): ").strip().lower()
        train = train_choice != 'n'
        
        # 运行实验
        success = run_ablation_experiment(experiment_name, dataset, gpu, train)
        
        if success:
            print(f"\n✅ 实验 {experiment_name} 成功完成！")
        else:
            print(f"\n❌ 实验 {experiment_name} 失败！")
            
    except KeyboardInterrupt:
        print("\n用户取消实验")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()
