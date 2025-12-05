#!/usr/bin/env python3
"""
列出所有可用的消融实验
"""

import re
import os

def list_experiments_from_config():
    """从配置文件中提取所有消融实验"""
    
    config_file = 'configs/umc_MIntRec.py'
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return []
    
    experiments = []
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 查找所有 elif experiment_name == '...' 的模式
        pattern = r"elif\s+experiment_name\s*==\s*['\"]([^'\"]+)['\"]"
        matches = re.findall(pattern, content)
        
        for match in matches:
            experiments.append(match)
    
    return experiments

def print_experiment_info(experiment_name):
    """打印实验的详细信息"""
    
    config_file = 'configs/umc_MIntRec.py'
    
    if not os.path.exists(config_file):
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找实验配置的开始和结束
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if f"elif experiment_name == '{experiment_name}'" in line or \
           f'elif experiment_name == "{experiment_name}"' in line:
            start_line = i
        elif start_line is not None and (line.strip().startswith('elif') or 
                                          line.strip().startswith('else:')):
            end_line = i
            break
    
    if start_line is not None:
        print(f"\n实验配置 (第 {start_line+1} 行):")
        print("-" * 60)
        # 打印配置代码（最多20行）
        for i in range(start_line, min(start_line + 20, len(lines))):
            if end_line and i >= end_line:
                break
            print(f"{i+1:4d} | {lines[i].rstrip()}")
        print("-" * 60)

def main():
    print("=" * 60)
    print("UMC 消融实验列表")
    print("=" * 60)
    
    # 从配置文件提取实验
    experiments = list_experiments_from_config()
    
    if not experiments:
        print("\n❌ 未找到任何消融实验配置")
        print("\n提示：")
        print("1. 检查配置文件是否存在: configs/umc_MIntRec.py")
        print("2. 确认配置文件中是否有消融实验定义")
        print("3. 参考 MIntRec消融实验修改指南.md 添加实验")
        return
    
    print(f"\n✅ 找到 {len(experiments)} 个已定义的消融实验：\n")
    
    # 打印实验列表
    for i, exp in enumerate(experiments, 1):
        print(f"{i:2d}. {exp}")
    
    # 打印实验说明
    print("\n" + "=" * 60)
    print("实验说明")
    print("=" * 60)
    
    experiment_descriptions = {
        'no_dual_projection': '禁用双投影机制（创新点一）',
        'no_clustering_loss': '禁用聚类损失（创新点三部分）',
        'no_progressive_learning': '禁用渐进式学习（创新点三）',
        'full_confede': '完整ConFEDE机制',
    }
    
    for exp in experiments:
        desc = experiment_descriptions.get(exp, '未提供说明')
        print(f"  {exp:30s} - {desc}")
    
    # 交互式查看详细配置
    print("\n" + "=" * 60)
    print("查看详细配置")
    print("=" * 60)
    
    while True:
        print("\n输入实验名称查看详细配置（或输入 'q' 退出）：")
        exp_name = input("> ").strip()
        
        if exp_name.lower() == 'q':
            break
        
        if exp_name in experiments:
            print_experiment_info(exp_name)
        else:
            print(f"❌ 未找到实验: {exp_name}")
            print(f"可用实验: {', '.join(experiments)}")
    
    print("\n" + "=" * 60)
    print("运行示例")
    print("=" * 60)
    print("\n运行单个消融实验：")
    print(f"  python run.py --dataset MIntRec --ablation_experiment {experiments[0] if experiments else '实验名称'} --train")
    print("\n运行多次取平均：")
    print("  for seed in 0 1 2 3 4; do")
    print(f"    python run.py --dataset MIntRec --ablation_experiment {experiments[0] if experiments else '实验名称'} --seed $seed --train")
    print("  done")

if __name__ == '__main__':
    main()








