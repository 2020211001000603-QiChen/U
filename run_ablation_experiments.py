#!/usr/bin/env python3
"""
UMC消融实验运行脚本
基于3个主要创新点的消融实验
"""

import os
import sys
import argparse
import json
from datetime import datetime
from configs.umc_ablation_experiments import UMCAblationConfig

def run_ablation_experiments(args):
    """运行消融实验"""
    
    # 初始化消融实验配置
    ablation_config = UMCAblationConfig()
    
    # 获取实验分组
    experiment_groups = ablation_config.get_experiment_groups()
    
    # 选择要运行的实验组
    if args.experiment_group == 'all':
        groups_to_run = list(experiment_groups.keys())
    else:
        groups_to_run = [args.experiment_group]
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ablation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"开始运行消融实验...")
    print(f"结果将保存到: {results_dir}")
    print(f"实验组: {groups_to_run}")
    
    all_results = {}
    
    for group_name in groups_to_run:
        print(f"\n{'='*50}")
        print(f"运行实验组: {group_name}")
        print(f"{'='*50}")
        
        group_experiments = experiment_groups[group_name]
        group_results = {}
        
        for exp_name in group_experiments:
            print(f"\n运行实验: {exp_name}")
            
            # 获取实验配置
            experiment = ablation_config.get_experiment_config(exp_name)
            print(f"实验名称: {experiment['name']}")
            print(f"实验描述: {experiment['description']}")
            
            # 运行实验
            try:
                result = run_single_experiment(exp_name, experiment['config'], args)
                group_results[exp_name] = result
                print(f"实验 {exp_name} 完成")
                
            except Exception as e:
                print(f"实验 {exp_name} 失败: {str(e)}")
                group_results[exp_name] = {'error': str(e)}
        
        all_results[group_name] = group_results
        
        # 保存组结果
        group_result_file = os.path.join(results_dir, f"{group_name}_results.json")
        with open(group_result_file, 'w', encoding='utf-8') as f:
            json.dump(group_results, f, indent=2, ensure_ascii=False)
    
    # 保存所有结果
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成结果摘要
    generate_summary(all_results, results_dir)
    
    print(f"\n所有实验完成！结果保存在: {results_dir}")

def run_single_experiment(exp_name, config, args):
    """运行单个实验"""
    
    # 这里需要根据你的实际训练代码来修改
    # 假设你有一个训练函数 train_model(config, args)
    
    # 模拟实验结果（实际使用时需要替换为真实的训练代码）
    import random
    import time
    
    # 模拟训练时间
    time.sleep(1)
    
    # 模拟结果
    result = {
        'experiment_name': exp_name,
        'config': config,
        'metrics': {
            'nmi': random.uniform(0.6, 0.9),
            'ari': random.uniform(0.5, 0.8),
            'acc': random.uniform(0.4, 0.7),
            'training_time': random.uniform(100, 300),
            'convergence_epoch': random.randint(20, 50)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return result

def generate_summary(all_results, results_dir):
    """生成结果摘要"""
    
    summary = {
        'experiment_summary': {},
        'best_results': {},
        'comparison_table': {}
    }
    
    for group_name, group_results in all_results.items():
        group_summary = {
            'total_experiments': len(group_results),
            'successful_experiments': len([r for r in group_results.values() if 'error' not in r]),
            'failed_experiments': len([r for r in group_results.values() if 'error' in r])
        }
        
        # 找到最佳结果
        best_nmi = 0
        best_exp = None
        
        for exp_name, result in group_results.items():
            if 'error' not in result and 'metrics' in result:
                nmi = result['metrics']['nmi']
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_exp = exp_name
        
        if best_exp:
            group_summary['best_experiment'] = best_exp
            group_summary['best_nmi'] = best_nmi
        
        summary['experiment_summary'][group_name] = group_summary
    
    # 保存摘要
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    generate_markdown_report(summary, results_dir)

def generate_markdown_report(summary, results_dir):
    """生成Markdown格式的实验报告"""
    
    report_content = f"""# UMC消融实验结果报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验概览

"""
    
    for group_name, group_summary in summary['experiment_summary'].items():
        report_content += f"""### {group_name}

- 总实验数: {group_summary['total_experiments']}
- 成功实验数: {group_summary['successful_experiments']}
- 失败实验数: {group_summary['failed_experiments']}
"""
        
        if 'best_experiment' in group_summary:
            report_content += f"- 最佳实验: {group_summary['best_experiment']}\n"
            report_content += f"- 最佳NMI: {group_summary['best_nmi']:.4f}\n"
        
        report_content += "\n"
    
    report_content += """## 实验建议

1. **ConFEDE机制**: 验证双投影机制的有效性
2. **渐进策略**: 验证学习策略优化的效果
3. **架构设计**: 验证聚类优化架构的作用
4. **协同效果**: 验证创新点间的协同作用

## 下一步工作

1. 分析实验结果，确定各创新点的贡献
2. 优化表现不佳的组件
3. 进一步验证最佳配置
"""
    
    # 保存报告
    report_file = os.path.join(results_dir, "experiment_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"实验报告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='UMC消融实验')
    
    parser.add_argument('--experiment_group', type=str, default='all',
                       choices=['all', 'confede_ablation', 'progressive_ablation', 
                               'architecture_ablation', 'synergy_ablation'],
                       help='要运行的实验组')
    
    parser.add_argument('--dataset', type=str, default='MIntRec',
                       help='数据集名称')
    
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU设备')
    
    parser.add_argument('--num_runs', type=int, default=3,
                       help='每个实验的运行次数')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 运行消融实验
    run_ablation_experiments(args)

if __name__ == '__main__':
    main()
