"""
实验状态检查脚本
用于检查论文所需实验的完成情况
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict

class ExperimentStatusChecker:
    def __init__(self, output_dir="outputs", results_dir="outputs"):
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.datasets = ["MIntRec", "MELD-DA", "IEMOCAP-DA"]
        self.baseline_methods = ["CC", "MCN", "SCCL", "USNID", "UMC"]
        self.seeds = [0, 1, 2, 3, 4]
        
    def check_baseline_experiments(self):
        """检查基线对比实验的完成情况"""
        print("\n" + "="*60)
        print("基线对比实验状态检查")
        print("="*60)
        
        status = defaultdict(lambda: defaultdict(int))
        
        for dataset in self.datasets:
            print(f"\n数据集: {dataset}")
            for method in self.baseline_methods:
                count = 0
                # 查找结果文件
                pattern = f"{self.results_dir}/{dataset}/*{method.lower()}*.csv"
                result_files = glob.glob(pattern)
                
                # 检查是否有5次运行的结果
                if result_files:
                    # 尝试从结果文件中提取seed信息
                    for seed in self.seeds:
                        seed_files = [f for f in result_files if f"seed{seed}" in f.lower() or f"_s{seed}" in f.lower()]
                        if seed_files:
                            count += 1
                
                status[dataset][method] = count
                status_icon = "✅" if count >= 5 else "⚠️" if count > 0 else "❌"
                print(f"  {status_icon} {method}: {count}/5 次运行")
        
        return status
    
    def check_ablation_experiments(self):
        """检查消融实验的完成情况"""
        print("\n" + "="*60)
        print("消融实验状态检查")
        print("="*60)
        
        ablation_groups = {
            "ConFEDE机制消融": [
                "baseline_traditional", "confede_only", "text_guided_only",
                "gated_fusion_only", "confede_text_guided", "confede_gated", "full_confede"
            ],
            "渐进式策略消融": [
                "fixed_strategy", "progressive_only", "enhanced_loss_only",
                "kmeans_plus_only", "progressive_enhanced_loss", 
                "progressive_kmeans_plus", "full_progressive"
            ],
            "架构消融": [
                "standard_architecture", "clustering_projector_only",
                "clustering_fusion_only", "attention_pooling_only",
                "clustering_optimized", "full_optimized_architecture"
            ],
            "协同效果验证": [
                "only_confede", "only_progressive", "only_architecture",
                "confede_progressive", "confede_architecture",
                "progressive_architecture", "full_umc_model"
            ]
        }
        
        status = {}
        
        for group_name, experiments in ablation_groups.items():
            print(f"\n{group_name}:")
            group_status = {}
            
            for exp_name in experiments:
                # 查找消融实验的结果文件
                pattern = f"{self.results_dir}/*/*{exp_name}*.csv"
                result_files = glob.glob(pattern)
                
                count = len(result_files) if result_files else 0
                status_icon = "✅" if count >= 3 else "⚠️" if count > 0 else "❌"
                print(f"  {status_icon} {exp_name}: {count} 个结果文件")
                
                group_status[exp_name] = count
            
            status[group_name] = group_status
        
        return status
    
    def check_result_files(self):
        """检查结果文件的存在情况"""
        print("\n" + "="*60)
        print("结果文件检查")
        print("="*60)
        
        # 检查输出目录
        if not os.path.exists(self.output_dir):
            print(f"❌ 输出目录不存在: {self.output_dir}")
            return
        
        # 统计各数据集的结果文件
        for dataset in self.datasets:
            dataset_dir = os.path.join(self.output_dir, dataset)
            if os.path.exists(dataset_dir):
                csv_files = glob.glob(f"{dataset_dir}/*.csv")
                print(f"\n✅ {dataset}: {len(csv_files)} 个结果文件")
                if csv_files:
                    print(f"   示例: {os.path.basename(csv_files[0])}")
            else:
                print(f"\n❌ {dataset}: 目录不存在")
    
    def check_statistics(self):
        """检查是否有统计分析结果"""
        print("\n" + "="*60)
        print("统计分析检查")
        print("="*60)
        
        # 查找包含统计信息的文件
        stat_files = glob.glob(f"{self.results_dir}/*/*stats*.csv")
        stat_files += glob.glob(f"{self.results_dir}/*/*mean*.csv")
        stat_files += glob.glob(f"{self.results_dir}/*/*std*.csv")
        
        if stat_files:
            print(f"✅ 找到 {len(stat_files)} 个统计结果文件")
            for f in stat_files[:3]:
                print(f"   {f}")
        else:
            print("❌ 未找到统计分析结果文件")
            print("   建议运行 analyze_results.py 生成统计结果")
    
    def generate_report(self):
        """生成实验状态报告"""
        print("\n" + "="*60)
        print("生成实验状态报告")
        print("="*60)
        
        baseline_status = self.check_baseline_experiments()
        ablation_status = self.check_ablation_experiments()
        
        report = {
            "baseline_experiments": baseline_status,
            "ablation_experiments": ablation_status,
            "summary": {
                "total_baseline_experiments": sum(
                    sum(count for count in methods.values()) 
                    for methods in baseline_status.values()
                ),
                "total_ablation_experiments": sum(
                    sum(count for count in exps.values())
                    for exps in ablation_status.values()
                )
            }
        }
        
        # 保存报告
        report_file = "experiment_status_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 报告已保存到: {report_file}")
        
        return report
    
    def print_recommendations(self):
        """打印实验建议"""
        print("\n" + "="*60)
        print("实验建议")
        print("="*60)
        
        recommendations = [
            "1. 高优先级：完成所有基线方法的5次运行（seed: 0-4）",
            "2. 高优先级：完成关键消融实验（每个创新点的核心组件）",
            "3. 中优先级：运行超参数敏感性分析",
            "4. 中优先级：生成可视化结果（t-SNE、注意力热力图）",
            "5. 中优先级：测量计算效率（训练时间、参数量）",
            "6. 低优先级：进行失败案例分析",
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\n详细实验清单请参考: 论文实验清单.md")

def main():
    checker = ExperimentStatusChecker()
    
    print("="*60)
    print("UMC论文实验状态检查")
    print("="*60)
    
    # 检查结果文件
    checker.check_result_files()
    
    # 检查基线实验
    checker.check_baseline_experiments()
    
    # 检查消融实验
    checker.check_ablation_experiments()
    
    # 检查统计分析
    checker.check_statistics()
    
    # 生成报告
    report = checker.generate_report()
    
    # 打印建议
    checker.print_recommendations()
    
    print("\n" + "="*60)
    print("检查完成！")
    print("="*60)

if __name__ == "__main__":
    main()


