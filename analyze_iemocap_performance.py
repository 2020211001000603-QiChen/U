#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEMOCAP-DA 性能诊断脚本
用于分析数据集特征和可能的性能问题
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import argparse

def analyze_dataset_statistics(data_path, dataset_name):
    """分析数据集统计特征"""
    
    print("=" * 60)
    print(f"{dataset_name} 数据集统计分析")
    print("=" * 60)
    
    # 检查文件
    train_file = os.path.join(data_path, dataset_name, 'train.tsv')
    dev_file = os.path.join(data_path, dataset_name, 'dev.tsv')
    test_file = os.path.join(data_path, dataset_name, 'test.tsv')
    
    if not os.path.exists(train_file):
        print(f"❌ 错误: 找不到文件 {train_file}")
        return None
    
    # 读取数据
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    dev_df = pd.read_csv(dev_file, sep='\t', header=None) if os.path.exists(dev_file) else None
    test_df = pd.read_csv(test_file, sep='\t', header=None) if os.path.exists(test_file) else None
    
    # 确定标签列位置
    if dataset_name == 'IEMOCAP-DA':
        label_col_idx = 2
        text_col_idx = 1
    elif dataset_name == 'MELD-DA':
        label_col_idx = 3
        text_col_idx = 2
    else:
        label_col_idx = 4
        text_col_idx = 3
    
    # 1. 数据集大小
    print(f"\n📊 数据集大小:")
    print(f"   训练集: {len(train_df)} 样本")
    if dev_df is not None:
        print(f"   验证集: {len(dev_df)} 样本")
    if test_df is not None:
        print(f"   测试集: {len(test_df)} 样本")
    
    # 2. 类别分布
    print(f"\n🏷️  类别分布:")
    all_labels = train_df.iloc[:, label_col_idx].tolist()
    if dev_df is not None:
        all_labels.extend(dev_df.iloc[:, label_col_idx].tolist())
    
    label_counts = Counter(all_labels)
    total = len(all_labels)
    
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {label:10s}: {count:5d} ({pct:5.2f}%) {bar}")
    
    # 3. 类别不平衡度
    counts = list(label_counts.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"\n   ⚖️  类别不平衡度: {imbalance_ratio:.2f}")
    if imbalance_ratio > 5:
        print("   ⚠️  警告: 类别严重不平衡，可能影响聚类性能")
    elif imbalance_ratio > 3:
        print("   ⚠️  注意: 类别存在不平衡，建议使用类别权重")
    else:
        print("   ✅ 类别分布相对平衡")
    
    # 4. 文本长度分析
    print(f"\n📝 文本长度分析:")
    texts = train_df.iloc[:, text_col_idx].astype(str).tolist()
    if dev_df is not None:
        texts.extend(dev_df.iloc[:, text_col_idx].astype(str).tolist())
    
    lengths = [len(text.split()) for text in texts]
    lengths = [l for l in lengths if l > 0]  # 过滤空文本
    
    if lengths:
        print(f"   平均长度: {np.mean(lengths):.2f} tokens")
        print(f"   中位数:   {np.median(lengths):.2f} tokens")
        print(f"   标准差:   {np.std(lengths):.2f} tokens")
        print(f"   最小值:   {min(lengths)} tokens")
        print(f"   最大值:   {max(lengths)} tokens")
        print(f"   25%分位:  {np.percentile(lengths, 25):.2f} tokens")
        print(f"   75%分位:  {np.percentile(lengths, 75):.2f} tokens")
        
        # 与配置的序列长度对比
        if dataset_name == 'IEMOCAP-DA':
            config_length = 44
        elif dataset_name == 'MELD-DA':
            config_length = 70
        else:
            config_length = 30
        
        coverage = sum(1 for l in lengths if l <= config_length) / len(lengths) * 100
        print(f"\n   📏 配置序列长度: {config_length}")
        print(f"   📈 覆盖率: {coverage:.2f}% (长度 <= {config_length})")
        
        if np.mean(lengths) < config_length * 0.5:
            print("   ⚠️  警告: 平均文本长度远小于配置长度，信息可能不足")
        elif np.mean(lengths) > config_length * 0.9:
            print("   ⚠️  注意: 平均文本长度接近配置长度，可能有截断")
    
    # 5. 数据质量检查
    print(f"\n🔍 数据质量检查:")
    
    # 检查缺失值
    missing_labels = train_df.iloc[:, label_col_idx].isna().sum()
    missing_texts = train_df.iloc[:, text_col_idx].isna().sum()
    print(f"   缺失标签: {missing_labels}")
    print(f"   缺失文本: {missing_texts}")
    
    # 检查空文本
    empty_texts = sum(1 for t in texts if len(str(t).strip()) == 0)
    print(f"   空文本数: {empty_texts}")
    
    if empty_texts > 0:
        print("   ⚠️  警告: 存在空文本，可能影响训练")
    
    # 6. 与MELD-DA对比（如果可用）
    if dataset_name == 'IEMOCAP-DA':
        meld_path = os.path.join(data_path, 'MELD-DA', 'train.tsv')
        if os.path.exists(meld_path):
            print(f"\n🔄 与 MELD-DA 对比:")
            meld_df = pd.read_csv(meld_path, sep='\t', header=None)
            meld_texts = meld_df.iloc[:, 2].astype(str).tolist()
            meld_lengths = [len(t.split()) for t in meld_texts if len(t.split()) > 0]
            
            if meld_lengths and lengths:
                print(f"   文本长度对比:")
                print(f"     IEMOCAP-DA: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
                print(f"     MELD-DA:     {np.mean(meld_lengths):.2f} ± {np.std(meld_lengths):.2f}")
                ratio = np.mean(lengths) / np.mean(meld_lengths)
                print(f"     比例: {ratio:.2f} (IEMOCAP-DA / MELD-DA)")
                
                if ratio < 0.7:
                    print("     ⚠️  IEMOCAP-DA文本明显更短，信息量可能不足")
    
    # 7. 性能问题诊断
    print(f"\n💡 性能问题诊断:")
    issues = []
    suggestions = []
    
    if imbalance_ratio > 3:
        issues.append("类别不平衡")
        suggestions.append("- 考虑使用类别权重或重采样")
    
    if lengths and np.mean(lengths) < 15:
        issues.append("文本信息量不足")
        suggestions.append("- 考虑增加base_dim或使用更长的上下文")
        suggestions.append("- 检查是否可以合并相邻对话片段")
    
    if lengths and np.std(lengths) / np.mean(lengths) > 1.0:
        issues.append("文本长度方差大")
        suggestions.append("- 考虑使用动态序列长度或更好的padding策略")
    
    if issues:
        print("   发现的问题:")
        for issue in issues:
            print(f"     ⚠️  {issue}")
        print("\n   改进建议:")
        for suggestion in suggestions:
            print(f"     {suggestion}")
    else:
        print("   ✅ 未发现明显的数据质量问题")
    
    # 8. 参数调整建议
    print(f"\n⚙️  参数调整建议:")
    print("   - 降低学习率: lr = 2e-4 (当前: 5e-4)")
    print("   - 提高监督温度: train_temperature_sup = 20 (当前: 10)")
    print("   - 考虑增加base_dim: base_dim = 256 (当前: 128)")
    print("   - 调整阈值策略: thres = 0.05, delta = 0.02")
    
    print("\n" + "=" * 60)
    
    return {
        'dataset_size': len(train_df),
        'num_classes': len(label_counts),
        'imbalance_ratio': imbalance_ratio,
        'avg_text_length': np.mean(lengths) if lengths else 0,
        'text_length_std': np.std(lengths) if lengths else 0,
        'label_distribution': dict(label_counts)
    }

def compare_configs():
    """对比IEMOCAP-DA和MELD-DA的配置差异"""
    
    print("\n" + "=" * 60)
    print("配置参数对比: IEMOCAP-DA vs MELD-DA")
    print("=" * 60)
    
    comparison = {
        '参数': ['学习率 (lr)', '监督温度 (temp_sup)', '无监督温度 (temp_unsup)', 
                 '初始阈值 (thres)', '阈值增量 (delta)', '基础维度 (base_dim)',
                 '预训练 (pretrain)', '批次大小 (batch_size)'],
        'IEMOCAP-DA': ['5e-4', '10', '20', '0.1', '0.05', '128', 'True', '64'],
        'MELD-DA': ['2e-4', '20', '20', '0.1', '0.05', '128', 'True', '64'],
        '差异': ['高2.5倍', '低50%', '相同', '相同', '相同', '相同', '相同', '相同'],
        '建议': ['降低到2e-4', '提高到20', '保持', '保持', '保持', '可尝试256', '保持', '保持']
    }
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    print("\n💡 关键发现:")
    print("   1. ⚠️  学习率差异最大: IEMOCAP-DA (5e-4) vs MELD-DA (2e-4)")
    print("      → 建议: 降低到 2e-4，提高训练稳定性")
    print("   2. ⚠️  监督温度差异: IEMOCAP-DA (10) vs MELD-DA (20)")
    print("      → 建议: 提高到 20，学习更平滑的分布")
    print("   3. ✅ 其他参数相同，主要差异在超参数设置")

def main():
    parser = argparse.ArgumentParser(description='IEMOCAP-DA 性能诊断工具')
    parser.add_argument('--data_path', type=str, default='Datasets',
                       help='数据集路径')
    parser.add_argument('--dataset', type=str, default='IEMOCAP-DA',
                       choices=['IEMOCAP-DA', 'MELD-DA'],
                       help='要分析的数据集')
    parser.add_argument('--compare', action='store_true',
                       help='是否对比MELD-DA')
    
    args = parser.parse_args()
    
    # 分析IEMOCAP-DA
    stats_iemocap = analyze_dataset_statistics(args.data_path, 'IEMOCAP-DA')
    
    # 如果要求对比，分析MELD-DA
    if args.compare:
        stats_meld = analyze_dataset_statistics(args.data_path, 'MELD-DA')
        
        # 对比分析
        if stats_iemocap and stats_meld:
            print("\n" + "=" * 60)
            print("数据集对比分析")
            print("=" * 60)
            
            print(f"\n📊 数据集大小:")
            print(f"   IEMOCAP-DA: {stats_iemocap['dataset_size']} 样本")
            print(f"   MELD-DA:     {stats_meld['dataset_size']} 样本")
            print(f"   比例: {stats_iemocap['dataset_size'] / stats_meld['dataset_size']:.2f}")
            
            print(f"\n📝 文本长度:")
            print(f"   IEMOCAP-DA: {stats_iemocap['avg_text_length']:.2f} ± {stats_iemocap['text_length_std']:.2f}")
            print(f"   MELD-DA:     {stats_meld['avg_text_length']:.2f} ± {stats_meld['text_length_std']:.2f}")
            if stats_iemocap['avg_text_length'] < stats_meld['avg_text_length'] * 0.7:
                print("   ⚠️  IEMOCAP-DA文本明显更短")
            
            print(f"\n⚖️  类别不平衡度:")
            print(f"   IEMOCAP-DA: {stats_iemocap['imbalance_ratio']:.2f}")
            print(f"   MELD-DA:     {stats_meld['imbalance_ratio']:.2f}")
    
    # 配置对比
    compare_configs()
    
    print("\n✅ 诊断完成！")
    print("\n📋 下一步建议:")
    print("   1. 根据诊断结果调整超参数")
    print("   2. 运行对比实验验证改进效果")
    print("   3. 监控训练过程，观察指标变化")

if __name__ == '__main__':
    main()

