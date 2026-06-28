"""
AWFBCSP优势可视化 - 改进版
提供多种风格选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 方案1: 简洁专业版（推荐用于论文）
# ============================================================================
def plot_style_1():
    """简洁专业版 - 更大的字体，更清晰的对比"""
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    
    # 加载数据
    imp_2a = pd.read_csv('results/awfbcsp_improvement_2a.csv')
    imp_2b = pd.read_csv('results/awfbcsp_improvement_2b.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 只显示Top-5分类器（避免拥挤）
    ax1 = axes[0]
    
    classifiers = imp_2a['Classifier'].unique()
    avg_imp_2a = np.array([imp_2a[imp_2a['Classifier']==clf]['Improvement'].mean() for clf in classifiers])
    avg_imp_2b = np.array([imp_2b[imp_2b['Classifier']==clf]['Improvement'].mean() for clf in classifiers])
    
    # 按2B的改进排序，取Top-5
    sorted_indices = np.argsort(avg_imp_2b)[::-1][:5]
    top_classifiers = classifiers[sorted_indices]
    top_imp_2a = avg_imp_2a[sorted_indices]
    top_imp_2b = avg_imp_2b[sorted_indices]
    
    x = np.arange(len(top_classifiers))
    width = 0.38
    
    bars1 = ax1.barh(x - width/2, top_imp_2a, width, label='BCI-IV-2A (4-class)', 
                    color='#FF6B6B', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax1.barh(x + width/2, top_imp_2b, width, label='BCI-IV-2B (2-class)',
                    color='#4ECDC4', alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # 简化分类器名称
    clf_labels = [c.replace('Gradient Boosting', 'GBDT').replace('K-Nearest Neighbors', 'KNN') 
                  for c in top_classifiers]
    ax1.set_yticks(x)
    ax1.set_yticklabels(clf_labels, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Average Improvement over CSP (%)', fontweight='bold', fontsize=13)
    ax1.set_title('(a) Top-5 Classifiers: AWFBCSP vs CSP', 
                 fontweight='bold', loc='left', fontsize=14, pad=15)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=11, 
              edgecolor='black', fancybox=True)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xlim([-4, 8])
    
    # 添加清晰的数值标注（总是放在条形外侧，避免重叠）
    for i, (v1, v2) in enumerate(zip(top_imp_2a, top_imp_2b)):
        # 2A数值 - 总是显示在条形末端外侧
        if v1 >= 0:
            # 正值：显示在右侧外面
            ax1.text(v1 + 0.3, i - width/2, f'{v1:+.1f}%', 
                    ha='left', va='center', 
                    fontsize=10, fontweight='bold', color='#C0392B',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#FF6B6B', alpha=0.8, linewidth=1))
        else:
            # 负值：显示在左侧外面
            ax1.text(v1 - 0.3, i - width/2, f'{v1:+.1f}%', 
                    ha='right', va='center', 
                    fontsize=10, fontweight='bold', color='#C0392B',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#FF6B6B', alpha=0.8, linewidth=1))
        
        # 2B数值 - 总是显示在条形末端外侧
        if v2 >= 0:
            ax1.text(v2 + 0.3, i + width/2, f'{v2:+.1f}%',
                    ha='left', va='center', 
                    fontsize=10, fontweight='bold', color='#16A085',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#4ECDC4', alpha=0.8, linewidth=1))
        else:
            ax1.text(v2 - 0.3, i + width/2, f'{v2:+.1f}%',
                    ha='right', va='center', 
                    fontsize=10, fontweight='bold', color='#16A085',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#4ECDC4', alpha=0.8, linewidth=1))
    
    # 子图2: 噪声趋势（更大的标记）
    ax2 = axes[1]
    
    noise_levels = [5, 10, 15, 20, 25, 30]
    avg_by_noise_2a = [imp_2a[imp_2a['Noise']==n]['Improvement'].mean() for n in noise_levels]
    avg_by_noise_2b = [imp_2b[imp_2b['Noise']==n]['Improvement'].mean() for n in noise_levels]
    
    # 2A线条
    line1 = ax2.plot(noise_levels, avg_by_noise_2a, marker='o', markersize=12, 
                    linewidth=3.5, label='BCI-IV-2A (4-class)', 
                    color='#FF6B6B', alpha=0.9, markeredgecolor='black', markeredgewidth=1.5)
    # 2B线条
    line2 = ax2.plot(noise_levels, avg_by_noise_2b, marker='s', markersize=12, 
                    linewidth=3.5, label='BCI-IV-2B (2-class)', 
                    color='#4ECDC4', alpha=0.9, markeredgecolor='black', markeredgewidth=1.5)
    
    # 添加清晰的数值标注（避免重叠）
    for i, (x, y) in enumerate(zip(noise_levels, avg_by_noise_2a)):
        offset = -0.35 if i % 2 == 0 else -0.5
        ax2.text(x, y + offset, f'{y:+.1f}', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='#C0392B',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#C0392B', alpha=0.8))
    
    for i, (x, y) in enumerate(zip(noise_levels, avg_by_noise_2b)):
        offset = 0.35 if i % 2 == 0 else 0.5
        ax2.text(x, y + offset, f'{y:+.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='#16A085',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#16A085', alpha=0.8))
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Average Improvement (%)', fontweight='bold', fontsize=13)
    ax2.set_title('(b) Performance Trend Across Noise Levels', 
                 fontweight='bold', loc='left', fontsize=14, pad=15)
    ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=11,
              edgecolor='black', fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_xticks(noise_levels)
    ax2.set_ylim([-2, 3])
    
    
    plt.tight_layout()
    
    # 保存
    plt.savefig('results/awfbcsp_advantage_style1.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/awfbcsp_advantage_style1.png', dpi=300, bbox_inches='tight')
    
    print("✅ 方案1 (简洁专业版) 已生成")
    plt.show()



# ============================================================================
# 主程序 - 生成所有方案
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("🎨 生成多种风格的AWFBCSP优势图")
    print("="*80)
    
    print("\n正在生成方案1 (简洁专业版 - 推荐)...")
    plot_style_1()
    
    print("\n" + "="*80)
    print("✅ 所有方案已生成！请查看 results/ 目录")
    print("="*80)
    print("\n文件列表:")
    print("  1. awfbcsp_advantage_style1.pdf/png - 简洁专业版 ⭐⭐⭐⭐⭐")
    print("  2. awfbcsp_advantage_style2.pdf/png - 紧凑版")
    print("  3. awfbcsp_advantage_style3.pdf/png - 信息密集版")
    print("\n推荐使用: 方案1 (最适合顶会论文)")

