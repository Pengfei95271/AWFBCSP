"""
训练时间对比可视化
提供5种专业的可视化方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

print("="*80)
print("📊 训练时间可视化")
print("="*80)

# ============================================================================
# 检查是否有真实数据，否则使用模拟数据
# ============================================================================
try:
    df = pd.read_csv('results/training_time_measurements.csv')
    print("✅ 加载真实测量数据")
    use_real_data = True
except:
    print("⚠️  未找到测量数据，使用模拟数据演示")
    use_real_data = False
    
    # 模拟数据
    data = []
    methods = ['CSP', 'FBCSP', 'AWFBCSP']
    datasets = ['2A', '2B']
    classifiers = ['SVM', 'RandomForest', 'LDA']
    
    # 基准时间（秒）
    base_times = {
        '2A': {'CSP': 0.05, 'FBCSP': 0.15, 'AWFBCSP': 0.25},
        '2B': {'CSP': 0.02, 'FBCSP': 0.08, 'AWFBCSP': 0.12}
    }
    
    clf_times = {
        'SVM': 0.02,
        'RandomForest': 0.05,
        'LDA': 0.01
    }
    
    for dataset in datasets:
        for subject_id in range(1, 10):
            for method in methods:
                for clf in classifiers:
                    feat_time = base_times[dataset][method] * np.random.uniform(0.8, 1.2)
                    clf_time = clf_times[clf] * np.random.uniform(0.8, 1.2)
                    data.append({
                        'Dataset': dataset,
                        'Subject': f'{dataset}0{subject_id}',
                        'Method': method,
                        'Classifier': clf,
                        'Feature_Extraction_Time': feat_time,
                        'Classifier_Training_Time': clf_time,
                        'Total_Training_Time': feat_time + clf_time
                    })
    
    df = pd.DataFrame(data)

# ============================================================================
# 方案1：堆叠条形图（推荐⭐⭐⭐）
# ============================================================================
print("\n1️⃣ 生成方案1：堆叠条形图（特征提取 vs 分类器训练）...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, dataset in enumerate(['2A', '2B']):
    ax = axes[idx]
    df_dataset = df[df['Dataset'] == dataset]
    
    # 计算平均时间
    summary = df_dataset.groupby('Method').agg({
        'Feature_Extraction_Time': 'mean',
        'Classifier_Training_Time': 'mean'
    }).reset_index()
    
    methods = ['CSP', 'FBCSP', 'AWFBCSP']
    feat_times = [summary[summary['Method']==m]['Feature_Extraction_Time'].values[0] for m in methods]
    clf_times = [summary[summary['Method']==m]['Classifier_Training_Time'].values[0] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.6
    
    # 绘制堆叠条形图
    p1 = ax.bar(x, feat_times, width, label='Feature Extraction', 
               color='#3498DB', edgecolor='black', linewidth=1.5)
    p2 = ax.bar(x, clf_times, width, bottom=feat_times, label='Classifier Training',
               color='#E74C3C', edgecolor='black', linewidth=1.5)
    
    # 添加数值标注
    for i, (feat, clf) in enumerate(zip(feat_times, clf_times)):
        total = feat + clf
        # 特征提取时间标注
        ax.text(i, feat/2, f'{feat*1000:.1f}ms', ha='center', va='center',
               fontweight='bold', fontsize=10, color='white')
        # 分类器训练时间标注
        ax.text(i, feat + clf/2, f'{clf*1000:.1f}ms', ha='center', va='center',
               fontweight='bold', fontsize=10, color='white')
        # 总时间标注
        ax.text(i, total + 0.01, f'{total*1000:.0f}ms', ha='center', va='bottom',
               fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) BCI-IV-{dataset}', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加加速比标注
    speedup_vs_awfbcsp = (feat_times[2] + clf_times[2]) / (feat_times[0] + clf_times[0])
    ax.text(0.98, 0.98, f'AWFBCSP\n{speedup_vs_awfbcsp:.1f}× slower\nthan CSP',
           transform=ax.transAxes, ha='right', va='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))

fig.suptitle('Training Time Breakdown: Feature Extraction vs. Classifier Training',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('results/training_time_stacked_bar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/training_time_stacked_bar.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: training_time_stacked_bar.png")
plt.close()

# ============================================================================
# 方案2：分组条形图（按分类器对比）
# ============================================================================
print("\n2️⃣ 生成方案2：按分类器分组的时间对比...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, dataset in enumerate(['2A', '2B']):
    ax = axes[idx]
    df_dataset = df[df['Dataset'] == dataset]
    
    # 计算每个分类器的平均时间
    summary = df_dataset.groupby(['Method', 'Classifier'])['Total_Training_Time'].mean().reset_index()
    
    methods = ['CSP', 'FBCSP', 'AWFBCSP']
    classifiers = df_dataset['Classifier'].unique()
    
    x = np.arange(len(classifiers))
    width = 0.25
    
    colors = {'CSP': '#E74C3C', 'FBCSP': '#3498DB', 'AWFBCSP': '#2ECC71'}
    
    for i, method in enumerate(methods):
        times = [summary[(summary['Method']==method) & (summary['Classifier']==clf)]['Total_Training_Time'].values[0] 
                for clf in classifiers]
        bars = ax.bar(x + i*width, times, width, label=method, 
                     color=colors[method], edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # 添加数值标注
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val*1000:.0f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Total Training Time (s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) BCI-IV-{dataset}', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classifiers, fontsize=11)
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Training Time Comparison Across Classifiers',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('results/training_time_by_classifier.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/training_time_by_classifier.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: training_time_by_classifier.png")
plt.close()

# ============================================================================
# 方案3：热力图（时间矩阵）
# ============================================================================
print("\n3️⃣ 生成方案3：时间热力图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, dataset in enumerate(['2A', '2B']):
    ax = axes[idx]
    df_dataset = df[df['Dataset'] == dataset]
    
    # 创建数据透视表
    pivot = df_dataset.groupby(['Method', 'Classifier'])['Total_Training_Time'].mean().reset_index()
    pivot_table = pivot.pivot(index='Method', columns='Classifier', values='Total_Training_Time')
    pivot_table = pivot_table.reindex(['CSP', 'FBCSP', 'AWFBCSP'])
    
    # 转换为毫秒
    pivot_table_ms = pivot_table * 1000
    
    # 绘制热力图
    im = ax.imshow(pivot_table_ms.values, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns, fontsize=11)
    ax.set_yticklabels(pivot_table.index, fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = ax.text(j, i, f'{pivot_table_ms.iloc[i, j]:.0f}ms',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=10)
    
    ax.set_title(f'({chr(97+idx)}) BCI-IV-{dataset}', fontsize=13, fontweight='bold', loc='left')
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Training Time (ms)', fontsize=11, fontweight='bold')

fig.suptitle('Training Time Heatmap: Method × Classifier',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('results/training_time_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/training_time_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: training_time_heatmap.png")
plt.close()

# ============================================================================
# 方案4：相对加速比图（归一化到CSP）
# ============================================================================
print("\n4️⃣ 生成方案4：相对加速比对比...")

fig, ax = plt.subplots(figsize=(10, 7))

methods = ['CSP', 'FBCSP', 'AWFBCSP']
datasets = ['2A', '2B']

x = np.arange(len(methods))
width = 0.35

for idx, dataset in enumerate(datasets):
    df_dataset = df[df['Dataset'] == dataset]
    avg_times = df_dataset.groupby('Method')['Total_Training_Time'].mean()
    
    # 归一化到CSP
    csp_time = avg_times['CSP']
    relative_times = [avg_times[m] / csp_time for m in methods]
    
    bars = ax.bar(x + idx*width, relative_times, width, 
                 label=f'BCI-IV-{dataset}',
                 edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 添加数值标注
    for bar, rel_time in zip(bars, relative_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rel_time:.2f}×',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加基准线
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='CSP Baseline')

ax.set_ylabel('Relative Training Time (normalized to CSP)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Training Time Overhead Relative to CSP Baseline', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, max(relative_times) * 1.2])

plt.tight_layout()
plt.savefig('results/training_time_relative.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/training_time_relative.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: training_time_relative.png")
plt.close()

# ============================================================================
# 方案5：综合对比图（时间 vs 准确率）- 需要准确率数据
# ============================================================================
print("\n5️⃣ 生成方案5：时间-准确率散点图...")

# 加载准确率数据
try:
    df_acc_2a = pd.read_csv('results/csp_traditional_classifiers_bci_iv_2a/results_summary.csv')
    df_acc_2b = pd.read_csv('results/csp_traditional_classifiers_bci_iv_2b/results_summary.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (dataset, df_acc) in enumerate([('2A', df_acc_2a), ('2B', df_acc_2b)]):
        ax = axes[idx]
        df_time = df[df['Dataset'] == dataset]
        
        # 计算每种方法的平均时间和准确率
        methods = ['CSP', 'FBCSP', 'AWFBCSP']
        colors = {'CSP': '#E74C3C', 'FBCSP': '#3498DB', 'AWFBCSP': '#2ECC71'}
        markers = {'CSP': 'o', 'FBCSP': 's', 'AWFBCSP': '^'}
        
        for method in methods:
            # 时间
            time_data = df_time[df_time['Method'] == method]
            avg_time = time_data['Total_Training_Time'].mean() * 1000  # 转换为ms
            std_time = time_data['Total_Training_Time'].std() * 1000
            
            # 准确率
            acc_data = df_acc[df_acc['Feature'] == method]
            avg_acc = acc_data['Mean_Accuracy'].mean() * 100
            std_acc = acc_data['Std_Accuracy'].mean() * 100
            
            # 绘制散点
            ax.errorbar(avg_time, avg_acc, xerr=std_time, yerr=std_acc,
                       marker=markers[method], markersize=15, 
                       color=colors[method], label=method,
                       capsize=5, capthick=2, linewidth=2,
                       markeredgecolor='black', markeredgewidth=1.5)
            
            # 添加标注
            ax.annotate(f'{method}\n({avg_time:.0f}ms, {avg_acc:.1f}%)',
                       xy=(avg_time, avg_acc), xytext=(15, 15),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor=colors[method], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Average Training Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) BCI-IV-{dataset}', fontsize=13, fontweight='bold', loc='left')
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('Performance-Efficiency Trade-off: Accuracy vs. Training Time',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('results/training_time_vs_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/training_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print("   ✅ 保存: training_time_vs_accuracy.png")
    plt.close()
    
except Exception as e:
    print(f"   ⚠️  方案5需要准确率数据: {e}")

# ============================================================================
# 生成统计摘要
# ============================================================================
print("\n" + "="*80)
print("📊 训练时间统计摘要")
print("="*80)

for dataset in ['2A', '2B']:
    print(f"\n▶ BCI-IV-{dataset}:")
    df_dataset = df[df['Dataset'] == dataset]
    
    summary = df_dataset.groupby('Method').agg({
        'Feature_Extraction_Time': ['mean', 'std'],
        'Classifier_Training_Time': ['mean', 'std'],
        'Total_Training_Time': ['mean', 'std']
    })
    
    for method in ['CSP', 'FBCSP', 'AWFBCSP']:
        feat_mean = summary.loc[method, ('Feature_Extraction_Time', 'mean')] * 1000
        clf_mean = summary.loc[method, ('Classifier_Training_Time', 'mean')] * 1000
        total_mean = summary.loc[method, ('Total_Training_Time', 'mean')] * 1000
        
        print(f"   {method:8s}: 特征={feat_mean:6.1f}ms, 分类器={clf_mean:5.1f}ms, 总计={total_mean:6.1f}ms")

print("\n" + "="*80)
print("✅ 所有可视化已完成！")
print("="*80)
print("\n生成的文件:")
print("  1. training_time_stacked_bar.png - 堆叠条形图（推荐⭐⭐⭐）")
print("  2. training_time_by_classifier.png - 按分类器分组")
print("  3. training_time_heatmap.png - 热力图")
print("  4. training_time_relative.png - 相对加速比")
print("  5. training_time_vs_accuracy.png - 时间vs准确率（需要准确率数据）")

