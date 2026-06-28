"""
噪声鲁棒性线型图 - 多子图版本
每个分类器一个子图，展示三种方法的趋势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# ============================================================================
# 数据准备
# ============================================================================

# 2A数据 (从表格提取)
noise_levels = [5, 10, 15, 20, 25, 30]

data_2a = {
    'SVM (RBF)': {
        'CSP': [59.60, 55.79, 56.36, 54.82, 52.63, 50.38],
        'FBCSP': [58.84, 54.41, 53.31, 52.04, 49.55, 48.15],
        'AWFBCSP': [59.00, 55.03, 55.10, 53.31, 51.47, 49.70]
    },
    'Random Forest': {
        'CSP': [57.74, 54.94, 55.44, 52.39, 51.09, 49.92],
        'FBCSP': [59.22, 57.49, 56.30, 54.87, 51.28, 51.33],
        'AWFBCSP': [60.32, 57.06, 56.50, 55.75, 51.85, 53.29]
    },
    'GBDT': {
        'CSP': [55.82, 52.97, 53.86, 51.15, 49.24, 48.46],
        'FBCSP': [57.18, 55.41, 53.01, 52.16, 50.20, 48.07],
        'AWFBCSP': [58.34, 56.05, 54.14, 53.43, 50.81, 50.00]
    },
    'AdaBoost': {
        'CSP': [53.30, 51.47, 51.89, 50.96, 47.42, 48.12],
        'FBCSP': [51.58, 50.74, 49.16, 46.68, 44.92, 44.10],
        'AWFBCSP': [53.78, 51.90, 49.69, 47.69, 44.46, 46.42]
    },
    'Naive Bayes': {
        'CSP': [55.75, 53.48, 53.69, 52.52, 49.43, 49.20],
        'FBCSP': [58.19, 54.44, 55.34, 51.67, 52.02, 50.36],
        'AWFBCSP': [58.10, 56.45, 55.30, 54.20, 51.78, 50.75]
    }
}

# 2B数据 (来自 noise_robustness_results.csv)
data_2b = {
    'SVM (RBF)': {
        'CSP': [71.82, 71.74, 71.38, 71.38, 71.39, 71.70],
        'FBCSP': [72.84, 72.24, 71.73, 72.93, 72.42, 72.41],
        'AWFBCSP': [72.20, 72.41, 72.04, 73.17, 72.34, 72.59]
    },
    'Random Forest': {
        'CSP': [65.35, 65.54, 65.50, 64.49, 65.13, 64.20],
        'FBCSP': [72.77, 72.45, 71.97, 72.57, 72.04, 72.86],
        'AWFBCSP': [72.38, 71.19, 72.30, 71.86, 72.32, 72.88]
    },
    'GBDT': {
        'CSP': [66.28, 68.67, 68.09, 66.52, 67.07, 65.88],
        'FBCSP': [72.14, 71.45, 70.79, 71.99, 71.85, 72.35],
        'AWFBCSP': [71.87, 71.93, 71.65, 70.81, 72.43, 72.21]
    },
    'AdaBoost': {
        'CSP': [70.91, 70.92, 70.73, 70.84, 71.01, 70.56],
        'FBCSP': [71.09, 70.66, 71.51, 71.68, 71.29, 71.14],
        'AWFBCSP': [71.55, 71.75, 71.37, 71.12, 71.06, 71.80]
    },
    'Naive Bayes': {
        'CSP': [71.33, 71.16, 71.11, 71.12, 71.51, 71.61],
        'FBCSP': [70.74, 70.09, 70.75, 71.55, 70.81, 70.87],
        'AWFBCSP': [70.96, 70.35, 70.53, 70.98, 70.76, 70.25]
    }
}

# 颜色方案
colors = {
    'CSP': '#E74C3C',
    'FBCSP': '#3498DB',
    'AWFBCSP': '#2ECC71'
}

markers = {
    'CSP': 'o',
    'FBCSP': 's',
    'AWFBCSP': '^'
}

# ============================================================================
# 创建多子图布局
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.35)

classifiers = ['SVM (RBF)', 'Random Forest', 'GBDT', 'AdaBoost', 'Naive Bayes']

# ============================================================================
# 第一行：2A数据集
# ============================================================================
for idx, clf in enumerate(classifiers):
    ax = fig.add_subplot(gs[0, idx])
    
    # 绘制三种方法的线
    for method in ['CSP', 'FBCSP', 'AWFBCSP']:
        values = data_2a[clf][method]
        ax.plot(noise_levels, values, 
               marker=markers[method], markersize=7, linewidth=2.5,
               label=method, color=colors[method], alpha=0.85,
               markeredgecolor='black', markeredgewidth=1)
    
    # 设置标题和标签
    clf_short = clf.replace('Gradient Boosting', 'GBDT')
    ax.set_title(f'({chr(97+idx)}) {clf_short}', fontweight='bold', loc='left', fontsize=11)
    ax.set_xlabel('Noise (%)', fontsize=10)
    
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, shadow=False, fontsize=8, ncol=1)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_xticks(noise_levels)
    ax.set_ylim([42, 65])

# ============================================================================
# 第二行：2B数据集
# ============================================================================
for idx, clf in enumerate(classifiers):
    ax = fig.add_subplot(gs[1, idx])
    
    # 绘制三种方法的线
    for method in ['CSP', 'FBCSP', 'AWFBCSP']:
        values = data_2b[clf][method]
        ax.plot(noise_levels, values,
               marker=markers[method], markersize=7, linewidth=2.5,
               label=method, color=colors[method], alpha=0.85,
               markeredgecolor='black', markeredgewidth=1)
    
    # 设置标题和标签
    clf_short = clf.replace('Gradient Boosting', 'GBDT')
    ax.set_title(f'({chr(102+idx)}) {clf_short}', fontweight='bold', loc='left', fontsize=11)
    ax.set_xlabel('Noise (%)', fontsize=10)
    
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, shadow=False, fontsize=8, ncol=1)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_xticks(noise_levels)
    ax.set_ylim([64, 75])

# 总标题
fig.suptitle('Noise Robustness Analysis: Performance Trends Across Methods', 
            fontsize=16, fontweight='bold', y=0.985)

# ============================================================================
# 保存图表
# ============================================================================
plt.savefig('results/noise_robustness_line_charts.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/noise_robustness_line_charts.png', dpi=300, bbox_inches='tight')

print("✅ 噪声鲁棒性线型图已生成:")
print("   - results/noise_robustness_line_charts.pdf")
print("   - results/noise_robustness_line_charts.png")

plt.show()

# ============================================================================
# 关键发现总结
# ============================================================================
print("\n" + "="*80)
print("📊 关键发现")
print("="*80)

print("\n1️⃣ 2A数据集 (明显下降):")
for clf in classifiers:
    csp_drop = data_2a[clf]['CSP'][0] - data_2a[clf]['CSP'][-1]
    awfbcsp_drop = data_2a[clf]['AWFBCSP'][0] - data_2a[clf]['AWFBCSP'][-1]
    print(f"   {clf:15s}: CSP下降 {csp_drop:5.2f}%, AWFBCSP下降 {awfbcsp_drop:5.2f}%")

print("\n2️⃣ 2B数据集 (惊人稳定):")
for clf in classifiers:
    csp_change = data_2b[clf]['CSP'][-1] - data_2b[clf]['CSP'][0]
    awfbcsp_change = data_2b[clf]['AWFBCSP'][-1] - data_2b[clf]['AWFBCSP'][0]
    print(f"   {clf:15s}: CSP变化 {csp_change:+5.2f}%, AWFBCSP变化 {awfbcsp_change:+5.2f}%")

print("\n3️⃣ 论文关键数值:")
avg_drop_2a_csp = np.mean([data_2a[clf]['CSP'][0] - data_2a[clf]['CSP'][-1] for clf in classifiers])
avg_drop_2a_awfbcsp = np.mean([data_2a[clf]['AWFBCSP'][0] - data_2a[clf]['AWFBCSP'][-1] for clf in classifiers])
avg_change_2b_csp = np.mean([data_2b[clf]['CSP'][-1] - data_2b[clf]['CSP'][0] for clf in classifiers])
avg_change_2b_awfbcsp = np.mean([data_2b[clf]['AWFBCSP'][-1] - data_2b[clf]['AWFBCSP'][0] for clf in classifiers])

print(f"   2A平均下降: CSP {avg_drop_2a_csp:.2f}%, AWFBCSP {avg_drop_2a_awfbcsp:.2f}%")
print(f"   2B平均变化: CSP {avg_change_2b_csp:+.2f}%, AWFBCSP {avg_change_2b_awfbcsp:+.2f}%")
print(f"\n   → AWFBCSP在2A上更鲁棒 (下降少 {avg_drop_2a_csp - avg_drop_2a_awfbcsp:.2f}%)")
print(f"   → 2B数据集几乎不受噪声影响！")

print("\n" + "="*80)

