"""
AWFBCSP优势可视化 - 论文用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# 加载数据
imp_2a = pd.read_csv('results/awfbcsp_improvement_2a.csv')
imp_2b = pd.read_csv('results/awfbcsp_improvement_2b.csv')

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ============================================================================
# 子图1: 按分类器显示平均改进
# ============================================================================
ax1 = axes[0]

classifiers = imp_2a['Classifier'].unique()
avg_imp_2a = [imp_2a[imp_2a['Classifier']==clf]['Improvement'].mean() for clf in classifiers]
avg_imp_2b = [imp_2b[imp_2b['Classifier']==clf]['Improvement'].mean() for clf in classifiers]

x = np.arange(len(classifiers))
width = 0.35

bars1 = ax1.barh(x - width/2, avg_imp_2a, width, label='2A (4-class)', 
                color='#E67E22', alpha=0.8, edgecolor='black')
bars2 = ax1.barh(x + width/2, avg_imp_2b, width, label='2B (2-class)',
                color='#9B59B6', alpha=0.8, edgecolor='black')

ax1.set_yticks(x)
ax1.set_yticklabels([c.replace('Gradient Boosting', 'GBDT').replace('K-Nearest Neighbors', 'KNN') 
                      for c in classifiers], fontsize=9)
ax1.set_xlabel('Average Improvement over CSP (%)', fontweight='bold', fontsize=11)
ax1.set_title('(a) AWFBCSP vs CSP: Per-Classifier Improvement', 
             fontweight='bold', loc='left', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax1.legend(loc='lower right', frameon=True, shadow=True)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# 添加数值标注
for i, (v1, v2) in enumerate(zip(avg_imp_2a, avg_imp_2b)):
    if v1 != 0:
        ax1.text(v1 + (0.3 if v1 > 0 else -0.3), i - width/2, f'{v1:+.1f}', 
                ha='left' if v1 > 0 else 'right', va='center', fontsize=8)
    if v2 != 0:
        ax1.text(v2 + (0.5 if v2 > 0 else -0.5), i + width/2, f'{v2:+.1f}',
                ha='left' if v2 > 0 else 'right', va='center', fontsize=8, fontweight='bold')

# ============================================================================
# 子图2: 按噪声等级显示改进趋势
# ============================================================================
ax2 = axes[1]

noise_levels = [5, 10, 15, 20, 25, 30]
avg_by_noise_2a = [imp_2a[imp_2a['Noise']==n]['Improvement'].mean() for n in noise_levels]
avg_by_noise_2b = [imp_2b[imp_2b['Noise']==n]['Improvement'].mean() for n in noise_levels]

ax2.plot(noise_levels, avg_by_noise_2a, marker='o', markersize=10, linewidth=3,
        label='2A (4-class)', color='#E67E22', alpha=0.8)
ax2.plot(noise_levels, avg_by_noise_2b, marker='s', markersize=10, linewidth=3,
        label='2B (2-class)', color='#9B59B6', alpha=0.8)

# 添加数值标注
for x, y in zip(noise_levels, avg_by_noise_2a):
    ax2.text(x, y - 0.3, f'{y:+.1f}', ha='center', va='top', fontsize=9)
for x, y in zip(noise_levels, avg_by_noise_2b):
    ax2.text(x, y + 0.3, f'{y:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax2.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Average Improvement (%)', fontweight='bold', fontsize=11)
ax2.set_title('(b) AWFBCSP Advantage Across Noise Levels', 
             fontweight='bold', loc='left', fontsize=12)
ax2.legend(loc='upper right', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(noise_levels)

# 添加关键结论文本框
textstr = f'2A: {imp_2a["Improvement"].mean():+.2f}% avg (40.7% improved)\n2B: {imp_2b["Improvement"].mean():+.2f}% avg (87.0% improved)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()

# 保存
plt.savefig('results/awfbcsp_advantage_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/awfbcsp_advantage_publication.png', dpi=300, bbox_inches='tight')

print("✅ AWFBCSP优势图已生成:")
print("   - results/awfbcsp_advantage_publication.pdf")
print("   - results/awfbcsp_advantage_publication.png")

plt.show()

