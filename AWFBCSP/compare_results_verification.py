"""
验证notebook和Python脚本的结果差异
比较单个被试 vs 所有被试平均的准确率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("📊 Notebook vs Python脚本结果对比分析")
print("=" * 80)

# ============================================================================
# 1. 加载Python脚本的结果（所有被试平均）
# ============================================================================
print("\n1️⃣ Python脚本结果（9个被试平均）:")
print("-" * 80)

# 2a数据集
df_2a = pd.read_csv('results/csp_traditional_classifiers_bci_iv_2a/results_summary.csv')
print("\n📁 BCI-IV-2A数据集:")
print(df_2a[['Feature', 'Classifier', 'Mean_Accuracy', 'Std_Accuracy']].head(9))

# 2b数据集
df_2b = pd.read_csv('results/csp_traditional_classifiers_bci_iv_2b/results_summary.csv')
print("\n📁 BCI-IV-2B数据集:")
print(df_2b[['Feature', 'Classifier', 'Mean_Accuracy', 'Std_Accuracy']].head(9))

# ============================================================================
# 2. 分析个体被试差异
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣ 个体被试性能分析")
print("=" * 80)

# 从detailed_results.json中提取单个被试的结果
import json

with open('results/csp_traditional_classifiers_bci_iv_2b/detailed_results.json', 'r', encoding='utf-8') as f:
    detailed_2b = json.load(f)

# 提取B01被试的结果（对应notebook测试的被试）
subject_1_results = detailed_2b['individual_subject_results']['1']

print("\n👤 被试B01（Notebook测试的被试）的准确率:")
print("-" * 80)
print(f"{'特征方法':<15} {'分类器':<20} {'准确率':<10} {'vs 平均':<15}")
print("-" * 80)

for feature in ['CSP', 'FBCSP', 'AWFBCSP']:
    for clf in ['SVM (RBF)', 'Random Forest', 'LDA']:
        if feature in subject_1_results and clf in subject_1_results[feature]:
            b01_acc = subject_1_results[feature][clf]['accuracy'] * 100
            
            # 获取所有被试平均
            avg_row = df_2b[(df_2b['Feature'] == feature) & (df_2b['Classifier'] == clf)]
            if not avg_row.empty:
                avg_acc = avg_row['Mean_Accuracy'].values[0] * 100
                diff = b01_acc - avg_acc
                
                diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
                print(f"{feature:<15} {clf:<20} {b01_acc:6.2f}%   {diff_str:<15}")

# ============================================================================
# 3. 可视化对比
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣ 可视化分析")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 所有被试的准确率分布（CSP + SVM）
ax1 = axes[0, 0]
csp_svm_accuracies = []
for subject_id in range(1, 10):
    if str(subject_id) in detailed_2b['individual_subject_results']:
        acc = detailed_2b['individual_subject_results'][str(subject_id)]['CSP']['SVM (RBF)']['accuracy']
        csp_svm_accuracies.append(acc * 100)

subjects = [f'B{i:02d}' for i in range(1, 10)]
bars = ax1.bar(subjects, csp_svm_accuracies, alpha=0.7, edgecolor='black')
bars[0].set_color('red')  # 高亮B01
bars[0].set_alpha(0.9)

# 添加平均线
avg_line = np.mean(csp_svm_accuracies)
ax1.axhline(y=avg_line, color='blue', linestyle='--', linewidth=2, label=f'平均值: {avg_line:.2f}%')

ax1.set_xlabel('被试', fontweight='bold')
ax1.set_ylabel('准确率 (%)', fontweight='bold')
ax1.set_title('CSP + SVM (RBF) 各被试准确率\n(B01用红色标注)', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标注
for i, (bar, acc) in enumerate(zip(bars, csp_svm_accuracies)):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

# 子图2: B01 vs 平均值对比（多个分类器）
ax2 = axes[0, 1]

classifiers_to_compare = ['SVM (RBF)', 'Random Forest', 'LDA', 'Naive Bayes']
b01_accs = []
avg_accs = []

for clf in classifiers_to_compare:
    b01_acc = subject_1_results['CSP'][clf]['accuracy'] * 100
    avg_row = df_2b[(df_2b['Feature'] == 'CSP') & (df_2b['Classifier'] == clf)]
    avg_acc = avg_row['Mean_Accuracy'].values[0] * 100
    
    b01_accs.append(b01_acc)
    avg_accs.append(avg_acc)

x = np.arange(len(classifiers_to_compare))
width = 0.35

bars1 = ax2.bar(x - width/2, b01_accs, width, label='B01 (Notebook)', alpha=0.7, color='coral')
bars2 = ax2.bar(x + width/2, avg_accs, width, label='所有被试平均 (Python脚本)', alpha=0.7, color='skyblue')

ax2.set_xlabel('分类器', fontweight='bold')
ax2.set_ylabel('准确率 (%)', fontweight='bold')
ax2.set_title('B01 vs 平均值对比 (CSP特征)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(classifiers_to_compare, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 子图3: 三种特征方法在B01上的表现
ax3 = axes[1, 0]

features = ['CSP', 'FBCSP', 'AWFBCSP']
b01_best_accs = []
avg_best_accs = []

for feature in features:
    # B01的最佳准确率
    b01_best = max([subject_1_results[feature][clf]['accuracy'] 
                    for clf in subject_1_results[feature].keys()])
    b01_best_accs.append(b01_best * 100)
    
    # 所有被试平均的最佳准确率
    feature_df = df_2b[df_2b['Feature'] == feature]
    avg_best = feature_df['Mean_Accuracy'].max()
    avg_best_accs.append(avg_best * 100)

x = np.arange(len(features))
width = 0.35

bars1 = ax3.bar(x - width/2, b01_best_accs, width, label='B01', alpha=0.7, color='coral')
bars2 = ax3.bar(x + width/2, avg_best_accs, width, label='平均', alpha=0.7, color='skyblue')

ax3.set_xlabel('特征提取方法', fontweight='bold')
ax3.set_ylabel('最佳准确率 (%)', fontweight='bold')
ax3.set_title('不同特征方法的最佳性能对比', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(features)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 添加数值标注
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 子图4: 个体差异分析（标准差）
ax4 = axes[1, 1]

features_std = []
std_values = []

for feature in ['CSP', 'FBCSP', 'AWFBCSP']:
    feature_df = df_2b[df_2b['Feature'] == feature]
    avg_std = feature_df['Std_Accuracy'].mean() * 100
    features_std.append(feature)
    std_values.append(avg_std)

bars = ax4.bar(features_std, std_values, alpha=0.7, color='lightgreen', edgecolor='black')

ax4.set_xlabel('特征提取方法', fontweight='bold')
ax4.set_ylabel('平均标准差 (%)', fontweight='bold')
ax4.set_title('跨被试性能变异性\n(标准差越小越稳定)', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 添加数值标注
for bar, val in zip(bars, std_values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/notebook_vs_script_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ 对比图已保存: results/notebook_vs_script_comparison.png")
plt.show()

# ============================================================================
# 4. 总结报告
# ============================================================================
print("\n" + "=" * 80)
print("📝 结果差异总结")
print("=" * 80)

print("\n🔍 为什么Notebook和Python脚本结果不同？")
print("-" * 80)
print("1. 📊 测试范围不同:")
print("   - Notebook: 只测试1个被试 (B01)")
print("   - Python脚本: 测试所有9个被试并计算平均值")
print()
print("2. 👤 个体差异:")
b01_avg_acc = np.mean([subject_1_results['CSP'][clf]['accuracy'] * 100 
                       for clf in subject_1_results['CSP'].keys()])
all_avg_acc = df_2b[df_2b['Feature'] == 'CSP']['Mean_Accuracy'].mean() * 100
print(f"   - B01平均准确率: {b01_avg_acc:.2f}%")
print(f"   - 所有被试平均: {all_avg_acc:.2f}%")
print(f"   - 差异: {b01_avg_acc - all_avg_acc:+.2f}%")
print()
print("3. 🎯 最佳被试:")
best_subject = df_2b['Best_Subject'].mode()[0]
print(f"   - B{best_subject:02d} 在大多数组合中表现最好")
print(f"   - B01 {'是' if best_subject == 1 else '不是'}最佳被试")

print("\n" + "=" * 80)
print("💡 建议")
print("=" * 80)
print("\n对于论文发表:")
print("✅ 使用Python脚本的结果（所有被试平均 ± 标准差）")
print("✅ 在补充材料中提供每个被试的详细结果")
print("✅ Notebook用于快速验证和单被试分析")
print()
print("如果要让Notebook和脚本结果一致:")
print("1. 修改Notebook循环所有被试")
print("2. 或者在Notebook中只测试最佳被试 (B04)")
print("3. 或者在脚本中只测试B01来验证Notebook结果")

print("\n" + "=" * 80)


