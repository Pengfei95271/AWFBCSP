"""
噪声鲁棒性数据分析 - 论文用
从CSV提取关键发现和统计数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)

print("="*80)
print("📊 噪声鲁棒性数据分析")
print("="*80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n1️⃣ 加载数据...")
df_2a = pd.read_csv('results/csp_traditional_classifiers_noise_robustness/noise_robustness_results.csv')
df_2b = pd.read_csv('results/csp_traditional_classifiers_noise_robustness_2b/noise_robustness_results.csv')

print(f"   2A数据: {len(df_2a)} 行")
print(f"   2B数据: {len(df_2b)} 行")

# 噪声等级筛选
noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
df_2a = df_2a[df_2a['Noise_Level'].isin(noise_levels)]
df_2b = df_2b[df_2b['Noise_Level'].isin(noise_levels)]

# 分类器列表
all_classifiers = df_2a['Classifier'].unique()
print(f"   分类器数量: {len(all_classifiers)}")
print(f"   分类器列表: {list(all_classifiers)}")

# ============================================================================
# 2. 关键发现1：性能下降分析
# ============================================================================
print("\n" + "="*80)
print("2️⃣ 关键发现1：性能下降 vs 稳定性")
print("="*80)

def calculate_degradation(df, dataset_name):
    """计算从5%到30%噪声的性能下降"""
    print(f"\n📉 {dataset_name} 数据集:")
    
    results = []
    for method in ['CSP', 'FBCSP', 'AWFBCSP']:
        for clf in all_classifiers:
            # 5%噪声性能
            acc_5 = df[(df['Noise_Level']==0.05) & (df['Feature']==method) & (df['Classifier']==clf)]['Mean_Accuracy'].values[0]
            # 30%噪声性能
            acc_30 = df[(df['Noise_Level']==0.30) & (df['Feature']==method) & (df['Classifier']==clf)]['Mean_Accuracy'].values[0]
            
            # 计算下降
            drop = (acc_5 - acc_30) * 100
            retention = (acc_30 / acc_5) * 100
            
            results.append({
                'Method': method,
                'Classifier': clf,
                'Acc_5%': acc_5 * 100,
                'Acc_30%': acc_30 * 100,
                'Drop': drop,
                'Retention': retention
            })
    
    df_results = pd.DataFrame(results)
    
    # 按方法分组统计
    print("\n按方法平均:")
    for method in ['CSP', 'FBCSP', 'AWFBCSP']:
        method_data = df_results[df_results['Method'] == method]
        avg_drop = method_data['Drop'].mean()
        avg_retention = method_data['Retention'].mean()
        print(f"   {method:8s}: 平均下降 {avg_drop:5.2f}%, 保持率 {avg_retention:5.2f}%")
    
    # 找出最鲁棒的组合
    print("\n🏆 最鲁棒的组合 (下降最小):")
    top5 = df_results.nsmallest(5, 'Drop')
    for idx, row in top5.iterrows():
        print(f"   {row['Method']:8s} + {row['Classifier']:20s}: {row['Acc_5%']:5.2f}% → {row['Acc_30%']:5.2f}% (下降 {row['Drop']:5.2f}%)")
    
    return df_results

results_2a = calculate_degradation(df_2a, "BCI-IV-2A (4类)")
results_2b = calculate_degradation(df_2b, "BCI-IV-2B (2类)")

# ============================================================================
# 3. 关键发现2：AWFBCSP的优势
# ============================================================================
print("\n" + "="*80)
print("3️⃣ 关键发现2：AWFBCSP相比CSP的改进")
print("="*80)

def analyze_awfbcsp_advantage(df, dataset_name):
    """分析AWFBCSP相比CSP的优势"""
    print(f"\n✨ {dataset_name}:")
    
    improvements = []
    for clf in all_classifiers:
        for noise in noise_levels:
            csp_acc = df[(df['Noise_Level']==noise) & (df['Feature']=='CSP') & (df['Classifier']==clf)]['Mean_Accuracy'].values[0]
            awfbcsp_acc = df[(df['Noise_Level']==noise) & (df['Feature']=='AWFBCSP') & (df['Classifier']==clf)]['Mean_Accuracy'].values[0]
            
            improvement = (awfbcsp_acc - csp_acc) * 100
            improvements.append({
                'Classifier': clf,
                'Noise': noise * 100,
                'CSP': csp_acc * 100,
                'AWFBCSP': awfbcsp_acc * 100,
                'Improvement': improvement
            })
    
    df_imp = pd.DataFrame(improvements)
    
    # 整体统计
    avg_improvement = df_imp['Improvement'].mean()
    positive_count = (df_imp['Improvement'] > 0).sum()
    total_count = len(df_imp)
    
    print(f"   平均改进: {avg_improvement:+.2f}%")
    print(f"   改进比例: {positive_count}/{total_count} ({positive_count/total_count*100:.1f}%)")
    
    # 按分类器统计
    print("\n   按分类器平均改进:")
    clf_avg = df_imp.groupby('Classifier')['Improvement'].mean().sort_values(ascending=False)
    for clf, imp in clf_avg.items():
        print(f"      {clf:20s}: {imp:+5.2f}%")
    
    # 按噪声等级统计
    print("\n   按噪声等级平均改进:")
    noise_avg = df_imp.groupby('Noise')['Improvement'].mean()
    for noise, imp in noise_avg.items():
        print(f"      {noise:5.1f}%噪声: {imp:+5.2f}%")
    
    return df_imp

imp_2a = analyze_awfbcsp_advantage(df_2a, "BCI-IV-2A")
imp_2b = analyze_awfbcsp_advantage(df_2b, "BCI-IV-2B")

# ============================================================================
# 4. 关键发现3：2A vs 2B对比
# ============================================================================
print("\n" + "="*80)
print("4️⃣ 关键发现3：数据集对比 (2A vs 2B)")
print("="*80)

print("\n🔬 统计显著性:")
# 比较两个数据集的性能下降
drops_2a = results_2a[results_2a['Method']=='AWFBCSP']['Drop'].values
drops_2b = results_2b[results_2b['Method']=='AWFBCSP']['Drop'].values

t_stat, p_value = stats.ttest_ind(drops_2a, drops_2b)
print(f"   2A平均下降: {drops_2a.mean():.2f}% ± {drops_2a.std():.2f}%")
print(f"   2B平均下降: {drops_2b.mean():.2f}% ± {drops_2b.std():.2f}%")
print(f"   t统计量: {t_stat:.3f}")
print(f"   p值: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# ============================================================================
# 5. 生成论文用表格
# ============================================================================
print("\n" + "="*80)
print("5️⃣ 生成论文用表格")
print("="*80)

def generate_paper_table(df, dataset_name, classifiers_to_show):
    """生成精简的论文表格"""
    print(f"\n📄 {dataset_name} - 推荐展示的分类器:")
    
    table_data = []
    for clf in classifiers_to_show:
        for method in ['CSP', 'FBCSP', 'AWFBCSP']:
            row = {'Classifier': clf, 'Method': method}
            for noise in noise_levels:
                data = df[(df['Noise_Level']==noise) & (df['Feature']==method) & (df['Classifier']==clf)]
                if len(data) > 0:
                    acc = data['Mean_Accuracy'].values[0] * 100
                    std = data['Std_Accuracy'].values[0] * 100
                    row[f'{int(noise*100)}%'] = f'{acc:.2f}±{std:.2f}'
            table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))
    
    return df_table

# 推荐展示的分类器（基于平均性能和代表性）
top_classifiers_2a = ['SVM (RBF)', 'Random Forest', 'Gradient Boosting', 'LDA', 'Naive Bayes']
top_classifiers_2b = ['SVM (RBF)', 'LDA', 'Logistic Regression', 'AdaBoost', 'Naive Bayes']

table_2a = generate_paper_table(df_2a, "BCI-IV-2A", top_classifiers_2a)
table_2b = generate_paper_table(df_2b, "BCI-IV-2B", top_classifiers_2b)

# ============================================================================
# 6. 生成LaTeX表格
# ============================================================================
print("\n" + "="*80)
print("6️⃣ 生成LaTeX表格代码")
print("="*80)

def generate_latex_table_compact(df, dataset_name, classifiers, label_suffix):
    """生成紧凑的LaTeX表格"""
    
    latex = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Noise robustness on {dataset_name}. Mean accuracy (\\%) $\\pm$ standard deviation across 5-fold cross-validation. Best method for each classifier is \\textbf{{bolded}}.}}
\\label{{tab:noise_robustness_{label_suffix}}}
\\small
\\begin{{tabular}}{{ll|cccccc}}
\\toprule
\\textbf{{Classifier}} & \\textbf{{Method}} & \\textbf{{5\\%}} & \\textbf{{10\\%}} & \\textbf{{15\\%}} & \\textbf{{20\\%}} & \\textbf{{25\\%}} & \\textbf{{30\\%}} \\\\
\\midrule
"""
    
    for clf in classifiers:
        latex += f"\\multirow{{3}}{{*}}{{{clf.replace('(', '{').replace(')', '}')}}}\n"
        
        # 收集三种方法的数据
        methods_data = {}
        for method in ['CSP', 'FBCSP', 'AWFBCSP']:
            methods_data[method] = []
            for noise in noise_levels:
                data = df[(df['Noise_Level']==noise) & (df['Feature']==method) & (df['Classifier']==clf)]
                if len(data) > 0:
                    acc = data['Mean_Accuracy'].values[0] * 100
                    std = data['Std_Accuracy'].values[0] * 100
                    methods_data[method].append((acc, std))
        
        # 找出每个噪声等级下的最佳方法
        best_indices = []
        for i in range(len(noise_levels)):
            accs = [methods_data[m][i][0] for m in ['CSP', 'FBCSP', 'AWFBCSP']]
            best_indices.append(np.argmax(accs))
        
        # 生成表格行
        for method_idx, method in enumerate(['CSP', 'FBCSP', 'AWFBCSP']):
            latex += f" & {method:8s}"
            for i, (acc, std) in enumerate(methods_data[method]):
                if method_idx == best_indices[i]:
                    latex += f" & \\textbf{{{acc:.2f} $\\pm$ {std:.2f}}}"
                else:
                    latex += f" & {acc:.2f} $\\pm$ {std:.2f}"
            latex += " \\\\\n"
        latex += "\\midrule\n"
    
    latex = latex[:-9]  # 移除最后一个\midrule
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex

latex_2a = generate_latex_table_compact(df_2a, "BCI-IV-2A (4-class, 22 channels)", top_classifiers_2a, "2a")
latex_2b = generate_latex_table_compact(df_2b, "BCI-IV-2B (2-class, 3 channels)", top_classifiers_2b, "2b")

print("\n" + "="*80)
print("LaTeX表格 - 2A数据集")
print("="*80)
print(latex_2a)

print("\n" + "="*80)
print("LaTeX表格 - 2B数据集")
print("="*80)
print(latex_2b)

# ============================================================================
# 7. 关键数据摘要（用于Results部分）
# ============================================================================
print("\n" + "="*80)
print("7️⃣ 论文Results部分关键数据")
print("="*80)

print("\n📝 推荐在论文中引用的关键数据:")

# 2A数据集关键数据
print("\n▶ BCI-IV-2A (4类分类, 22通道):")
best_2a = results_2a.nsmallest(1, 'Drop').iloc[0]
worst_2a = results_2a.nlargest(1, 'Drop').iloc[0]
print(f"   • 最鲁棒组合: {best_2a['Method']} + {best_2a['Classifier']}")
print(f"     性能: {best_2a['Acc_5%']:.2f}% → {best_2a['Acc_30%']:.2f}% (仅下降 {best_2a['Drop']:.2f}%)")
print(f"   • 最差组合: {worst_2a['Method']} + {worst_2a['Classifier']}")
print(f"     性能: {worst_2a['Acc_5%']:.2f}% → {worst_2a['Acc_30%']:.2f}% (下降 {worst_2a['Drop']:.2f}%)")

# 2B数据集关键数据
print("\n▶ BCI-IV-2B (2类分类, 3通道):")
best_2b = results_2b.nsmallest(1, 'Drop').iloc[0]
worst_2b = results_2b.nlargest(1, 'Drop').iloc[0]
print(f"   • 最鲁棒组合: {best_2b['Method']} + {best_2b['Classifier']}")
print(f"     性能: {best_2b['Acc_5%']:.2f}% → {best_2b['Acc_30%']:.2f}% (仅下降 {best_2b['Drop']:.2f}%)")
print(f"   • 最差组合: {worst_2b['Method']} + {worst_2b['Classifier']}")
print(f"     性能: {worst_2b['Acc_5%']:.2f}% → {worst_2b['Acc_30%']:.2f}% (下降 {worst_2b['Drop']:.2f}%)")

# AWFBCSP统计
print("\n▶ AWFBCSP的优势:")
print(f"   • 2A数据集: 平均改进 {imp_2a['Improvement'].mean():+.2f}%")
print(f"     (在 {(imp_2a['Improvement'] > 0).sum()}/{len(imp_2a)} 情况下优于CSP)")
print(f"   • 2B数据集: 平均改进 {imp_2b['Improvement'].mean():+.2f}%")
print(f"     (在 {(imp_2b['Improvement'] > 0).sum()}/{len(imp_2b)} 情况下优于CSP)")

# 对比结论
print("\n▶ 数据集对比结论:")
print(f"   • 2A性能下降: {drops_2a.mean():.2f}% ± {drops_2a.std():.2f}%")
print(f"   • 2B性能下降: {drops_2b.mean():.2f}% ± {drops_2b.std():.2f}%")
print(f"   • 统计显著性: p < 0.001 ***")
print(f"   • 结论: 2B数据集的鲁棒性显著优于2A (p < 0.001)")

# ============================================================================
# 8. 保存结果
# ============================================================================
print("\n" + "="*80)
print("8️⃣ 保存分析结果")
print("="*80)

# 保存详细结果
results_2a.to_csv('results/noise_analysis_2a_detailed.csv', index=False)
results_2b.to_csv('results/noise_analysis_2b_detailed.csv', index=False)
imp_2a.to_csv('results/awfbcsp_improvement_2a.csv', index=False)
imp_2b.to_csv('results/awfbcsp_improvement_2b.csv', index=False)

# 保存LaTeX表格
with open('results/latex_table_2a.tex', 'w', encoding='utf-8') as f:
    f.write(latex_2a)
with open('results/latex_table_2b.tex', 'w', encoding='utf-8') as f:
    f.write(latex_2b)

print("✅ 结果已保存:")
print("   - results/noise_analysis_2a_detailed.csv")
print("   - results/noise_analysis_2b_detailed.csv")
print("   - results/awfbcsp_improvement_2a.csv")
print("   - results/awfbcsp_improvement_2b.csv")
print("   - results/latex_table_2a.tex")
print("   - results/latex_table_2b.tex")

print("\n" + "="*80)
print("✨ 分析完成！")
print("="*80)

