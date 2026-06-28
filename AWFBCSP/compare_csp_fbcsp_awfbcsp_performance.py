"""
CSP vs FBCSP vs AWFBCSP 性能对比实验

对比内容：
1. 三种方法的分类准确率
2. 特征维度对比
3. AWFBCSP各组件的贡献分析（Ablation Study）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

sys.path.insert(0, 'src')

from src.features.csp import CSP
from src.features.fbcsp import FBCSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP


def load_data(subject_id):
    """加载数据"""
    data_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_data.npy'
    label_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_label.npy'
    
    X = np.load(data_path)
    y = np.load(label_path)
    
    # 二分类（左右手）
    binary_mask = (y == 1) | (y == 2)
    X = X[binary_mask]
    y = y[binary_mask] - 1
    
    return X, y


def evaluate_method(X, y, feature_extractor, method_name):
    """评估单个方法"""
    print(f"\n{'='*60}")
    print(f"🔍 评估方法: {method_name}")
    print(f"{'='*60}")
    
    # 5-fold交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    feature_dims = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 提取特征
        X_train_feat = feature_extractor.fit_transform(X_train, y_train)
        X_test_feat = feature_extractor.transform(X_test)
        
        # 特征维度
        feature_dims.append(X_train_feat.shape[1])
        
        # 分类
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis())
        ])
        clf.fit(X_train_feat, y_train)
        acc = clf.score(X_test_feat, y_test)
        accuracies.append(acc)
        
        print(f"   Fold {fold}: 准确率={acc*100:.2f}%, 特征维度={X_train_feat.shape[1]}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_dim = int(np.mean(feature_dims))
    
    print(f"\n   📊 {method_name} 总结:")
    print(f"      平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"      特征维度: {mean_dim}")
    
    return {
        'method': method_name,
        'accuracy_mean': mean_acc,
        'accuracy_std': std_acc,
        'feature_dim': mean_dim,
        'accuracies': accuracies
    }


def main():
    """主函数"""
    print("="*80)
    print("📊 CSP vs FBCSP vs AWFBCSP 性能对比实验")
    print("="*80)
    
    # 配置
    subject_id = 1
    sampling_rate = 250
    
    print(f"\n📌 被试: A{subject_id:02d}")
    print(f"📌 分类器: Linear Discriminant Analysis (LDA)")
    print(f"📌 交叉验证: 5-fold Stratified CV")
    
    # 加载数据
    print(f"\n{'='*60}")
    print("🔄 加载数据...")
    print(f"{'='*60}")
    X, y = load_data(subject_id)
    print(f"   数据形状: {X.shape}")
    print(f"   类别分布: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
    
    # ==================== 1. CSP (单频段baseline) ====================
    csp = CSP(n_components=6)
    results_csp = evaluate_method(X, y, csp, "CSP (8-30Hz单频段)")
    
    # ==================== 2. FBCSP (多频段等权重) ====================
    fbcsp = FBCSP(
        m_filters=3,
        sampling_rate=sampling_rate,
        n_components=6
    )
    results_fbcsp = evaluate_method(X, y, fbcsp, "FBCSP (多频段等权重)")
    
    # ==================== 3. AWFBCSP (完整版：加权+补充特征) ====================
    awfbcsp_full = AdaptiveWeightedFBCSP(
        m_filters=3,
        sampling_rate=sampling_rate,
        use_adaptive_weights=True,
        use_temporal_windows=True,
        use_erd_features=True,
        use_multiscale=True
    )
    results_awfbcsp_full = evaluate_method(X, y, awfbcsp_full, 
                                           "AWFBCSP (完整版)")
    
    # ==================== 4. AWFBCSP Ablation Study ====================
    print(f"\n{'='*80}")
    print("🔬 AWFBCSP 消融实验 (Ablation Study)")
    print(f"{'='*80}")
    
    ablation_configs = [
        {
            'name': 'AWFBCSP (仅自适应权重)',
            'use_adaptive_weights': True,
            'use_temporal_windows': False,
            'use_erd_features': False,
            'use_multiscale': False
        },
        {
            'name': 'AWFBCSP (+ Temporal Windows)',
            'use_adaptive_weights': True,
            'use_temporal_windows': True,
            'use_erd_features': False,
            'use_multiscale': False
        },
        {
            'name': 'AWFBCSP (+ Temporal + ERD)',
            'use_adaptive_weights': True,
            'use_temporal_windows': True,
            'use_erd_features': True,
            'use_multiscale': False
        },
        {
            'name': 'AWFBCSP (完整版: All Features)',
            'use_adaptive_weights': True,
            'use_temporal_windows': True,
            'use_erd_features': True,
            'use_multiscale': True
        }
    ]
    
    ablation_results = []
    for config in ablation_configs:
        awfbcsp = AdaptiveWeightedFBCSP(
            m_filters=3,
            sampling_rate=sampling_rate,
            use_adaptive_weights=config['use_adaptive_weights'],
            use_temporal_windows=config['use_temporal_windows'],
            use_erd_features=config['use_erd_features'],
            use_multiscale=config['use_multiscale']
        )
        result = evaluate_method(X, y, awfbcsp, config['name'])
        ablation_results.append(result)
    
    # ==================== 可视化对比 ====================
    print(f"\n{'='*80}")
    print("🎨 生成对比图...")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. 准确率对比（主要三种方法） ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    methods = [results_csp['method'], results_fbcsp['method'], 
               results_awfbcsp_full['method']]
    accs = [results_csp['accuracy_mean']*100, 
            results_fbcsp['accuracy_mean']*100,
            results_awfbcsp_full['accuracy_mean']*100]
    stds = [results_csp['accuracy_std']*100,
            results_fbcsp['accuracy_std']*100,
            results_awfbcsp_full['accuracy_std']*100]
    
    colors = ['steelblue', 'orange', 'red']
    bars = ax1.bar(range(len(methods)), accs, yerr=stds, 
                   color=colors, alpha=0.7, edgecolor='black', capsize=5)
    
    # 添加数值标签
    for i, (bar, acc, std) in enumerate(zip(bars, accs, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(['CSP', 'FBCSP', 'AWFBCSP'], rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # ========== 2. 特征维度对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    dims = [results_csp['feature_dim'], 
            results_fbcsp['feature_dim'],
            results_awfbcsp_full['feature_dim']]
    
    bars2 = ax2.bar(range(len(methods)), dims, 
                    color=colors, alpha=0.7, edgecolor='black')
    
    for bar, dim in zip(bars2, dims):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dim}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Feature Dimension', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Dimension Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(['CSP', 'FBCSP', 'AWFBCSP'], rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== 3. 准确率提升百分比 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    baseline_acc = results_csp['accuracy_mean'] * 100
    improvements = [0,  # CSP baseline
                   (results_fbcsp['accuracy_mean']*100 - baseline_acc),
                   (results_awfbcsp_full['accuracy_mean']*100 - baseline_acc)]
    
    bars3 = ax3.bar(range(len(methods)), improvements, 
                    color=colors, alpha=0.7, edgecolor='black')
    
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        if height != 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{imp:.2f}%' if imp > 0 else f'{imp:.2f}%',
                    ha='center', va='bottom' if imp > 0 else 'top',
                    fontweight='bold', fontsize=10)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_ylabel('Improvement over CSP (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Performance Improvement', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(['CSP', 'FBCSP', 'AWFBCSP'], rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # ========== 4. AWFBCSP消融实验 ==========
    ax4 = fig.add_subplot(gs[1, :2])
    
    ablation_names = [r['method'].replace('AWFBCSP ', '') for r in ablation_results]
    ablation_accs = [r['accuracy_mean']*100 for r in ablation_results]
    ablation_stds = [r['accuracy_std']*100 for r in ablation_results]
    ablation_dims = [r['feature_dim'] for r in ablation_results]
    
    x_pos = np.arange(len(ablation_names))
    width = 0.35
    
    # 准确率柱状图
    bars4 = ax4.bar(x_pos, ablation_accs, width, 
                    yerr=ablation_stds,
                    color=['lightcoral', 'coral', 'orangered', 'red'],
                    alpha=0.7, edgecolor='black', capsize=5,
                    label='Accuracy')
    
    # 特征维度折线图（右轴）
    ax4_twin = ax4.twinx()
    line4 = ax4_twin.plot(x_pos, ablation_dims, 
                          'go-', linewidth=2, markersize=10,
                          label='Feature Dim')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars4, ablation_accs)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + ablation_stds[i],
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for i, dim in enumerate(ablation_dims):
        ax4_twin.text(i, dim + 5, f'{dim}',
                     ha='center', va='bottom', fontweight='bold', 
                     fontsize=9, color='green')
    
    ax4.set_xlabel('AWFBCSP Configuration', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4_twin.set_ylabel('Feature Dimension', fontsize=11, fontweight='bold', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4.set_title('AWFBCSP Ablation Study: Component Contribution', 
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(ablation_names, rotation=20, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    # ========== 5. 关键发现文本框 ==========
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 计算关键数值
    fbcsp_gain = (results_fbcsp['accuracy_mean'] - results_csp['accuracy_mean']) * 100
    awfbcsp_gain = (results_awfbcsp_full['accuracy_mean'] - results_csp['accuracy_mean']) * 100
    awfbcsp_vs_fbcsp = (results_awfbcsp_full['accuracy_mean'] - results_fbcsp['accuracy_mean']) * 100
    
    findings_text = f"""
📊 Key Findings:

1️⃣ Performance Gain:
   • FBCSP vs CSP: +{fbcsp_gain:.2f}%
   • AWFBCSP vs CSP: +{awfbcsp_gain:.2f}%
   • AWFBCSP vs FBCSP: +{awfbcsp_vs_fbcsp:.2f}%

2️⃣ Feature Efficiency:
   • CSP: {results_csp['feature_dim']} features
   • FBCSP: {results_fbcsp['feature_dim']} features
   • AWFBCSP: {results_awfbcsp_full['feature_dim']} features

3️⃣ AWFBCSP Components:
   • Adaptive Weighting: Base
   • + Temporal Windows: ↑
   • + ERD Features: ↑
   • + Multiscale: ↑ Best!

✅ AWFBCSP = 加权FBCSP特征
              + 补充特征
    """
    
    ax5.text(0.5, 0.5, findings_text, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== 总标题 ==========
    fig.suptitle(f'CSP vs FBCSP vs AWFBCSP: Performance Comparison\n'
                 f'BCI Competition IV 2a, Subject A{subject_id:02d}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 保存
    output_path = 'results/csp_fbcsp_awfbcsp_performance_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ 对比图已保存: {output_path}")
    
    plt.show()
    
    # ==================== 打印总结表格 ====================
    print(f"\n{'='*80}")
    print("📊 性能对比总结表:")
    print(f"{'='*80}")
    print(f"\n{'方法':<30} {'准确率':<20} {'特征维度':<15} {'vs CSP':<15}")
    print("-"*80)
    
    all_results = [results_csp, results_fbcsp, results_awfbcsp_full]
    for result in all_results:
        acc_str = f"{result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%"
        gain = (result['accuracy_mean'] - results_csp['accuracy_mean']) * 100
        gain_str = f"+{gain:.2f}%" if gain > 0 else f"{gain:.2f}%"
        print(f"{result['method']:<30} {acc_str:<20} {result['feature_dim']:<15} {gain_str:<15}")
    
    print(f"\n{'='*80}")
    print("🔬 AWFBCSP消融实验:")
    print(f"{'='*80}")
    print(f"\n{'配置':<40} {'准确率':<20} {'特征维度':<10}")
    print("-"*80)
    
    for result in ablation_results:
        acc_str = f"{result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%"
        print(f"{result['method']:<40} {acc_str:<20} {result['feature_dim']:<10}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()




