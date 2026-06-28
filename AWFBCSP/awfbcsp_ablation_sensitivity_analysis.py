"""
AWFBCSP消融实验与敏感性分析

分析内容:
1. 消融实验: 逐步移除各个组件，分析贡献
2. 敏感性分析: 关键参数变化对性能的影响
3. 频带方案对比: 等宽 vs μ/β 细分
4. 交互项分析: cross-band 交互的贡献

作者: AI Assistant
日期: 2025-01-27
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
import warnings
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 导入自定义模块
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
sys.path.insert(0, script_dir)

from src.features.csp import CSP
from src.features.fbcsp import FBCSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP
from eeg_preprocessing_improved import EEGPreprocessor

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🔬 AWFBCSP消融实验与敏感性分析")
print("=" * 80)
print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    # 数据集参数
    subjects = list(range(1, 10))
    sampling_rate = 250
    n_channels = 22
    n_timepoints = 1000
    
    # 消融实验参数
    ablation_components = [
        'full',           # 完整AWFBCSP
        'no_mi_weight',   # 无MI加权
        'no_erd_ers',     # 无ERD/ERS
        'no_interaction', # 无交互项
        'no_adaptive',    # 无自适应权重
        'basic_fbcsp'     # 基础FBCSP
    ]
    
    # 敏感性分析参数
    B_values = [4, 6, 8, 10]  # 子带数
    m_values = [1, 2, 3]      # 每带滤波器数
    tau_values = np.arange(0.3, 1.3, 0.1)  # 温度参数
    
    # 频带方案
    freq_schemes = {
        'equal_width': [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
        'mu_beta_fine': [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30), (30, 32)],
        'motor_bands': [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
    }
    
    # 结果保存路径
    results_dir = 'results/awfbcsp_ablation_sensitivity'
    os.makedirs(results_dir, exist_ok=True)


config = Config()

# 定义分类器
classifiers = {
    'SVM': SVC(C=10, kernel='rbf', gamma=0.01, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LDA': LinearDiscriminantAnalysis()
}

print(f"⚙️  配置信息:")
print(f"   - 消融组件: {config.ablation_components}")
print(f"   - B值范围: {config.B_values}")
print(f"   - m值范围: {config.m_values}")
print(f"   - τ值范围: {config.tau_values}")
print(f"   - 频带方案: {list(config.freq_schemes.keys())}\n")


# ============================================================================
# 数据加载和预处理
# ============================================================================
def load_bci_iv_2a_data():
    """加载BCIC-IV-2A数据集"""
    dataset_path = 'dataset/bci_iv_2a'
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    print("📥 加载BCIC-IV-2A数据...")
    
    for subject_id in config.subjects:
        subject_name = f"A{subject_id:02d}"
        train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
        train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
        
        if os.path.exists(train_data_path) and os.path.exists(train_label_path):
            train_data = np.load(train_data_path)
            train_labels = np.load(train_label_path)
            
            all_data.append(train_data)
            all_labels.append(train_labels)
            all_subjects.extend([subject_id] * len(train_labels))
            
            print(f"   被试 {subject_name}: {train_data.shape[0]} trials")
    
    if not all_data:
        raise ValueError("没有找到有效的BCIC-IV-2A训练数据文件")
    
    X = np.vstack(all_data)
    y = np.hstack(all_labels) - 1  # 转换标签从1-4到0-3
    subjects = np.array(all_subjects)
    
    return X, y, subjects


# ============================================================================
# 1. 消融实验
# ============================================================================
def ablation_study(X, y, subjects):
    """消融实验"""
    print("=" * 80)
    print("🔬 1. 消融实验")
    print("=" * 80)
    
    # 预处理
    preprocessor = EEGPreprocessor(
        sampling_rate=config.sampling_rate,
        filter_band=(8, 30),
        notch_freq=50,
        baseline_correction=True,
        artifact_removal=True,
        time_window=(0.5, 4.0),
        standardize=True
    )
    
    X_processed = preprocessor.preprocess(X)
    
    ablation_results = {}
    
    for component in config.ablation_components:
        print(f"\n🔍 测试组件: {component}")
        
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_processed[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracies = []
            
            for train_idx, test_idx in kfold.split(X_subject, y_subject):
                X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                
                # 根据组件类型选择特征提取方法
                if component == 'full':
                    # 完整AWFBCSP
                    extractor = AdaptiveWeightedFBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate,
                        use_adaptive_weights=True,
                        use_mi_weighting=True,
                        use_erd_ers=True,
                        use_interaction=True
                    )
                elif component == 'no_mi_weight':
                    # 无MI加权
                    extractor = AdaptiveWeightedFBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate,
                        use_adaptive_weights=True,
                        use_mi_weighting=False,
                        use_erd_ers=True,
                        use_interaction=True
                    )
                elif component == 'no_erd_ers':
                    # 无ERD/ERS
                    extractor = AdaptiveWeightedFBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate,
                        use_adaptive_weights=True,
                        use_mi_weighting=True,
                        use_erd_ers=False,
                        use_interaction=True
                    )
                elif component == 'no_interaction':
                    # 无交互项
                    extractor = AdaptiveWeightedFBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate,
                        use_adaptive_weights=True,
                        use_mi_weighting=True,
                        use_erd_ers=True,
                        use_interaction=False
                    )
                elif component == 'no_adaptive':
                    # 无自适应权重
                    extractor = AdaptiveWeightedFBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate,
                        use_adaptive_weights=False,
                        use_mi_weighting=True,
                        use_erd_ers=True,
                        use_interaction=True
                    )
                elif component == 'basic_fbcsp':
                    # 基础FBCSP
                    extractor = FBCSP(
                        m_filters=6,
                        sampling_rate=config.sampling_rate
                    )
                
                # 特征提取
                extractor.fit(X_train, y_train)
                train_feats = extractor.transform(X_train)
                test_feats = extractor.transform(X_test)
                
                # 特征标准化
                scaler = StandardScaler()
                train_feats = scaler.fit_transform(train_feats)
                test_feats = scaler.transform(test_feats)
                
                # 分类
                clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
                clf.fit(train_feats, y_train)
                y_pred = clf.predict(test_feats)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            subject_accuracies[subject_id] = np.mean(accuracies)
        
        # 计算平均准确率
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            ablation_results[component] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'subject_accuracies': subject_accuracies
            }
            
            print(f"   平均准确率: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    return ablation_results


# ============================================================================
# 2. 敏感性分析
# ============================================================================
def sensitivity_analysis(X, y, subjects):
    """敏感性分析"""
    print("\n" + "=" * 80)
    print("📊 2. 敏感性分析")
    print("=" * 80)
    
    # 预处理
    preprocessor = EEGPreprocessor(
        sampling_rate=config.sampling_rate,
        filter_band=(8, 30),
        notch_freq=50,
        baseline_correction=True,
        artifact_removal=True,
        time_window=(0.5, 4.0),
        standardize=True
    )
    
    X_processed = preprocessor.preprocess(X)
    
    sensitivity_results = {}
    
    # 2.1 B值敏感性分析
    print("\n📈 2.1 B值敏感性分析")
    B_results = {}
    
    for B in config.B_values:
        print(f"   测试 B = {B}")
        
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_processed[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracies = []
            
            for train_idx, test_idx in kfold.split(X_subject, y_subject):
                X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                
                # 创建频带
                freq_bands = []
                for i in range(B):
                    start_freq = 8 + i * (24 / B)
                    end_freq = 8 + (i + 1) * (24 / B)
                    freq_bands.append((start_freq, end_freq))
                
                # 特征提取
                extractor = AdaptiveWeightedFBCSP(
                    m_filters=6,
                    sampling_rate=config.sampling_rate,
                    freq_bands=freq_bands,
                    use_adaptive_weights=True
                )
                
                extractor.fit(X_train, y_train)
                train_feats = extractor.transform(X_train)
                test_feats = extractor.transform(X_test)
                
                # 特征标准化
                scaler = StandardScaler()
                train_feats = scaler.fit_transform(train_feats)
                test_feats = scaler.transform(test_feats)
                
                # 分类
                clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
                clf.fit(train_feats, y_train)
                y_pred = clf.predict(test_feats)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            subject_accuracies[subject_id] = np.mean(accuracies)
        
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            B_results[B] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy
            }
            
            print(f"     B={B}: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    sensitivity_results['B_sensitivity'] = B_results
    
    # 2.2 m值敏感性分析
    print("\n📈 2.2 m值敏感性分析")
    m_results = {}
    
    for m in config.m_values:
        print(f"   测试 m = {m}")
        
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_processed[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracies = []
            
            for train_idx, test_idx in kfold.split(X_subject, y_subject):
                X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                
                # 特征提取
                extractor = AdaptiveWeightedFBCSP(
                    m_filters=m,
                    sampling_rate=config.sampling_rate,
                    use_adaptive_weights=True
                )
                
                extractor.fit(X_train, y_train)
                train_feats = extractor.transform(X_train)
                test_feats = extractor.transform(X_test)
                
                # 特征标准化
                scaler = StandardScaler()
                train_feats = scaler.fit_transform(train_feats)
                test_feats = scaler.transform(test_feats)
                
                # 分类
                clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
                clf.fit(train_feats, y_train)
                y_pred = clf.predict(test_feats)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            subject_accuracies[subject_id] = np.mean(accuracies)
        
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            m_results[m] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy
            }
            
            print(f"     m={m}: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    sensitivity_results['m_sensitivity'] = m_results
    
    # 2.3 τ值敏感性分析
    print("\n📈 2.3 τ值敏感性分析")
    tau_results = {}
    
    for tau in config.tau_values:
        print(f"   测试 τ = {tau:.1f}")
        
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_processed[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracies = []
            
            for train_idx, test_idx in kfold.split(X_subject, y_subject):
                X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                
                # 特征提取
                extractor = AdaptiveWeightedFBCSP(
                    m_filters=6,
                    sampling_rate=config.sampling_rate,
                    temperature=tau,
                    use_adaptive_weights=True
                )
                
                extractor.fit(X_train, y_train)
                train_feats = extractor.transform(X_train)
                test_feats = extractor.transform(X_test)
                
                # 特征标准化
                scaler = StandardScaler()
                train_feats = scaler.fit_transform(train_feats)
                test_feats = scaler.transform(test_feats)
                
                # 分类
                clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
                clf.fit(train_feats, y_train)
                y_pred = clf.predict(test_feats)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            subject_accuracies[subject_id] = np.mean(accuracies)
        
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            tau_results[tau] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy
            }
            
            print(f"     τ={tau:.1f}: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    sensitivity_results['tau_sensitivity'] = tau_results
    
    return sensitivity_results


# ============================================================================
# 3. 频带方案对比
# ============================================================================
def frequency_band_comparison(X, y, subjects):
    """频带方案对比"""
    print("\n" + "=" * 80)
    print("🎵 3. 频带方案对比")
    print("=" * 80)
    
    # 预处理
    preprocessor = EEGPreprocessor(
        sampling_rate=config.sampling_rate,
        filter_band=(8, 30),
        notch_freq=50,
        baseline_correction=True,
        artifact_removal=True,
        time_window=(0.5, 4.0),
        standardize=True
    )
    
    X_processed = preprocessor.preprocess(X)
    
    freq_results = {}
    
    for scheme_name, freq_bands in config.freq_schemes.items():
        print(f"\n🎵 测试频带方案: {scheme_name}")
        print(f"   频带: {freq_bands}")
        
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_processed[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracies = []
            
            for train_idx, test_idx in kfold.split(X_subject, y_subject):
                X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                
                # 特征提取
                extractor = AdaptiveWeightedFBCSP(
                    m_filters=6,
                    sampling_rate=config.sampling_rate,
                    freq_bands=freq_bands,
                    use_adaptive_weights=True
                )
                
                extractor.fit(X_train, y_train)
                train_feats = extractor.transform(X_train)
                test_feats = extractor.transform(X_test)
                
                # 特征标准化
                scaler = StandardScaler()
                train_feats = scaler.fit_transform(train_feats)
                test_feats = scaler.transform(test_feats)
                
                # 分类
                clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
                clf.fit(train_feats, y_train)
                y_pred = clf.predict(test_feats)
                
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            
            subject_accuracies[subject_id] = np.mean(accuracies)
        
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            freq_results[scheme_name] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'freq_bands': freq_bands,
                'n_bands': len(freq_bands)
            }
            
            print(f"   平均准确率: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
            print(f"   频带数量: {len(freq_bands)}")
    
    return freq_results


# ============================================================================
# 4. 结果可视化
# ============================================================================
def visualize_results(ablation_results, sensitivity_results, freq_results):
    """结果可视化"""
    print("\n" + "=" * 80)
    print("📊 4. 结果可视化")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 消融实验结果
    ax1 = axes[0, 0]
    components = list(ablation_results.keys())
    accuracies = [ablation_results[comp]['mean_accuracy'] for comp in components]
    stds = [ablation_results[comp]['std_accuracy'] for comp in components]
    
    bars = ax1.bar(components, accuracies, yerr=stds, capsize=5, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('组件')
    ax1.set_ylabel('准确率')
    ax1.set_title('消融实验结果')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. B值敏感性
    ax2 = axes[0, 1]
    B_values = list(sensitivity_results['B_sensitivity'].keys())
    B_accuracies = [sensitivity_results['B_sensitivity'][B]['mean_accuracy'] for B in B_values]
    B_stds = [sensitivity_results['B_sensitivity'][B]['std_accuracy'] for B in B_values]
    
    ax2.errorbar(B_values, B_accuracies, yerr=B_stds, marker='o', capsize=5, capthick=2)
    ax2.set_xlabel('子带数 B')
    ax2.set_ylabel('准确率')
    ax2.set_title('B值敏感性分析')
    ax2.grid(True, alpha=0.3)
    
    # 3. m值敏感性
    ax3 = axes[0, 2]
    m_values = list(sensitivity_results['m_sensitivity'].keys())
    m_accuracies = [sensitivity_results['m_sensitivity'][m]['mean_accuracy'] for m in m_values]
    m_stds = [sensitivity_results['m_sensitivity'][m]['std_accuracy'] for m in m_values]
    
    ax3.errorbar(m_values, m_accuracies, yerr=m_stds, marker='s', capsize=5, capthick=2)
    ax3.set_xlabel('每带滤波器数 m')
    ax3.set_ylabel('准确率')
    ax3.set_title('m值敏感性分析')
    ax3.grid(True, alpha=0.3)
    
    # 4. τ值敏感性
    ax4 = axes[1, 0]
    tau_values = list(sensitivity_results['tau_sensitivity'].keys())
    tau_accuracies = [sensitivity_results['tau_sensitivity'][tau]['mean_accuracy'] for tau in tau_values]
    tau_stds = [sensitivity_results['tau_sensitivity'][tau]['std_accuracy'] for tau in tau_values]
    
    ax4.errorbar(tau_values, tau_accuracies, yerr=tau_stds, marker='^', capsize=5, capthick=2)
    ax4.set_xlabel('温度参数 τ')
    ax4.set_ylabel('准确率')
    ax4.set_title('τ值敏感性分析')
    ax4.grid(True, alpha=0.3)
    
    # 5. 频带方案对比
    ax5 = axes[1, 1]
    freq_schemes = list(freq_results.keys())
    freq_accuracies = [freq_results[scheme]['mean_accuracy'] for scheme in freq_schemes]
    freq_stds = [freq_results[scheme]['std_accuracy'] for scheme in freq_schemes]
    
    bars = ax5.bar(freq_schemes, freq_accuracies, yerr=freq_stds, capsize=5, alpha=0.8, edgecolor='black')
    ax5.set_xlabel('频带方案')
    ax5.set_ylabel('准确率')
    ax5.set_title('频带方案对比')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. 组件贡献分析
    ax6 = axes[1, 2]
    full_acc = ablation_results['full']['mean_accuracy']
    component_contributions = []
    component_names = []
    
    for comp in ['no_mi_weight', 'no_erd_ers', 'no_interaction', 'no_adaptive']:
        if comp in ablation_results:
            contribution = full_acc - ablation_results[comp]['mean_accuracy']
            component_contributions.append(contribution)
            component_names.append(comp.replace('no_', '').replace('_', ' ').title())
    
    bars = ax6.bar(component_names, component_contributions, alpha=0.8, edgecolor='black')
    ax6.set_xlabel('组件')
    ax6.set_ylabel('性能贡献')
    ax6.set_title('各组件性能贡献')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.results_dir}/ablation_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表保存: {config.results_dir}/ablation_sensitivity_analysis.png")


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序"""
    try:
        # 加载数据
        X, y, subjects = load_bci_iv_2a_data()
        print(f"✅ 数据加载成功: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints")
        
        # 1. 消融实验
        ablation_results = ablation_study(X, y, subjects)
        
        # 2. 敏感性分析
        sensitivity_results = sensitivity_analysis(X, y, subjects)
        
        # 3. 频带方案对比
        freq_results = frequency_band_comparison(X, y, subjects)
        
        # 4. 结果可视化
        visualize_results(ablation_results, sensitivity_results, freq_results)
        
        # 5. 保存结果
        results = {
            'ablation_study': ablation_results,
            'sensitivity_analysis': sensitivity_results,
            'frequency_band_comparison': freq_results,
            'config': {
                'dataset': 'BCIC-IV-2A',
                'subjects': len(config.subjects),
                'ablation_components': config.ablation_components,
                'B_values': config.B_values,
                'm_values': config.m_values,
                'tau_values': config.tau_values.tolist(),
                'freq_schemes': config.freq_schemes,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(f'{config.results_dir}/ablation_sensitivity_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 消融实验与敏感性分析完成!")
        print(f"📁 结果保存在: {config.results_dir}/")
        
        # 总结报告
        print("\n" + "=" * 80)
        print("📋 消融实验与敏感性分析总结报告")
        print("=" * 80)
        
        # 消融实验总结
        print(f"🔬 消融实验结果:")
        full_acc = ablation_results['full']['mean_accuracy']
        for comp, result in ablation_results.items():
            if comp != 'full':
                contribution = full_acc - result['mean_accuracy']
                print(f"   - {comp}: {result['mean_accuracy']*100:.2f}% (贡献: {contribution*100:.2f}%)")
        
        # 敏感性分析总结
        print(f"\n📊 敏感性分析结果:")
        B_best = max(sensitivity_results['B_sensitivity'].items(), key=lambda x: x[1]['mean_accuracy'])
        m_best = max(sensitivity_results['m_sensitivity'].items(), key=lambda x: x[1]['mean_accuracy'])
        tau_best = max(sensitivity_results['tau_sensitivity'].items(), key=lambda x: x[1]['mean_accuracy'])
        
        print(f"   - 最佳B值: {B_best[0]} ({B_best[1]['mean_accuracy']*100:.2f}%)")
        print(f"   - 最佳m值: {m_best[0]} ({m_best[1]['mean_accuracy']*100:.2f}%)")
        print(f"   - 最佳τ值: {tau_best[0]:.1f} ({tau_best[1]['mean_accuracy']*100:.2f}%)")
        
        # 频带方案总结
        print(f"\n🎵 频带方案对比结果:")
        freq_best = max(freq_results.items(), key=lambda x: x[1]['mean_accuracy'])
        print(f"   - 最佳频带方案: {freq_best[0]} ({freq_best[1]['mean_accuracy']*100:.2f}%)")
        print(f"   - 频带数量: {freq_best[1]['n_bands']}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
