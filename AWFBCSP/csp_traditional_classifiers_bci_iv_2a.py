"""
CSP/FBCSP/AWFBCSP + 传统分类器性能对比实验 (BCIC-IV-2A数据集)

研究目标:
1. 测试CSP、FBCSP、AWFBCSP三种特征提取方法
2. 使用9种传统分类器进行性能对比
3. 基于BCIC-IV-2A数据集，进行4类分类：左手 vs 右手 vs 双脚 vs 舌头
4. 使用22个EEG通道

特征提取方法:
- CSP: 经典共空间模式
- FBCSP: 滤波器组共空间模式  
- AWFBCSP: 自适应加权滤波器组共空间模式

分类器:
- SVM (RBF)
- Random Forest
- Gradient Boosting
- AdaBoost
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- LDA
- Logistic Regression

分类任务:
- 4类分类：左手 vs 右手 vs 双脚 vs 舌头
- 数据集：BCIC-IV-2A (9个被试，每被试288个trials，22通道，1000时间点)

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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
from eeg_preprocessing_improved import EEGPreprocessor  # 使用改进的预处理

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🧠 CSP/FBCSP/AWFBCSP + 传统分类器性能对比实验 (BCIC-IV-2A数据集)")
print("=" * 80)
print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    # 数据集参数
    subjects = list(range(1, 10))  # BCIC-IV-2A有9个被试
    sampling_rate = 250  # BCIC-IV-2A采样率
    n_channels = 22  # BCIC-IV-2A通道数
    n_timepoints = 1000  # BCIC-IV-2A时间点数
    trials_per_subject = 576  # 每被试576个trials (训练288 + 评估288)
    
    # CSP参数
    m_filters = 6  # CSP滤波器对数 (4类分类需要更多滤波器)
    freq_bands = [
        (8, 12),   # Alpha
        (12, 16),  # Low Beta
        (16, 20),  # Mid Beta
        (20, 24),  # High Beta
        (24, 30),  # Low Gamma
    ]
    
    # 交叉验证
    cv_folds = 5
    
    # 结果保存路径
    results_dir = 'results/csp_traditional_classifiers_bci_iv_2a'
    os.makedirs(results_dir, exist_ok=True)


config = Config()

# 定义分类器
classifiers = {
    'SVM (RBF)': SVC(C=10, kernel='rbf', gamma=0.01, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# 定义特征提取方法
feature_extractors = {
    'CSP': 'csp',
    'FBCSP': 'fbcsp', 
    'AWFBCSP': 'awfbcsp'
}

print(f"⚙️  配置信息:")
print(f"   - 数据集: BCIC-IV-2A")
print(f"   - 被试数: {len(config.subjects)}")
print(f"   - 通道数: {config.n_channels}")
print(f"   - 采样率: {config.sampling_rate} Hz")
print(f"   - 时间点数: {config.n_timepoints}")
print(f"   - 特征提取方法: {list(feature_extractors.keys())}")
print(f"   - 分类器数量: {len(classifiers)}")
print(f"   - 交叉验证折数: {config.cv_folds}\n")


# ============================================================================
# 步骤1: 加载BCIC-IV-2A数据集
# ============================================================================
print("=" * 80)
print("📥 步骤1: 加载BCIC-IV-2A数据集")
print("=" * 80)

def load_bci_iv_2a_data():
    """加载BCIC-IV-2A数据集 - 只使用训练数据（有标签的数据）"""
    dataset_path = 'dataset/bci_iv_2a'
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    print("📥 只加载训练数据（有标签的数据）...")
    
    for subject_id in config.subjects:
        subject_name = f"A{subject_id:02d}"
        subject_trials = 0
        
        # 只加载训练数据 (T) - 有标签的数据
        train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
        train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
        
        if os.path.exists(train_data_path) and os.path.exists(train_label_path):
            train_data = np.load(train_data_path)
            train_labels = np.load(train_label_path)
            
            all_data.append(train_data)
            all_labels.append(train_labels)
            all_subjects.extend([subject_id] * len(train_labels))
            subject_trials += len(train_labels)
            
            print(f"   被试 {subject_name}: 训练数据 {train_data.shape[0]} trials")
        else:
            print(f"   ⚠️  被试 {subject_name} 训练数据文件不存在")
        
        print(f"   被试 {subject_name} 总计: {subject_trials} trials")
    
    if not all_data:
        raise ValueError("没有找到有效的BCIC-IV-2A训练数据文件")
    
    # 合并所有被试数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    subjects = np.array(all_subjects)
    
    # 转换标签从1-4到0-3
    y = y - 1
    
    print(f"✅ 数据加载完成! 只使用训练数据，确保标签正确性")
    
    return X, y, subjects

try:
    X, y, subjects = load_bci_iv_2a_data()
    
    print(f"✅ 数据加载成功!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 通道数: {X.shape[1]}")
    print(f"   - 时间点数: {X.shape[2]}")
    print(f"   - 标签分布: 左手={np.sum(y==0)}, 右手={np.sum(y==1)}, 双脚={np.sum(y==2)}, 舌头={np.sum(y==3)}")
    print(f"   - 被试分布: {np.bincount(subjects)}\n")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)


# ============================================================================
# 步骤2: EEG数据预处理
# ============================================================================
print("=" * 80)
print("🔍 步骤2: EEG数据预处理")
print("=" * 80)

print(f"📊 原始数据信息:")
print(f"   - 数据形状: {X.shape}")
print(f"   - 标签分布: 左手={np.sum(y==0)}, 右手={np.sum(y==1)}, 双脚={np.sum(y==2)}, 舌头={np.sum(y==3)}")
print(f"   - 被试分布: {np.bincount(subjects)}")
print(f"   - 数据范围: [{X.min():.3f}, {X.max():.3f}]")
print(f"   - 数据均值: {X.mean():.3f} ± {X.std():.3f}")

# 初始化预处理器
preprocessor = EEGPreprocessor(
    sampling_rate=config.sampling_rate,
    filter_band=(8, 30),  # 运动想象相关频段
    notch_freq=50,        # 工频干扰
    baseline_correction=True,
    artifact_removal=True,
    time_window=(0.5, 4.0),  # 选择MI最明显的时间段
    standardize=True
)

print(f"\n🔧 开始预处理...")
X_processed = preprocessor.preprocess(X)

print(f"\n✅ 预处理完成!")
print(f"   - 处理后形状: {X_processed.shape}")
print(f"   - 处理后范围: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
print(f"   - 处理后均值: {X_processed.mean():.3f} ± {X_processed.std():.3f}")

# 更新数据
X = X_processed
print()


# ============================================================================
# 步骤3: 特征提取函数
# ============================================================================
print("=" * 80)
print("🔬 步骤3: 特征提取")
print("=" * 80)

def extract_features(X_train, y_train, X_test, feature_type='csp'):
    """提取EEG特征"""
    if feature_type == 'csp':
        extractor = CSP(n_components=config.m_filters * 2)
        extractor.fit(X_train, y_train)
        train_feats = extractor.transform(X_train)
        test_feats = extractor.transform(X_test)
        
    elif feature_type == 'fbcsp':
        extractor = FBCSP(
            m_filters=config.m_filters,
            sampling_rate=config.sampling_rate
        )
        extractor.fit(X_train, y_train)
        train_feats = extractor.transform(X_train)
        test_feats = extractor.transform(X_test)
        
    elif feature_type == 'awfbcsp':
        extractor = AdaptiveWeightedFBCSP(
            m_filters=config.m_filters,
            sampling_rate=config.sampling_rate,
            use_adaptive_weights=True
        )
        extractor.fit(X_train, y_train)
        train_feats = extractor.transform(X_train)
        test_feats = extractor.transform(X_test)
    
    return train_feats, test_feats


# ============================================================================
# 步骤4: 个体准确率评估实验
# ============================================================================
print("=" * 80)
print("🚀 步骤4: 个体准确率评估实验")
print("=" * 80)

# 存储结果
all_results = {}
subject_results = {}

print(f"测试 {len(feature_extractors)} 种特征提取方法 × {len(classifiers)} 种分类器")
print(f"对每个被试进行 {config.cv_folds}-Fold 交叉验证\n")

# 对每个被试单独评估
for subject_id in config.subjects:
    print(f"\n{'='*60}")
    print(f"👤 被试 A{subject_id:02d}")
    print(f"{'='*60}")
    
    # 获取该被试的数据
    subject_mask = subjects == subject_id
    X_subject = X[subject_mask]
    y_subject = y[subject_mask]
    
    print(f"数据量: {X_subject.shape[0]} trials")
    print(f"标签分布: {np.bincount(y_subject)}")
    
    subject_results[subject_id] = {}
    
    # 对该被试进行交叉验证
    kfold = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
    
    for feature_name, feature_type in feature_extractors.items():
        print(f"\n🔬 特征提取方法: {feature_name}")
        
        feature_results = {}
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_subject, y_subject), 1):
            print(f"  📍 Fold {fold_idx}/{config.cv_folds}")
            
            # 分割数据
            X_train, X_test = X_subject[train_idx], X_subject[test_idx]
            y_train, y_test = y_subject[train_idx], y_subject[test_idx]
            
            # 提取特征
            train_feats, test_feats = extract_features(X_train, y_train, X_test, feature_type)
            
            # 特征标准化
            scaler = StandardScaler()
            train_feats = scaler.fit_transform(train_feats)
            test_feats = scaler.transform(test_feats)
            
            # 测试每个分类器
            for clf_name, classifier in classifiers.items():
                # 训练分类器
                classifier.fit(train_feats, y_train)
                
                # 预测
                y_pred = classifier.predict(test_feats)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                # 存储结果
                if clf_name not in feature_results:
                    feature_results[clf_name] = []
                
                feature_results[clf_name].append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        # 计算该特征提取方法的平均指标
        feature_summary = {}
        for clf_name, fold_results in feature_results.items():
            feature_summary[clf_name] = {
                'accuracy': np.mean([r['accuracy'] for r in fold_results]),
                'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
                'precision': np.mean([r['precision'] for r in fold_results]),
                'precision_std': np.std([r['precision'] for r in fold_results]),
                'recall': np.mean([r['recall'] for r in fold_results]),
                'recall_std': np.std([r['recall'] for r in fold_results]),
                'f1': np.mean([r['f1'] for r in fold_results]),
                'f1_std': np.std([r['f1'] for r in fold_results])
            }
        
        subject_results[subject_id][feature_name] = feature_summary
        
        # 显示该被试该特征提取方法的最佳结果
        best_clf = max(feature_summary.items(), key=lambda x: x[1]['accuracy'])
        print(f"    最佳分类器: {best_clf[0]} - 准确率: {best_clf[1]['accuracy']*100:.2f}% ± {best_clf[1]['accuracy_std']*100:.2f}%")

# 计算所有被试的平均结果
print(f"\n{'='*80}")
print("📊 所有被试平均结果汇总")
print(f"{'='*80}")

for feature_name in feature_extractors.keys():
    print(f"\n🔬 特征提取方法: {feature_name}")
    print("-" * 80)
    print(f"{'分类器':<20} {'平均准确率':<15} {'标准差':<10} {'最佳被试':<10} {'最差被试':<10}")
    print("-" * 80)
    
    # 收集所有被试该特征提取方法的结果
    all_subject_accuracies = {}
    for clf_name in classifiers.keys():
        accuracies = []
        for subject_id in config.subjects:
            if subject_id in subject_results and feature_name in subject_results[subject_id]:
                acc = subject_results[subject_id][feature_name][clf_name]['accuracy']
                accuracies.append(acc)
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            best_subject = config.subjects[np.argmax(accuracies)]
            worst_subject = config.subjects[np.argmin(accuracies)]
            
            all_subject_accuracies[clf_name] = {
                'mean': mean_acc,
                'std': std_acc,
                'best_subject': best_subject,
                'worst_subject': worst_subject
            }
            
            print(f"{clf_name:<20} {mean_acc*100:6.2f}% ± {std_acc*100:5.2f}%    "
                  f"A{best_subject:02d}        A{worst_subject:02d}")
    
    # 存储到all_results中
    all_results[feature_name] = all_subject_accuracies
    print("-" * 80)


# ============================================================================
# 步骤5: 结果汇总与可视化
# ============================================================================
print("\n" + "=" * 80)
print("📊 步骤5: 个体准确率结果汇总")
print("=" * 80)

# 创建个体准确率结果表格
results_df = []
for feature_name, feature_results in all_results.items():
    for clf_name, stats in feature_results.items():
        results_df.append({
            'Feature': feature_name,
            'Classifier': clf_name,
            'Mean_Accuracy': stats['mean'],
            'Std_Accuracy': stats['std'],
            'Best_Subject': stats['best_subject'],
            'Worst_Subject': stats['worst_subject']
        })

results_df = pd.DataFrame(results_df)

# 按平均准确率排序
results_df_sorted = results_df.sort_values('Mean_Accuracy', ascending=False)

print("\n🏆 所有组合性能排名 (Top 15):")
print("-" * 120)
print(f"{'排名':<4} {'特征提取':<10} {'分类器':<20} {'平均准确率':<15} {'标准差':<10} {'最佳被试':<10} {'最差被试':<10}")
print("-" * 120)

for i, (_, row) in enumerate(results_df_sorted.head(15).iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<10} {row['Classifier']:<20} "
          f"{row['Mean_Accuracy']*100:6.2f}% ± {row['Std_Accuracy']*100:5.2f}%    "
          f"A{row['Best_Subject']:02d}        A{row['Worst_Subject']:02d}")

print("-" * 120)

# 创建个体被试结果表格
print("\n👤 个体被试结果详情:")
print("-" * 100)
print(f"{'被试':<6} {'特征提取':<10} {'最佳分类器':<20} {'准确率':<15} {'标准差':<10}")
print("-" * 100)

for subject_id in config.subjects:
    for feature_name in feature_extractors.keys():
        if subject_id in subject_results and feature_name in subject_results[subject_id]:
            best_clf = max(subject_results[subject_id][feature_name].items(), 
                          key=lambda x: x[1]['accuracy'])
            clf_name, stats = best_clf
            print(f"A{subject_id:02d}    {feature_name:<10} {clf_name:<20} "
                  f"{stats['accuracy']*100:6.2f}% ± {stats['accuracy_std']*100:5.2f}%")
print("-" * 100)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 特征提取方法对比
ax1 = axes[0, 0]
feature_means = results_df.groupby('Feature')['Mean_Accuracy'].mean().sort_values(ascending=True)
feature_stds = results_df.groupby('Feature')['Std_Accuracy'].mean()
bars = ax1.barh(feature_means.index, feature_means.values, xerr=feature_stds.values, 
                capsize=5, alpha=0.8, edgecolor='black')
ax1.set_xlabel('平均准确率 (%)', fontweight='bold')
ax1.set_title('特征提取方法性能对比 (个体准确率)', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. 分类器性能对比
ax2 = axes[0, 1]
clf_means = results_df.groupby('Classifier')['Mean_Accuracy'].mean().sort_values(ascending=True)
clf_stds = results_df.groupby('Classifier')['Std_Accuracy'].mean()
bars = ax2.barh(clf_means.index, clf_means.values, xerr=clf_stds.values, 
                capsize=5, alpha=0.8, color='orange', edgecolor='black')
ax2.set_xlabel('平均准确率 (%)', fontweight='bold')
ax2.set_title('分类器性能对比 (个体准确率)', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. 热力图 - 特征提取方法 vs 分类器
ax3 = axes[1, 0]
pivot_table = results_df.pivot(index='Feature', columns='Classifier', values='Mean_Accuracy')
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '平均准确率'})
ax3.set_title('特征提取方法 vs 分类器 性能热力图', fontweight='bold')
ax3.set_xlabel('分类器')
ax3.set_ylabel('特征提取方法')

# 4. 个体被试性能分布
ax4 = axes[1, 1]
subject_accuracies = []
subject_labels = []
for subject_id in config.subjects:
    for feature_name in feature_extractors.keys():
        if subject_id in subject_results and feature_name in subject_results[subject_id]:
            best_clf = max(subject_results[subject_id][feature_name].items(), 
                          key=lambda x: x[1]['accuracy'])
            subject_accuracies.append(best_clf[1]['accuracy'])
            subject_labels.append(f"A{subject_id:02d}\n{feature_name}")

if subject_accuracies:
    x_pos = range(len(subject_accuracies))
    bars = ax4.bar(x_pos, subject_accuracies, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('被试-特征组合', fontweight='bold')
    ax4.set_ylabel('准确率', fontweight='bold')
    ax4.set_title('个体被试最佳性能分布', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(subject_labels, rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{config.results_dir}/comprehensive_results.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 图表保存: {config.results_dir}/comprehensive_results.png")

# 保存详细结果
detailed_results = {
    'summary': results_df.to_dict('records'),
    'individual_subject_results': subject_results,
    'config': {
        'dataset': 'BCIC-IV-2A',
        'subjects': len(config.subjects),
        'cv_folds': config.cv_folds,
        'sampling_rate': config.sampling_rate,
        'n_channels': config.n_channels,
        'n_timepoints': config.n_timepoints,
        'm_filters': config.m_filters,
        'paradigm': 'CSP/FBCSP/AWFBCSP + Traditional Classifiers (Individual Accuracy)',
        'evaluation_method': 'Individual subject cross-validation'
    }
}

with open(f'{config.results_dir}/detailed_results.json', 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, indent=2, ensure_ascii=False)

# 保存CSV格式结果
results_df.to_csv(f'{config.results_dir}/results_summary.csv', index=False, encoding='utf-8')

print(f"✅ 结果保存: {config.results_dir}/detailed_results.json")
print(f"✅ CSV保存: {config.results_dir}/results_summary.csv")

# 完成
print("\n" + "=" * 80)
print("✅ 实验完成!")
print("=" * 80)
print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 显示最佳结果
best_result = results_df_sorted.iloc[0]
print(f"\n🎯 最佳组合 (平均个体准确率):")
print(f"   特征提取: {best_result['Feature']}")
print(f"   分类器: {best_result['Classifier']}")
print(f"   平均准确率: {best_result['Mean_Accuracy']*100:.2f}% ± {best_result['Std_Accuracy']*100:.2f}%")
print(f"   最佳被试: A{best_result['Best_Subject']:02d}")
print(f"   最差被试: A{best_result['Worst_Subject']:02d}")

print(f"\n📁 结果保存在: {config.results_dir}/")
print("=" * 80)
