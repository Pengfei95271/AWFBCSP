"""
CSP/FBCSP/AWFBCSP + 传统分类器噪声鲁棒性分析 (BCIC-IV-2B数据集)

研究目标:
1. 测试CSP、FBCSP、AWFBCSP三种特征提取方法
2. 使用9种传统分类器进行性能对比
3. 基于BCIC-IV-2B数据集，进行2类分类：左手 vs 右手
4. 使用3个EEG通道
5. 测试不同噪声水平下的性能表现

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
- 2类分类：左手 vs 右手
- 数据集：BCIC-IV-2B (9个被试，每被试400个trials，3通道，1000时间点)
- 噪声测试：0%, 5%, 10%, 15%, 20%, 25%, 30% 高斯噪声

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
print("🛡️ CSP/FBCSP/AWFBCSP + 传统分类器噪声鲁棒性分析 (BCIC-IV-2B数据集)")
print("=" * 80)
print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    # 数据集参数
    subjects = list(range(1, 10))  # BCIC-IV-2B有9个被试
    sampling_rate = 250  # BCIC-IV-2B采样率
    n_channels = 3  # BCIC-IV-2B通道数（仅3个通道！）
    n_timepoints = 1000  # BCIC-IV-2B时间点数
    trials_per_subject = 400  # 每被试400个trials
    n_classes = 2  # 二分类（左手 vs 右手）
    
    # CSP参数（适配3通道）
    m_filters = 2  # CSP滤波器对数
    # 注意：3通道二分类实际只能提取2个有效CSP成分
    # 因为 n_components <= min(n_channels, n_classes) = min(3, 2) = 2
    freq_bands = [
        (8, 12),   # Alpha
        (12, 16),  # Low Beta
        (16, 20),  # Mid Beta
        (20, 24),  # High Beta
        (24, 30),  # Low Gamma
    ]
    
    # 交叉验证
    cv_folds = 5
    
    # 噪声测试参数
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 噪声水平
    
    # 结果保存路径
    results_dir = 'results/csp_traditional_classifiers_noise_robustness_2b'
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
print(f"   - 数据集: BCIC-IV-2B")
print(f"   - 被试数: {len(config.subjects)}")
print(f"   - 通道数: {config.n_channels} (⚠️ 仅3个通道)")
print(f"   - 采样率: {config.sampling_rate} Hz")
print(f"   - 时间点数: {config.n_timepoints}")
print(f"   - 分类任务: 二分类（左手 vs 右手）")
print(f"   - 特征提取方法: {list(feature_extractors.keys())}")
print(f"   - 分类器数量: {len(classifiers)}")
print(f"   - 交叉验证折数: {config.cv_folds}")
print(f"   - 噪声水平: {config.noise_levels}\n")


# ============================================================================
# 步骤1: 加载BCIC-IV-2B数据集
# ============================================================================
print("=" * 80)
print("📥 步骤1: 加载BCIC-IV-2B数据集")
print("=" * 80)

def load_bci_iv_2b_data():
    """加载BCIC-IV-2B数据集 - 二分类数据（左手 vs 右手）"""
    dataset_path = 'dataset/bci_iv_2b/raw'
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    print("📥 加载BCIC-IV-2B训练数据...")
    
    for subject_id in config.subjects:
        subject_name = f"B{subject_id:02d}"
        
        # 加载训练数据 (T) - 有标签的数据
        train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
        train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
        
        if os.path.exists(train_data_path) and os.path.exists(train_label_path):
            train_data = np.load(train_data_path)
            train_labels = np.load(train_label_path)
            
            all_data.append(train_data)
            all_labels.append(train_labels)
            all_subjects.extend([subject_id] * len(train_labels))
            
            print(f"   被试 {subject_name}: {train_data.shape[0]} trials")
        else:
            print(f"   ⚠️  被试 {subject_name} 训练数据文件不存在")
    
    if not all_data:
        raise ValueError("没有找到有效的BCIC-IV-2B训练数据文件")
    
    # 合并所有被试数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    subjects = np.array(all_subjects)
    
    # 转换标签从1-2到0-1（二分类）
    y = y - 1
    
    print(f"✅ 数据加载完成! BCIC-IV-2B是原生二分类数据集")
    
    return X, y, subjects

try:
    X, y, subjects = load_bci_iv_2b_data()
    
    print(f"✅ 数据加载成功!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 通道数: {X.shape[1]} (仅3个通道)")
    print(f"   - 时间点数: {X.shape[2]}")
    print(f"   - 标签分布: 左手={np.sum(y==0)}, 右手={np.sum(y==1)}")
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
print(f"   - 标签分布: 左手={np.sum(y==0)}, 右手={np.sum(y==1)}")
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
# 噪声添加函数
# ============================================================================
def add_noise(data, noise_level):
    """添加高斯噪声"""
    if noise_level == 0:
        return data
    
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# ============================================================================
# 步骤3: 特征提取函数
# ============================================================================
print("=" * 80)
print("🔬 步骤3: 特征提取")
print("=" * 80)

def extract_features(X_train, y_train, X_test, feature_type='csp'):
    """提取EEG特征（适配3通道）"""
    if feature_type == 'csp':
        # 3通道二分类：最多提取2个有效CSP成分
        # n_components 必须 <= min(n_channels, n_classes) = min(3, 2) = 2
        n_csp_components = min(config.m_filters * 2, X_train.shape[1], len(np.unique(y_train)))
        extractor = CSP(n_components=n_csp_components)
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
# 步骤4: 噪声鲁棒性评估实验
# ============================================================================
print("=" * 80)
print("🛡️ 步骤4: 噪声鲁棒性评估实验")
print("=" * 80)

# 存储结果
all_noise_results = {}
noise_subject_results = {}

print(f"测试 {len(feature_extractors)} 种特征提取方法 × {len(classifiers)} 种分类器")
print(f"噪声水平: {config.noise_levels}")
print(f"对每个被试进行 {config.cv_folds}-Fold 交叉验证\n")

# 对每个噪声水平进行测试
for noise_level in config.noise_levels:
    print(f"\n{'='*60}")
    print(f"🔊 噪声水平: {noise_level*100:.0f}%")
    print(f"{'='*60}")
    
    # 添加噪声
    X_noisy = add_noise(X, noise_level)
    
    # 存储该噪声水平的结果
    noise_subject_results[noise_level] = {}
    
    # 对每个被试单独评估
    for subject_id in config.subjects:
        print(f"\n👤 被试 B{subject_id:02d}")
        
        # 获取该被试的数据
        subject_mask = subjects == subject_id
        X_subject = X_noisy[subject_mask]
        y_subject = y[subject_mask]
        
        print(f"   数据量: {X_subject.shape[0]} trials")
        print(f"   标签分布: {np.bincount(y_subject)}")
        
        noise_subject_results[noise_level][subject_id] = {}
        
        # 对该被试进行交叉验证
        kfold = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        
        for feature_name, feature_type in feature_extractors.items():
            print(f"   🔬 特征提取方法: {feature_name}")
            
            feature_results = {}
            
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_subject, y_subject), 1):
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
                    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                    
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
            
            noise_subject_results[noise_level][subject_id][feature_name] = feature_summary
            
            # 显示该被试该特征提取方法的最佳结果
            best_clf = max(feature_summary.items(), key=lambda x: x[1]['accuracy'])
            print(f"      最佳分类器: {best_clf[0]} - 准确率: {best_clf[1]['accuracy']*100:.2f}% ± {best_clf[1]['accuracy_std']*100:.2f}%")

# 计算所有噪声水平的平均结果
print(f"\n{'='*80}")
print("📊 所有噪声水平平均结果汇总")
print(f"{'='*80}")

for noise_level in config.noise_levels:
    print(f"\n🔊 噪声水平: {noise_level*100:.0f}%")
    print("-" * 80)
    print(f"{'特征提取':<10} {'分类器':<20} {'平均准确率':<15} {'标准差':<10} {'最佳被试':<10} {'最差被试':<10}")
    print("-" * 80)
    
    # 收集所有被试该噪声水平的结果
    all_subject_accuracies = {}
    for feature_name in feature_extractors.keys():
        for clf_name in classifiers.keys():
            accuracies = []
            for subject_id in config.subjects:
                if (noise_level in noise_subject_results and 
                    subject_id in noise_subject_results[noise_level] and
                    feature_name in noise_subject_results[noise_level][subject_id]):
                    acc = noise_subject_results[noise_level][subject_id][feature_name][clf_name]['accuracy']
                    accuracies.append(acc)
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                best_subject = config.subjects[np.argmax(accuracies)]
                worst_subject = config.subjects[np.argmin(accuracies)]
                
                key = f"{feature_name}_{clf_name}"
                all_subject_accuracies[key] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'best_subject': best_subject,
                    'worst_subject': worst_subject,
                    'feature': feature_name,
                    'classifier': clf_name
                }
                
                print(f"{feature_name:<10} {clf_name:<20} {mean_acc*100:6.2f}% ± {std_acc*100:5.2f}%    "
                      f"B{best_subject:02d}        B{worst_subject:02d}")
    
    # 存储到all_noise_results中
    all_noise_results[noise_level] = all_subject_accuracies
    print("-" * 80)


# ============================================================================
# 步骤5: 结果汇总与可视化
# ============================================================================
print("\n" + "=" * 80)
print("📊 步骤5: 噪声鲁棒性结果汇总")
print("=" * 80)

# 创建噪声鲁棒性结果表格
noise_results_df = []
for noise_level, noise_results in all_noise_results.items():
    for key, stats in noise_results.items():
        noise_results_df.append({
            'Noise_Level': noise_level,
            'Feature': stats['feature'],
            'Classifier': stats['classifier'],
            'Mean_Accuracy': stats['mean'],
            'Std_Accuracy': stats['std'],
            'Best_Subject': stats['best_subject'],
            'Worst_Subject': stats['worst_subject']
        })

noise_results_df = pd.DataFrame(noise_results_df)

# 按噪声水平和平均准确率排序
noise_results_df_sorted = noise_results_df.sort_values(['Noise_Level', 'Mean_Accuracy'], ascending=[True, False])

print("\n🏆 各噪声水平下最佳组合:")
print("-" * 120)
print(f"{'噪声水平':<10} {'特征提取':<10} {'分类器':<20} {'平均准确率':<15} {'标准差':<10} {'最佳被试':<10} {'最差被试':<10}")
print("-" * 120)

for noise_level in config.noise_levels:
    noise_data = noise_results_df_sorted[noise_results_df_sorted['Noise_Level'] == noise_level]
    if not noise_data.empty:
        best_result = noise_data.iloc[0]
        print(f"{noise_level*100:6.0f}%    {best_result['Feature']:<10} {best_result['Classifier']:<20} "
              f"{best_result['Mean_Accuracy']*100:6.2f}% ± {best_result['Std_Accuracy']*100:5.2f}%    "
              f"B{best_result['Best_Subject']:02d}        B{best_result['Worst_Subject']:02d}")

print("-" * 120)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 噪声鲁棒性曲线 - 按特征提取方法
ax1 = axes[0, 0]
for feature_name in feature_extractors.keys():
    feature_data = noise_results_df[noise_results_df['Feature'] == feature_name]
    if not feature_data.empty:
        # 计算该特征提取方法在所有分类器上的平均准确率
        feature_means = feature_data.groupby('Noise_Level')['Mean_Accuracy'].mean()
        feature_stds = feature_data.groupby('Noise_Level')['Std_Accuracy'].mean()
        
        ax1.errorbar(feature_means.index, feature_means.values, yerr=feature_stds.values, 
                    marker='o', label=feature_name, capsize=5, capthick=2)

ax1.set_xlabel('噪声水平')
ax1.set_ylabel('平均准确率')
ax1.set_title('特征提取方法噪声鲁棒性对比\nBCIC-IV-2B (3通道)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# 2. 噪声鲁棒性曲线 - 按分类器
ax2 = axes[0, 1]
for clf_name in list(classifiers.keys())[:5]:  # 只显示前5个以保持可读性
    clf_data = noise_results_df[noise_results_df['Classifier'] == clf_name]
    if not clf_data.empty:
        # 计算该分类器在所有特征提取方法上的平均准确率
        clf_means = clf_data.groupby('Noise_Level')['Mean_Accuracy'].mean()
        clf_stds = clf_data.groupby('Noise_Level')['Std_Accuracy'].mean()
        
        ax2.errorbar(clf_means.index, clf_means.values, yerr=clf_stds.values, 
                    marker='s', label=clf_name, capsize=5, capthick=2)

ax2.set_xlabel('噪声水平')
ax2.set_ylabel('平均准确率')
ax2.set_title('分类器噪声鲁棒性对比\nBCIC-IV-2B (Top 5)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# 3. 热力图 - 噪声水平 vs 特征提取方法
ax3 = axes[0, 2]
pivot_table = noise_results_df.groupby(['Noise_Level', 'Feature'])['Mean_Accuracy'].mean().unstack()
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '平均准确率'})
ax3.set_title('噪声水平 vs 特征提取方法\nBCIC-IV-2B (3通道)')
ax3.set_xlabel('特征提取方法')
ax3.set_ylabel('噪声水平')

# 4. 鲁棒性指标 - 性能下降百分比
ax4 = axes[1, 0]
robustness_data = []
for feature_name in feature_extractors.keys():
    feature_data = noise_results_df[noise_results_df['Feature'] == feature_name]
    if not feature_data.empty:
        baseline_acc = feature_data[feature_data['Noise_Level'] == 0]['Mean_Accuracy'].mean()
        high_noise_acc = feature_data[feature_data['Noise_Level'] == 0.3]['Mean_Accuracy'].mean()
        robustness = (baseline_acc - high_noise_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        robustness_data.append({'Feature': feature_name, 'Robustness': robustness})

if robustness_data:
    robustness_df = pd.DataFrame(robustness_data)
    bars = ax4.bar(robustness_df['Feature'], robustness_df['Robustness'], alpha=0.8, edgecolor='black')
    ax4.set_xlabel('特征提取方法')
    ax4.set_ylabel('性能下降 (%)')
    ax4.set_title('噪声鲁棒性指标\nBCIC-IV-2B (0% vs 30%噪声)')
    ax4.grid(True, alpha=0.3)

# 5. 最佳组合在不同噪声水平下的表现
ax5 = axes[1, 1]
# 找到无噪声时最佳组合
baseline_data = noise_results_df[noise_results_df['Noise_Level'] == 0]
if not baseline_data.empty:
    best_combo = baseline_data.loc[baseline_data['Mean_Accuracy'].idxmax()]
    best_feature = best_combo['Feature']
    best_classifier = best_combo['Classifier']
    
    combo_data = noise_results_df[
        (noise_results_df['Feature'] == best_feature) & 
        (noise_results_df['Classifier'] == best_classifier)
    ]
    
    ax5.errorbar(combo_data['Noise_Level'], combo_data['Mean_Accuracy'], 
                yerr=combo_data['Std_Accuracy'], marker='o', capsize=5, capthick=2)
    ax5.set_xlabel('噪声水平')
    ax5.set_ylabel('准确率')
    ax5.set_title(f'最佳组合噪声鲁棒性\n{best_feature} + {best_classifier}\nBCIC-IV-2B')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

# 6. 噪声水平分布
ax6 = axes[1, 2]
noise_levels = config.noise_levels
accuracies_by_noise = []
for noise_level in noise_levels:
    noise_data = noise_results_df[noise_results_df['Noise_Level'] == noise_level]
    if not noise_data.empty:
        mean_acc = noise_data['Mean_Accuracy'].mean()
        accuracies_by_noise.append(mean_acc)
    else:
        accuracies_by_noise.append(0)

bars = ax6.bar(noise_levels, accuracies_by_noise, alpha=0.8, edgecolor='black')
ax6.set_xlabel('噪声水平')
ax6.set_ylabel('平均准确率')
ax6.set_title('不同噪声水平下的平均性能\nBCIC-IV-2B (3通道)')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{config.results_dir}/noise_robustness_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 图表保存: {config.results_dir}/noise_robustness_analysis.png")

# 保存详细结果
detailed_results = {
    'summary': noise_results_df.to_dict('records'),
    'noise_subject_results': noise_subject_results,
    'config': {
        'dataset': 'BCIC-IV-2B',
        'subjects': len(config.subjects),
        'cv_folds': config.cv_folds,
        'sampling_rate': config.sampling_rate,
        'n_channels': config.n_channels,
        'n_timepoints': config.n_timepoints,
        'n_classes': config.n_classes,
        'm_filters': config.m_filters,
        'noise_levels': config.noise_levels,
        'paradigm': 'CSP/FBCSP/AWFBCSP + Traditional Classifiers (Noise Robustness)',
        'evaluation_method': 'Individual subject cross-validation with noise',
        'note': 'BCIC-IV-2B uses only 3 EEG channels for binary classification'
    }
}

with open(f'{config.results_dir}/detailed_results.json', 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, indent=2, ensure_ascii=False)

# 保存CSV格式结果
noise_results_df.to_csv(f'{config.results_dir}/noise_robustness_results.csv', index=False, encoding='utf-8')

print(f"✅ 结果保存: {config.results_dir}/detailed_results.json")
print(f"✅ CSV保存: {config.results_dir}/noise_robustness_results.csv")

# 完成
print("\n" + "=" * 80)
print("✅ 噪声鲁棒性分析完成!")
print("=" * 80)
print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 显示最佳结果
baseline_data = noise_results_df[noise_results_df['Noise_Level'] == 0]
if not baseline_data.empty:
    best_result = baseline_data.loc[baseline_data['Mean_Accuracy'].idxmax()]
    print(f"\n🎯 无噪声最佳组合:")
    print(f"   特征提取: {best_result['Feature']}")
    print(f"   分类器: {best_result['Classifier']}")
    print(f"   平均准确率: {best_result['Mean_Accuracy']*100:.2f}% ± {best_result['Std_Accuracy']*100:.2f}%")

# 显示鲁棒性总结
print(f"\n🛡️ 噪声鲁棒性总结 (BCIC-IV-2B, 3通道):")
for feature_name in feature_extractors.keys():
    feature_data = noise_results_df[noise_results_df['Feature'] == feature_name]
    if not feature_data.empty:
        baseline_acc = feature_data[feature_data['Noise_Level'] == 0]['Mean_Accuracy'].mean()
        high_noise_acc = feature_data[feature_data['Noise_Level'] == 0.3]['Mean_Accuracy'].mean()
        if baseline_acc > 0:
            robustness = (baseline_acc - high_noise_acc) / baseline_acc * 100
            print(f"   - {feature_name}: 性能保持 {100-robustness:.1f}% (0% → 30%噪声)")

print(f"\n📁 结果保存在: {config.results_dir}/")
print(f"\n💡 注意: BCIC-IV-2B数据集仅使用3个EEG通道进行二分类")
print("=" * 80)

