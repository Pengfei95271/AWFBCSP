"""
CSP/FBCSP/AWFBCSP + 传统分类器性能对比实验

研究目标:
1. 测试CSP、FBCSP、AWFBCSP三种特征提取方法
2. 使用9种传统分类器进行性能对比
3. 基于PhysioNet数据集，进行4类分类：左手 vs 右手 vs 双脚 vs 静息
4. 使用前额区+运动区通道

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
- 4类分类：左手 vs 右手 vs 双脚 vs 静息
- 预期准确率：58-65% (基于文献报告)

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
from eeg_preprocessing import EEGPreprocessor

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🧠 CSP/FBCSP/AWFBCSP + 传统分类器性能对比实验 (4类分类)")
print("=" * 80)
print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    # 数据集参数
    subjects = list(range(1, 110))  # 使用全部109个被试
    sampling_rate = 160  # PhysioNet实际采样率
    
    # CSP参数
    m_filters = 3  # CSP滤波器对数
    freq_bands = [
        (8, 12),   # Alpha
        (12, 16),  # Low Beta
        (16, 20),  # Mid Beta
        (20, 24),  # High Beta
        (24, 28),  # Low Gamma
        (28, 32)   # Mid Gamma
    ]
    
    # 交叉验证
    cv_folds = 5
    
    # 结果保存路径
    results_dir = 'results/csp_traditional_classifiers_4class'
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
print(f"   - 被试数: {len(config.subjects)}")
print(f"   - 特征提取方法: {list(feature_extractors.keys())}")
print(f"   - 分类器数量: {len(classifiers)}")
print(f"   - 交叉验证折数: {config.cv_folds}\n")


# ============================================================================
# 步骤1: 加载数据
# ============================================================================
print("=" * 80)
print("📥 步骤1: 加载PhysioNet数据集")
print("=" * 80)

try:
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery
    
    dataset = PhysionetMI()
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand', 'feet', 'rest'],
        n_classes=4,
        resample=config.sampling_rate
    )
    
    print(f"正在加载被试 {config.subjects}...")
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=config.subjects)
    
    print(f"✅ 数据加载成功!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 通道数: {X.shape[1]}")
    print(f"   - 采样点数: {X.shape[2]}")
    
    # 转换标签为0/1/2/3
    label_mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}
    y = np.array([label_mapping[label] for label in labels])
    
    print(f"   - 左手: {np.sum(y==0)}, 右手: {np.sum(y==1)}, 双脚: {np.sum(y==2)}, 静息: {np.sum(y==3)}")
    print(f"   - 总类别数: {len(np.unique(y))}\n")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)


# ============================================================================
# 步骤2: 通道筛选和预处理
# ============================================================================
print("=" * 80)
print("🔍 步骤2: 通道筛选和预处理")
print("=" * 80)

# 获取通道信息
sessions = dataset.get_data(subjects=[config.subjects[0]])
raw = sessions[config.subjects[0]]['0']['0']
all_channels = raw.ch_names

# 定义前额区通道 (捕捉注意力、认知控制、眼动相关)
frontal_channels = [ch for ch in all_channels if any(x in ch for x in 
                    ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8'])]

# 定义运动区通道 (捕捉运动想象核心信号)
motor_channels = [ch for ch in all_channels if any(x in ch for x in 
                  ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                   'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'])]

# 获取通道索引
frontal_indices = [i for i, ch in enumerate(all_channels) if ch in frontal_channels]
motor_indices = [i for i, ch in enumerate(all_channels) if ch in motor_channels]

# 分离数据
X_frontal = X[:, frontal_indices, :]
X_motor = X[:, motor_indices, :]

print(f"✅ 脑区通道分离完成:")
print(f"   - 前额区通道: {len(frontal_channels)}个")
print(f"     {frontal_channels}")
print(f"   - 运动区通道: {len(motor_channels)}个") 
print(f"     {motor_channels[:10]}... (显示前10个)")
print(f"   - 前额区数据: {X_frontal.shape}")
print(f"   - 运动区数据: {X_motor.shape}")

# EEG预处理
print(f"\n🔧 EEG预处理...")
preprocessor = EEGPreprocessor(
    sampling_rate=config.sampling_rate,
    filter_low=8.0,
    filter_high=30.0,
    notch_freq=60.0,
    baseline_window=0.5,
    artifact_threshold=150.0,
    reference_type='average'
)

# 预处理前额区数据
print("前额区EEG预处理:")
X_frontal_clean, frontal_valid = preprocessor.fit_transform(X_frontal, verbose=True)

# 预处理运动区数据
print("\n运动区EEG预处理:")
X_motor_clean, motor_valid = preprocessor.fit_transform(X_motor, verbose=True)

# 取两个脑区都有效的trials
valid_trials = np.intersect1d(frontal_valid, motor_valid)
print(f"\n✅ 两脑区共同有效trials: {len(valid_trials)}/{len(y)}")

# 更新数据
X_frontal = X_frontal_clean[np.isin(frontal_valid, valid_trials)]
X_motor = X_motor_clean[np.isin(motor_valid, valid_trials)]
y = y[valid_trials]

print(f"   最终数据 - 前额区: {X_frontal.shape}, 运动区: {X_motor.shape}")
print(f"   标签分布: 左手={np.sum(y==0)}, 右手={np.sum(y==1)}, 双脚={np.sum(y==2)}, 静息={np.sum(y==3)}\n")


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
# 步骤4: 交叉验证实验
# ============================================================================
print("=" * 80)
print("🚀 步骤4: 开始交叉验证实验")
print("=" * 80)

# 存储结果
all_results = {}

# 使用运动区数据进行特征提取（传统BCI方法）
X_data = X_motor

kfold = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

print(f"使用 {config.cv_folds}-Fold 交叉验证")
print(f"测试 {len(feature_extractors)} 种特征提取方法 × {len(classifiers)} 种分类器\n")

for feature_name, feature_type in feature_extractors.items():
    print(f"\n{'='*60}")
    print(f"🔬 特征提取方法: {feature_name}")
    print(f"{'='*60}")
    
    feature_results = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_data, y), 1):
        print(f"\n📍 Fold {fold_idx}/{config.cv_folds}")
        
        # 分割数据
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 提取特征
        print(f"   提取{feature_name}特征...")
        train_feats, test_feats = extract_features(X_train, y_train, X_test, feature_type)
        
        # 特征标准化
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)
        
        print(f"   特征维度: {train_feats.shape[1]}")
        
        # 测试每个分类器
        for clf_name, classifier in classifiers.items():
            print(f"   训练 {clf_name}...")
            
            # 训练分类器
            classifier.fit(train_feats, y_train)
            
            # 预测
            y_pred = classifier.predict(test_feats)
            
            # 计算指标 (多类分类使用macro平均)
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
            
            print(f"      准确率: {accuracy*100:.2f}%")
    
    # 计算平均指标
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
    
    all_results[feature_name] = feature_summary
    
    # 显示该特征提取方法的结果
    print(f"\n📊 {feature_name} 结果汇总:")
    print("-" * 60)
    print(f"{'分类器':<20} {'准确率 (Mean ± Std)':<25} {'F1 Score'}")
    print("-" * 60)
    
    sorted_clfs = sorted(feature_summary.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for clf_name, stats in sorted_clfs:
        print(f"{clf_name:<20} {stats['accuracy']*100:5.2f}% ± {stats['accuracy_std']*100:4.2f}%       "
              f"{stats['f1']:.4f} ± {stats['f1_std']:.4f}")
    print("-" * 60)


# ============================================================================
# 步骤5: 结果汇总与可视化
# ============================================================================
print("\n" + "=" * 80)
print("📊 步骤5: 实验结果汇总")
print("=" * 80)

# 创建综合结果表格
results_df = []
for feature_name, feature_results in all_results.items():
    for clf_name, stats in feature_results.items():
        results_df.append({
            'Feature': feature_name,
            'Classifier': clf_name,
            'Accuracy': stats['accuracy'],
            'Accuracy_Std': stats['accuracy_std'],
            'F1': stats['f1'],
            'F1_Std': stats['f1_std']
        })

results_df = pd.DataFrame(results_df)

# 按准确率排序
results_df_sorted = results_df.sort_values('Accuracy', ascending=False)

print("\n🏆 所有组合性能排名 (Top 15):")
print("-" * 100)
print(f"{'排名':<4} {'特征提取':<10} {'分类器':<20} {'准确率 (Mean ± Std)':<25} {'F1 Score'}")
print("-" * 100)

for i, (_, row) in enumerate(results_df_sorted.head(15).iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<10} {row['Classifier']:<20} "
          f"{row['Accuracy']*100:5.2f}% ± {row['Accuracy_Std']*100:4.2f}%       "
          f"{row['F1']:.4f} ± {row['F1_Std']:.4f}")

print("-" * 100)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 特征提取方法对比
ax1 = axes[0, 0]
feature_means = results_df.groupby('Feature')['Accuracy'].mean().sort_values(ascending=True)
feature_stds = results_df.groupby('Feature')['Accuracy'].std()
bars = ax1.barh(feature_means.index, feature_means.values, xerr=feature_stds.values, 
                capsize=5, alpha=0.8, edgecolor='black')
ax1.set_xlabel('准确率 (%)', fontweight='bold')
ax1.set_title('特征提取方法性能对比', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. 分类器性能对比
ax2 = axes[0, 1]
clf_means = results_df.groupby('Classifier')['Accuracy'].mean().sort_values(ascending=True)
clf_stds = results_df.groupby('Classifier')['Accuracy'].std()
bars = ax2.barh(clf_means.index, clf_means.values, xerr=clf_stds.values, 
                capsize=5, alpha=0.8, color='orange', edgecolor='black')
ax2.set_xlabel('准确率 (%)', fontweight='bold')
ax2.set_title('分类器性能对比', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. 热力图 - 特征提取方法 vs 分类器
ax3 = axes[1, 0]
pivot_table = results_df.pivot(index='Feature', columns='Classifier', values='Accuracy')
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '准确率'})
ax3.set_title('特征提取方法 vs 分类器 性能热力图', fontweight='bold')
ax3.set_xlabel('分类器')
ax3.set_ylabel('特征提取方法')

# 4. Top 10 组合
ax4 = axes[1, 1]
top10 = results_df_sorted.head(10)
x_pos = range(len(top10))
bars = ax4.bar(x_pos, top10['Accuracy'].values, yerr=top10['Accuracy_Std'].values, 
               capsize=5, alpha=0.8, edgecolor='black')
ax4.set_xlabel('组合排名', fontweight='bold')
ax4.set_ylabel('准确率', fontweight='bold')
ax4.set_title('Top 10 特征-分类器组合', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"{row['Feature']}\n{row['Classifier']}" for _, row in top10.iterrows()], 
                    rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{config.results_dir}/comprehensive_results.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 图表保存: {config.results_dir}/comprehensive_results.png")

# 保存详细结果
detailed_results = {
    'summary': results_df.to_dict('records'),
    'config': {
        'subjects': len(config.subjects),
        'cv_folds': config.cv_folds,
        'sampling_rate': config.sampling_rate,
        'm_filters': config.m_filters,
        'paradigm': 'CSP/FBCSP/AWFBCSP + Traditional Classifiers'
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
print(f"\n🎯 最佳组合:")
print(f"   特征提取: {best_result['Feature']}")
print(f"   分类器: {best_result['Classifier']}")
print(f"   准确率: {best_result['Accuracy']*100:.2f}% ± {best_result['Accuracy_Std']*100:.2f}%")
print(f"   F1 Score: {best_result['F1']:.4f} ± {best_result['F1_Std']:.4f}")

print(f"\n📁 结果保存在: {config.results_dir}/")
print("=" * 80)
