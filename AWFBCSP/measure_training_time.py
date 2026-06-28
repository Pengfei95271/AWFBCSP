"""
测量CSP、FBCSP、AWFBCSP的真实训练时间
包含特征提取和分类器训练两部分
"""

import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 导入特征提取方法
import sys
sys.path.append('src')
from features.csp import CSP
from features.fbcsp import FBCSP
from features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP as AWFBCSP
from utils.data_loader import load_single_subject

print("="*80)
print("⏱️  训练时间测量实验")
print("="*80)

# ============================================================================
# 配置
# ============================================================================
class Config:
    # 数据集设置
    datasets = {
        '2A': {
            'path': 'dataset/bci_iv_2a',
            'subjects': list(range(1, 10)),
            'n_channels': 22,
            'n_classes': 4,
            'sampling_rate': 250
        },
        '2B': {
            'path': 'dataset/bci_iv_2b/raw',
            'subjects': list(range(1, 10)),
            'n_channels': 3,
            'n_classes': 2,
            'sampling_rate': 250
        }
    }
    
    # 特征提取参数
    n_components = 4  # CSP成分数（对应m_filters=2，因为2*2=4）
    m_filters = 2  # FBCSP每个频段的滤波器对数
    freq_bands = [[4, 8], [8, 12], [12, 16], [16, 20], 
                  [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]  # 9个频段
    
    # 分类器
    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LDA': LDA()
    }
    
    # 重复次数
    n_repeats = 5
    n_folds = 5

config = Config()

# ============================================================================
# 时间测量函数
# ============================================================================
def measure_feature_extraction_time(X_train, y_train, X_test, method='CSP', repeat=10):
    """测量特征提取时间"""
    times = []
    
    for _ in range(repeat):
        if method == 'CSP':
            start = time.time()
            extractor = CSP(n_components=config.n_components)
            extractor.fit(X_train, y_train)
            _ = extractor.transform(X_train)
            _ = extractor.transform(X_test)
            elapsed = time.time() - start
            
        elif method == 'FBCSP':
            start = time.time()
            extractor = FBCSP(
                m_filters=config.m_filters,
                freq_bands=config.freq_bands,
                sampling_rate=250
            )
            extractor.fit(X_train, y_train)
            _ = extractor.transform(X_train)
            _ = extractor.transform(X_test)
            elapsed = time.time() - start
            
        elif method == 'AWFBCSP':
            start = time.time()
            extractor = AWFBCSP(
                m_filters=config.m_filters,
                sampling_rate=250,
                use_adaptive_weights=True,
                use_temporal_windows=False,
                use_erd_features=False,
                use_multiscale=False
            )
            extractor.fit(X_train, y_train)
            _ = extractor.transform(X_train)
            _ = extractor.transform(X_test)
            elapsed = time.time() - start
        
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

def measure_classifier_training_time(X_train, y_train, classifier_name, repeat=10):
    """测量分类器训练时间"""
    times = []
    
    for _ in range(repeat):
        clf = config.classifiers[classifier_name]
        start = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

# ============================================================================
# 主实验循环
# ============================================================================
results = []

for dataset_name, dataset_config in config.datasets.items():
    print(f"\n{'='*80}")
    print(f"📊 数据集: BCI-IV-{dataset_name}")
    print(f"{'='*80}")
    
    for subject_id in dataset_config['subjects']:
        print(f"\n👤 被试 {dataset_name}0{subject_id}")
        
        # 加载数据
        try:
            if dataset_name == '2A':
                X, y = load_single_subject(
                    dataset_config['path'],
                    subject_id,
                    dataset='2a'
                )
            else:
                X, y = load_single_subject(
                    dataset_config['path'],
                    subject_id,
                    dataset='2b'
                )
                y = y - 1  # 转换为0-1
            
            print(f"   数据形状: {X.shape}, 标签: {np.unique(y)}")
        except Exception as e:
            print(f"   ⚠️  加载失败: {e}")
            continue
        
        # 5-Fold交叉验证
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
        
        fold_results = {
            'CSP': {'feat_time': [], 'clf_time': {}},
            'FBCSP': {'feat_time': [], 'clf_time': {}},
            'AWFBCSP': {'feat_time': [], 'clf_time': {}}
        }
        
        for clf_name in config.classifiers.keys():
            for method in ['CSP', 'FBCSP', 'AWFBCSP']:
                fold_results[method]['clf_time'][clf_name] = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 测量每种方法的特征提取时间
            for method in ['CSP', 'FBCSP', 'AWFBCSP']:
                feat_time_mean, feat_time_std = measure_feature_extraction_time(
                    X_train, y_train, X_test, method, repeat=config.n_repeats
                )
                fold_results[method]['feat_time'].append(feat_time_mean)
                
                # 提取特征用于分类器训练
                if method == 'CSP':
                    extractor = CSP(n_components=config.n_components)
                elif method == 'FBCSP':
                    extractor = FBCSP(
                        m_filters=config.m_filters,
                        freq_bands=config.freq_bands,
                        sampling_rate=250
                    )
                elif method == 'AWFBCSP':
                    extractor = AWFBCSP(
                        m_filters=config.m_filters,
                        sampling_rate=250,
                        use_adaptive_weights=True,
                        use_temporal_windows=False,
                        use_erd_features=False,
                        use_multiscale=False
                    )
                
                extractor.fit(X_train, y_train)
                X_train_feat = extractor.transform(X_train)
                
                # 测量分类器训练时间
                for clf_name in config.classifiers.keys():
                    clf_time_mean, clf_time_std = measure_classifier_training_time(
                        X_train_feat, y_train, clf_name, repeat=config.n_repeats
                    )
                    fold_results[method]['clf_time'][clf_name].append(clf_time_mean)
            
            print(f"   ✓ Fold {fold_idx+1}/{config.n_folds}")
        
        # 计算平均时间并保存结果
        for method in ['CSP', 'FBCSP', 'AWFBCSP']:
            avg_feat_time = np.mean(fold_results[method]['feat_time'])
            std_feat_time = np.std(fold_results[method]['feat_time'])
            
            for clf_name in config.classifiers.keys():
                avg_clf_time = np.mean(fold_results[method]['clf_time'][clf_name])
                std_clf_time = np.std(fold_results[method]['clf_time'][clf_name])
                total_time = avg_feat_time + avg_clf_time
                
                results.append({
                    'Dataset': dataset_name,
                    'Subject': f'{dataset_name}0{subject_id}',
                    'Method': method,
                    'Classifier': clf_name,
                    'Feature_Extraction_Time': avg_feat_time,
                    'Feature_Extraction_Std': std_feat_time,
                    'Classifier_Training_Time': avg_clf_time,
                    'Classifier_Training_Std': std_clf_time,
                    'Total_Training_Time': total_time
                })
        
        print(f"   ✅ 完成")

# ============================================================================
# 保存结果
# ============================================================================
df_results = pd.DataFrame(results)
df_results.to_csv('results/training_time_measurements.csv', index=False)

print("\n" + "="*80)
print("📊 时间测量摘要")
print("="*80)

# 按数据集和方法汇总
summary = df_results.groupby(['Dataset', 'Method']).agg({
    'Feature_Extraction_Time': ['mean', 'std'],
    'Classifier_Training_Time': ['mean', 'std'],
    'Total_Training_Time': ['mean', 'std']
}).round(4)

print("\n▶ 特征提取时间 (秒):")
print(summary['Feature_Extraction_Time'])

print("\n▶ 分类器训练时间 (秒):")
print(summary['Classifier_Training_Time'])

print("\n▶ 总训练时间 (秒):")
print(summary['Total_Training_Time'])

print("\n" + "="*80)
print("✅ 结果已保存到: results/training_time_measurements.csv")
print("="*80)

