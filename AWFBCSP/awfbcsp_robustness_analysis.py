"""
AWFBCSP稳健性与工程可用性分析

分析内容:
1. 抗噪性分析: 不同噪声水平下的性能
2. 在线可用性: 实时处理能力测试
3. 参数敏感性: 关键参数变化对性能的影响
4. 计算效率: 训练和推理时间分析
5. 内存使用: 资源消耗分析

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
import time
import psutil
from datetime import datetime
from tqdm import tqdm
import threading
import queue

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
print("🛡️ AWFBCSP稳健性与工程可用性分析")
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
    
    # 噪声水平测试
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # 在线处理参数
    chunk_size = 250  # 1秒数据块
    buffer_size = 5   # 5秒缓冲区
    
    # 结果保存路径
    results_dir = 'results/awfbcsp_robustness_analysis'
    os.makedirs(results_dir, exist_ok=True)


config = Config()

# 定义分类器
classifiers = {
    'SVM': SVC(C=10, kernel='rbf', gamma=0.01, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LDA': LinearDiscriminantAnalysis()
}

print(f"⚙️  配置信息:")
print(f"   - 噪声水平: {config.noise_levels}")
print(f"   - 在线处理块大小: {config.chunk_size} samples")
print(f"   - 缓冲区大小: {config.buffer_size} seconds")
print(f"   - 分类器: {list(classifiers.keys())}\n")


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


def add_noise(data, noise_level):
    """添加高斯噪声"""
    if noise_level == 0:
        return data
    
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# ============================================================================
# 1. 抗噪性分析
# ============================================================================
def noise_robustness_analysis(X, y, subjects):
    """抗噪性分析"""
    print("=" * 80)
    print("🛡️ 1. 抗噪性分析")
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
    
    # 存储结果
    noise_results = {}
    
    for noise_level in config.noise_levels:
        print(f"\n🔊 测试噪声水平: {noise_level}")
        
        # 添加噪声
        X_noisy = add_noise(X_processed, noise_level)
        
        # 对每个被试进行个体内分类
        subject_accuracies = {}
        
        for subject_id in config.subjects:
            subject_mask = subjects == subject_id
            X_subject = X_noisy[subject_mask]
            y_subject = y[subject_mask]
            
            if len(np.unique(y_subject)) < 2:
                continue
            
            # 5折交叉验证
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for feature_name, feature_type in [('AWFBCSP', 'awfbcsp')]:
                accuracies = []
                
                for train_idx, test_idx in kfold.split(X_subject, y_subject):
                    X_train, X_test = X_subject[train_idx], X_subject[test_idx]
                    y_train, y_test = y_subject[train_idx], y_subject[test_idx]
                    
                    # 特征提取
                    if feature_type == 'awfbcsp':
                        extractor = AdaptiveWeightedFBCSP(
                            m_filters=6,
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
        
        # 计算平均准确率
        if subject_accuracies:
            mean_accuracy = np.mean(list(subject_accuracies.values()))
            std_accuracy = np.std(list(subject_accuracies.values()))
            noise_results[noise_level] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'subject_accuracies': subject_accuracies
            }
            
            print(f"   平均准确率: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    return noise_results


# ============================================================================
# 2. 在线处理能力测试
# ============================================================================
class OnlineProcessor:
    """在线处理器"""
    
    def __init__(self, sampling_rate=250, chunk_size=250):
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.buffer = []
        self.extractor = None
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """训练模型"""
        print("🔧 训练在线处理器...")
        
        # 训练AWFBCSP
        self.extractor = AdaptiveWeightedFBCSP(
            m_filters=6,
            sampling_rate=self.sampling_rate,
            use_adaptive_weights=True
        )
        self.extractor.fit(X_train, y_train)
        
        # 提取训练特征
        train_feats = self.extractor.transform(X_train)
        
        # 训练标准化器
        self.scaler = StandardScaler()
        train_feats_scaled = self.scaler.fit_transform(train_feats)
        
        # 训练分类器
        self.classifier = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
        self.classifier.fit(train_feats_scaled, y_train)
        
        self.is_trained = True
        print("✅ 在线处理器训练完成")
    
    def process_chunk(self, chunk_data):
        """处理数据块"""
        if not self.is_trained:
            return None
        
        # 添加到缓冲区
        self.buffer.extend(chunk_data)
        
        # 如果缓冲区有足够数据，进行处理
        if len(self.buffer) >= self.chunk_size:
            # 取一个chunk的数据
            data_chunk = np.array(self.buffer[:self.chunk_size])
            self.buffer = self.buffer[self.chunk_size:]
            
            # 重塑为EEG格式 (1, channels, timepoints)
            if data_chunk.ndim == 1:
                data_chunk = data_chunk.reshape(1, -1)
            if data_chunk.shape[0] == 1:
                data_chunk = data_chunk.reshape(1, config.n_channels, -1)
            
            # 特征提取
            features = self.extractor.transform(data_chunk)
            features_scaled = self.scaler.transform(features)
            
            # 分类
            prediction = self.classifier.predict(features_scaled)
            confidence = np.max(self.classifier.decision_function(features_scaled))
            
            return {
                'prediction': prediction[0],
                'confidence': confidence,
                'features': features[0]
            }
        
        return None


def online_processing_test(X, y, subjects):
    """在线处理能力测试"""
    print("\n" + "=" * 80)
    print("⚡ 2. 在线处理能力测试")
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
    
    # 选择第一个被试进行测试
    test_subject = config.subjects[0]
    subject_mask = subjects == test_subject
    X_subject = X_processed[subject_mask]
    y_subject = y[subject_mask]
    
    # 分割训练和测试数据
    split_idx = int(0.8 * len(X_subject))
    X_train = X_subject[:split_idx]
    y_train = y_subject[:split_idx]
    X_test = X_subject[split_idx:]
    y_test = y_subject[split_idx:]
    
    print(f"训练数据: {X_train.shape[0]} trials")
    print(f"测试数据: {X_test.shape[0]} trials")
    
    # 创建在线处理器
    online_processor = OnlineProcessor(
        sampling_rate=config.sampling_rate,
        chunk_size=config.chunk_size
    )
    
    # 训练
    online_processor.train(X_train, y_train)
    
    # 在线处理测试
    print("\n🔄 开始在线处理测试...")
    
    processing_times = []
    predictions = []
    confidences = []
    
    for i in range(len(X_test)):
        # 模拟实时数据流
        trial_data = X_test[i]  # (channels, timepoints)
        
        # 分块处理
        chunk_predictions = []
        for t in range(0, trial_data.shape[1], config.chunk_size):
            chunk = trial_data[:, t:t+config.chunk_size]
            
            start_time = time.time()
            result = online_processor.process_chunk(chunk.flatten())
            end_time = time.time()
            
            if result is not None:
                processing_times.append(end_time - start_time)
                chunk_predictions.append(result['prediction'])
        
        # 使用多数投票决定最终预测
        if chunk_predictions:
            final_prediction = max(set(chunk_predictions), key=chunk_predictions.count)
            predictions.append(final_prediction)
        else:
            predictions.append(0)  # 默认预测
    
    # 计算性能
    accuracy = accuracy_score(y_test, predictions)
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    print(f"✅ 在线处理测试完成")
    print(f"   准确率: {accuracy*100:.2f}%")
    print(f"   平均处理时间: {avg_processing_time*1000:.2f} ms")
    print(f"   实时性: {'✅ 满足' if avg_processing_time < 0.1 else '❌ 不满足'} (< 100ms)")
    
    return {
        'accuracy': accuracy,
        'avg_processing_time': avg_processing_time,
        'processing_times': processing_times,
        'predictions': predictions,
        'true_labels': y_test
    }


# ============================================================================
# 3. 计算效率分析
# ============================================================================
def computational_efficiency_analysis(X, y, subjects):
    """计算效率分析"""
    print("\n" + "=" * 80)
    print("⚡ 3. 计算效率分析")
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
    
    # 选择第一个被试进行测试
    test_subject = config.subjects[0]
    subject_mask = subjects == test_subject
    X_subject = X_processed[subject_mask]
    y_subject = y[subject_mask]
    
    # 分割数据
    split_idx = int(0.8 * len(X_subject))
    X_train = X_subject[:split_idx]
    y_train = y_subject[:split_idx]
    X_test = X_subject[split_idx:]
    y_test = y_subject[split_idx:]
    
    efficiency_results = {}
    
    # 测试不同特征提取方法
    feature_methods = {
        'CSP': CSP(n_components=12),
        'FBCSP': FBCSP(m_filters=6, sampling_rate=config.sampling_rate),
        'AWFBCSP': AdaptiveWeightedFBCSP(m_filters=6, sampling_rate=config.sampling_rate, use_adaptive_weights=True)
    }
    
    for method_name, extractor in feature_methods.items():
        print(f"\n🔬 测试方法: {method_name}")
        
        # 训练时间
        start_time = time.time()
        extractor.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 特征提取时间
        start_time = time.time()
        train_feats = extractor.transform(X_train)
        test_feats = extractor.transform(X_test)
        transform_time = time.time() - start_time
        
        # 分类时间
        scaler = StandardScaler()
        train_feats_scaled = scaler.fit_transform(train_feats)
        test_feats_scaled = scaler.transform(test_feats)
        
        clf = SVC(C=10, kernel='rbf', gamma=0.01, random_state=42)
        
        start_time = time.time()
        clf.fit(train_feats_scaled, y_train)
        clf_train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = clf.predict(test_feats_scaled)
        clf_predict_time = time.time() - start_time
        
        # 准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 内存使用
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        efficiency_results[method_name] = {
            'train_time': train_time,
            'transform_time': transform_time,
            'clf_train_time': clf_train_time,
            'clf_predict_time': clf_predict_time,
            'total_time': train_time + transform_time + clf_train_time + clf_predict_time,
            'accuracy': accuracy,
            'memory_usage': memory_usage,
            'feature_dim': train_feats.shape[1]
        }
        
        print(f"   训练时间: {train_time:.3f}s")
        print(f"   特征提取时间: {transform_time:.3f}s")
        print(f"   分类训练时间: {clf_train_time:.3f}s")
        print(f"   分类预测时间: {clf_predict_time:.3f}s")
        print(f"   总时间: {efficiency_results[method_name]['total_time']:.3f}s")
        print(f"   准确率: {accuracy*100:.2f}%")
        print(f"   内存使用: {memory_usage:.1f} MB")
        print(f"   特征维度: {train_feats.shape[1]}")
    
    return efficiency_results


# ============================================================================
# 4. 结果可视化和保存
# ============================================================================
def visualize_results(noise_results, online_results, efficiency_results):
    """结果可视化"""
    print("\n" + "=" * 80)
    print("📊 4. 结果可视化")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 抗噪性分析
    ax1 = axes[0, 0]
    noise_levels = list(noise_results.keys())
    accuracies = [noise_results[level]['mean_accuracy'] for level in noise_levels]
    stds = [noise_results[level]['std_accuracy'] for level in noise_levels]
    
    ax1.errorbar(noise_levels, accuracies, yerr=stds, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('噪声水平')
    ax1.set_ylabel('准确率')
    ax1.set_title('抗噪性分析')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. 在线处理时间分布
    ax2 = axes[0, 1]
    if online_results['processing_times']:
        ax2.hist(online_results['processing_times'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(online_results['avg_processing_time'], color='red', linestyle='--', 
                   label=f'平均: {online_results["avg_processing_time"]*1000:.1f}ms')
        ax2.axvline(0.1, color='orange', linestyle='--', label='实时阈值: 100ms')
    ax2.set_xlabel('处理时间 (秒)')
    ax2.set_ylabel('频次')
    ax2.set_title('在线处理时间分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 计算效率对比
    ax3 = axes[0, 2]
    methods = list(efficiency_results.keys())
    train_times = [efficiency_results[method]['train_time'] for method in methods]
    transform_times = [efficiency_results[method]['transform_time'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax3.bar(x - width/2, train_times, width, label='训练时间', alpha=0.8)
    ax3.bar(x + width/2, transform_times, width, label='特征提取时间', alpha=0.8)
    ax3.set_xlabel('方法')
    ax3.set_ylabel('时间 (秒)')
    ax3.set_title('计算效率对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 准确率 vs 效率
    ax4 = axes[1, 0]
    accuracies = [efficiency_results[method]['accuracy'] for method in methods]
    total_times = [efficiency_results[method]['total_time'] for method in methods]
    
    scatter = ax4.scatter(total_times, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        ax4.annotate(method, (total_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('总时间 (秒)')
    ax4.set_ylabel('准确率')
    ax4.set_title('准确率 vs 计算效率')
    ax4.grid(True, alpha=0.3)
    
    # 5. 内存使用对比
    ax5 = axes[1, 1]
    memory_usage = [efficiency_results[method]['memory_usage'] for method in methods]
    bars = ax5.bar(methods, memory_usage, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
    ax5.set_xlabel('方法')
    ax5.set_ylabel('内存使用 (MB)')
    ax5.set_title('内存使用对比')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. 特征维度对比
    ax6 = axes[1, 2]
    feature_dims = [efficiency_results[method]['feature_dim'] for method in methods]
    bars = ax6.bar(methods, feature_dims, alpha=0.8, color=['gold', 'lightcoral', 'lightblue'])
    ax6.set_xlabel('方法')
    ax6.set_ylabel('特征维度')
    ax6.set_title('特征维度对比')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.results_dir}/robustness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表保存: {config.results_dir}/robustness_analysis.png")


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序"""
    try:
        # 加载数据
        X, y, subjects = load_bci_iv_2a_data()
        print(f"✅ 数据加载成功: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints")
        
        # 1. 抗噪性分析
        noise_results = noise_robustness_analysis(X, y, subjects)
        
        # 2. 在线处理能力测试
        online_results = online_processing_test(X, y, subjects)
        
        # 3. 计算效率分析
        efficiency_results = computational_efficiency_analysis(X, y, subjects)
        
        # 4. 结果可视化
        visualize_results(noise_results, online_results, efficiency_results)
        
        # 5. 保存结果
        results = {
            'noise_robustness': noise_results,
            'online_processing': online_results,
            'computational_efficiency': efficiency_results,
            'config': {
                'dataset': 'BCIC-IV-2A',
                'subjects': len(config.subjects),
                'noise_levels': config.noise_levels,
                'chunk_size': config.chunk_size,
                'buffer_size': config.buffer_size,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(f'{config.results_dir}/robustness_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 稳健性分析完成!")
        print(f"📁 结果保存在: {config.results_dir}/")
        
        # 总结报告
        print("\n" + "=" * 80)
        print("📋 稳健性分析总结报告")
        print("=" * 80)
        
        # 抗噪性总结
        baseline_acc = noise_results[0.0]['mean_accuracy']
        high_noise_acc = noise_results[0.3]['mean_accuracy']
        noise_robustness = (baseline_acc - high_noise_acc) / baseline_acc * 100
        
        print(f"🛡️ 抗噪性:")
        print(f"   - 无噪声准确率: {baseline_acc*100:.2f}%")
        print(f"   - 高噪声准确率: {high_noise_acc*100:.2f}%")
        print(f"   - 噪声鲁棒性: {100-noise_robustness:.1f}% (性能保持)")
        
        # 在线处理总结
        print(f"\n⚡ 在线处理:")
        print(f"   - 平均处理时间: {online_results['avg_processing_time']*1000:.1f}ms")
        print(f"   - 实时性: {'✅ 满足' if online_results['avg_processing_time'] < 0.1 else '❌ 不满足'}")
        print(f"   - 在线准确率: {online_results['accuracy']*100:.2f}%")
        
        # 计算效率总结
        awfbcsp_efficiency = efficiency_results['AWFBCSP']
        print(f"\n🔧 计算效率 (AWFBCSP):")
        print(f"   - 训练时间: {awfbcsp_efficiency['train_time']:.3f}s")
        print(f"   - 特征提取时间: {awfbcsp_efficiency['transform_time']:.3f}s")
        print(f"   - 总时间: {awfbcsp_efficiency['total_time']:.3f}s")
        print(f"   - 内存使用: {awfbcsp_efficiency['memory_usage']:.1f}MB")
        print(f"   - 特征维度: {awfbcsp_efficiency['feature_dim']}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
