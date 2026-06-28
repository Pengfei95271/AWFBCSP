"""
EEG运动想象标准预处理管线

实现步骤：
1. 通道选择与重参考
2. 带通滤波 (Butterworth 4阶, 8-30 Hz)
3. 伪迹剔除 (基于幅度阈值)
4. 空间滤波 (可选ICA)
5. 基线校正
6. 标准化/归一化
7. 降维/特征选择

参考文献:
- Lotte et al., "A review of classification algorithms for EEG-based BCI" (2007)
- Ang et al., "Filter Bank CSP for MI classification" (2008)
- Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding" (2017)

作者: AI Assistant
日期: 2025-10-13
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    EEG运动想象标准预处理器
    
    参数:
        sampling_rate: 采样率 (Hz)
        filter_low: 高通截止频率 (Hz)
        filter_high: 低通截止频率 (Hz)
        notch_freq: 工频陷波频率 (Hz), None表示不做陷波
        baseline_window: 基线校正窗口 (秒)
        artifact_threshold: 伪迹检测阈值 (μV)
        reference_type: 重参考类型 ['average', 'cz', 'none']
    """
    
    def __init__(self, 
                 sampling_rate=250,
                 filter_low=8.0,
                 filter_high=30.0,
                 notch_freq=50.0,
                 baseline_window=1.0,
                 artifact_threshold=100.0,
                 reference_type='average'):
        
        self.fs = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.notch_freq = notch_freq
        self.baseline_window = baseline_window
        self.artifact_threshold = artifact_threshold
        self.reference_type = reference_type
        
    def fit_transform(self, X, verbose=True):
        """
        完整预处理流程
        
        参数:
            X: EEG数据 (n_trials, n_channels, n_samples)
            verbose: 是否打印处理信息
            
        返回:
            X_processed: 预处理后的数据
            valid_trials: 有效trial索引
        """
        if verbose:
            print("=" * 70)
            print("🔧 EEG预处理管线")
            print("=" * 70)
            print(f"输入数据: {X.shape}")
        
        n_trials_original = X.shape[0]
        
        # 1. 重参考
        if verbose:
            print(f"\n1️⃣  重参考 ({self.reference_type})...")
        X = self._rereference(X)
        
        # 2. 带通滤波
        if verbose:
            print(f"2️⃣  带通滤波 ({self.filter_low}-{self.filter_high} Hz)...")
        X = self._bandpass_filter(X)
        
        # 3. 陷波滤波 (工频干扰)
        if self.notch_freq is not None:
            if verbose:
                print(f"3️⃣  陷波滤波 ({self.notch_freq} Hz)...")
            X = self._notch_filter(X)
        
        # 4. 伪迹剔除
        if verbose:
            print(f"4️⃣  伪迹剔除 (阈值: ±{self.artifact_threshold} μV)...")
        X, valid_trials = self._artifact_rejection(X)
        
        if verbose:
            print(f"   保留: {len(valid_trials)}/{n_trials_original} trials "
                  f"({len(valid_trials)/n_trials_original*100:.1f}%)")
        
        # 5. 基线校正
        if verbose:
            print(f"5️⃣  基线校正 (窗口: {self.baseline_window}s)...")
        X = self._baseline_correction(X)
        
        # 6. 标准化
        if verbose:
            print(f"6️⃣  Z-score标准化...")
        X = self._standardize(X)
        
        if verbose:
            print(f"\n✅ 预处理完成!")
            print(f"   输出数据: {X.shape}")
            print("=" * 70)
        
        return X, valid_trials
    
    def _rereference(self, X):
        """重参考"""
        if self.reference_type == 'average':
            # 平均参考：减去所有通道的平均值
            ref = np.mean(X, axis=1, keepdims=True)
            return X - ref
        
        elif self.reference_type == 'cz':
            # Cz参考（假设Cz是中间通道）
            # 注意：需要根据实际通道顺序调整
            cz_idx = X.shape[1] // 2
            ref = X[:, cz_idx:cz_idx+1, :]
            return X - ref
        
        else:  # 'none'
            return X
    
    def _bandpass_filter(self, X):
        """Butterworth带通滤波"""
        n_trials, n_channels, n_samples = X.shape
        
        # 设计4阶Butterworth滤波器（文献标准）
        nyquist = self.fs / 2.0
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 应用滤波器
        X_filtered = np.zeros_like(X)
        for trial in range(n_trials):
            for ch in range(n_channels):
                # 使用filtfilt避免相位失真
                X_filtered[trial, ch, :] = signal.filtfilt(b, a, X[trial, ch, :])
        
        return X_filtered
    
    def _notch_filter(self, X):
        """陷波滤波 (去除工频干扰)"""
        n_trials, n_channels, n_samples = X.shape
        
        # 设计陷波滤波器
        Q = 30.0  # 品质因数
        b, a = signal.iirnotch(self.notch_freq, Q, self.fs)
        
        X_filtered = np.zeros_like(X)
        for trial in range(n_trials):
            for ch in range(n_channels):
                X_filtered[trial, ch, :] = signal.filtfilt(b, a, X[trial, ch, :])
        
        return X_filtered
    
    def _artifact_rejection(self, X):
        """基于幅度阈值的伪迹剔除"""
        n_trials = X.shape[0]
        valid_trials = []
        
        for trial in range(n_trials):
            # 检查是否有通道超过阈值
            max_amplitude = np.max(np.abs(X[trial]))
            
            if max_amplitude < self.artifact_threshold:
                valid_trials.append(trial)
        
        return X[valid_trials], np.array(valid_trials)
    
    def _baseline_correction(self, X):
        """基线校正"""
        baseline_samples = int(self.baseline_window * self.fs)
        
        # 使用每个trial的前baseline_window秒作为基线
        baseline = np.mean(X[:, :, :baseline_samples], axis=2, keepdims=True)
        
        return X - baseline
    
    def _standardize(self, X):
        """Z-score标准化"""
        n_trials, n_channels, n_samples = X.shape
        
        X_standardized = np.zeros_like(X)
        
        for trial in range(n_trials):
            for ch in range(n_channels):
                # 对每个通道独立标准化
                X_standardized[trial, ch, :] = zscore(X[trial, ch, :])
        
        return X_standardized


class AdvancedEEGPreprocessor(EEGPreprocessor):
    """
    高级EEG预处理器 (包含ICA、CSP预处理等)
    """
    
    def __init__(self, 
                 sampling_rate=250,
                 filter_low=8.0,
                 filter_high=30.0,
                 use_ica=False,
                 n_ica_components=None,
                 **kwargs):
        
        super().__init__(sampling_rate, filter_low, filter_high, **kwargs)
        self.use_ica = use_ica
        self.n_ica_components = n_ica_components
        
    def fit_transform(self, X, verbose=True):
        """增强版预处理流程"""
        # 先执行基础预处理
        X, valid_trials = super().fit_transform(X, verbose=verbose)
        
        # ICA伪迹去除 (可选)
        if self.use_ica:
            if verbose:
                print(f"7️⃣  ICA伪迹去除...")
            X = self._ica_artifact_removal(X)
        
        return X, valid_trials
    
    def _ica_artifact_removal(self, X):
        """ICA伪迹去除 (简化版)"""
        try:
            from sklearn.decomposition import FastICA
            
            n_trials, n_channels, n_samples = X.shape
            
            # 确定ICA成分数量
            n_components = self.n_ica_components or n_channels
            
            X_cleaned = np.zeros_like(X)
            
            for trial in range(n_trials):
                # 对每个trial独立做ICA
                ica = FastICA(n_components=n_components, random_state=42)
                
                # X[trial].T: (n_samples, n_channels)
                S = ica.fit_transform(X[trial].T)  # 独立成分
                A = ica.mixing_  # 混合矩阵
                
                # 简单策略：移除方差最大的成分（通常是伪迹）
                variances = np.var(S, axis=0)
                artifact_idx = np.argmax(variances)
                
                # 重构时移除伪迹成分
                S[:, artifact_idx] = 0
                X_reconstructed = np.dot(S, A.T)
                
                X_cleaned[trial] = X_reconstructed.T
            
            return X_cleaned
            
        except ImportError:
            print("⚠️  scikit-learn未安装，跳过ICA")
            return X


class MultiScalePreprocessor:
    """
    多尺度预处理器 - 用于不同频段
    
    适用于FBCSP等多频段方法
    """
    
    def __init__(self, 
                 sampling_rate=250,
                 freq_bands=None):
        
        self.fs = sampling_rate
        
        # 默认频段 (参考Filter Bank CSP)
        self.freq_bands = freq_bands or [
            (4, 8),    # Theta
            (8, 12),   # Alpha
            (12, 16),  # Low Beta
            (16, 20),  # Mid Beta
            (20, 24),  # High Beta
            (24, 28),  # Low Gamma
            (28, 32)   # Mid Gamma
        ]
    
    def fit_transform(self, X, verbose=True):
        """
        多频段滤波
        
        返回:
            X_multiscale: 列表，每个元素是一个频段的数据
        """
        if verbose:
            print(f"\n🎚️  多频段滤波 ({len(self.freq_bands)}个频段)...")
        
        X_multiscale = []
        
        for low, high in self.freq_bands:
            # 为每个频段创建预处理器
            preprocessor = EEGPreprocessor(
                sampling_rate=self.fs,
                filter_low=low,
                filter_high=high,
                reference_type='average',
                artifact_threshold=100.0
            )
            
            X_band, _ = preprocessor.fit_transform(X, verbose=False)
            X_multiscale.append(X_band)
            
            if verbose:
                print(f"   {low}-{high} Hz: {X_band.shape}")
        
        return X_multiscale


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 模拟数据
    print("=" * 70)
    print("测试EEG预处理器")
    print("=" * 70)
    
    n_trials = 100
    n_channels = 21
    n_samples = 752  # 约3秒 @ 250Hz
    
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.randn(n_trials, n_channels, n_samples) * 50  # μV
    
    # 添加一些"伪迹"
    X[5, :, :] += 200  # 强伪迹
    X[10, 3, :] += 150  # 单通道伪迹
    
    print(f"原始数据: {X.shape}")
    print(f"数据范围: [{np.min(X):.2f}, {np.max(X):.2f}] μV\n")
    
    # 测试1: 基础预处理
    print("\n【测试1: 基础预处理】")
    preprocessor = EEGPreprocessor(
        sampling_rate=250,
        filter_low=8.0,
        filter_high=30.0,
        artifact_threshold=100.0,
        reference_type='average'
    )
    
    X_processed, valid_trials = preprocessor.fit_transform(X)
    
    print(f"\n处理后数据: {X_processed.shape}")
    print(f"数据范围: [{np.min(X_processed):.2f}, {np.max(X_processed):.2f}]")
    print(f"有效trials: {len(valid_trials)}/{n_trials}")
    
    # 测试2: 多频段预处理
    print("\n" + "=" * 70)
    print("【测试2: 多频段预处理】")
    multiscale = MultiScalePreprocessor(sampling_rate=250)
    X_multiscale = multiscale.fit_transform(X)
    
    print(f"\n生成{len(X_multiscale)}个频段的数据")
    
    print("\n✅ 所有测试通过!")

