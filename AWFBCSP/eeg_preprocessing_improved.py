"""
改进的EEG预处理模块
专为BCIC-IV-2A数据集优化

主要功能:
1. 带通滤波 (8-30 Hz) - 运动想象相关频段
2. 基线校正 - 去除直流偏移
3. 伪迹检测和去除 - 眼动、肌电等
4. 数据标准化 - 提高分类性能
5. 时间窗口选择 - 选择MI最明显的时间段

作者: AI Assistant
日期: 2025-01-27
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfilt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class EEGPreprocessor:
    """改进的EEG预处理类"""
    
    def __init__(self, sampling_rate=250, 
                 filter_band=(8, 30),  # 运动想象相关频段
                 notch_freq=50,        # 工频干扰
                 baseline_correction=True,
                 artifact_removal=True,
                 time_window=(0.5, 4.0),  # 选择MI最明显的时间段
                 standardize=True):
        """
        参数:
        ------
        sampling_rate : int
            采样率 (Hz)
        filter_band : tuple
            带通滤波频段 (low, high)
        notch_freq : int
            陷波滤波频率 (工频干扰)
        baseline_correction : bool
            是否进行基线校正
        artifact_removal : bool
            是否进行伪迹去除
        time_window : tuple
            时间窗口 (start, end) 秒
        standardize : bool
            是否标准化
        """
        self.sampling_rate = sampling_rate
        self.filter_band = filter_band
        self.notch_freq = notch_freq
        self.baseline_correction = baseline_correction
        self.artifact_removal = artifact_removal
        self.time_window = time_window
        self.standardize = standardize
        
        # 设计滤波器
        self._design_filters()
        
    def _design_filters(self):
        """设计滤波器"""
        nyquist = self.sampling_rate / 2
        
        # 带通滤波器 (8-30 Hz)
        low = self.filter_band[0] / nyquist
        high = self.filter_band[1] / nyquist
        self.bandpass_sos = butter(4, [low, high], btype='band', output='sos')
        
        # 陷波滤波器 (50 Hz工频干扰)
        if self.notch_freq:
            notch_freq_norm = self.notch_freq / nyquist
            # iirnotch 不支持 output='sos'，使用默认的 (b, a) 格式
            self.notch_b, self.notch_a = iirnotch(notch_freq_norm, Q=30)
        
    def preprocess(self, X):
        """
        预处理EEG数据
        
        参数:
        ------
        X : array-like, shape (n_trials, n_channels, n_samples)
            原始EEG数据
            
        返回:
        ------
        X_processed : array-like, shape (n_trials, n_channels, n_samples_processed)
            预处理后的EEG数据
        """
        X = np.array(X, dtype=np.float64)
        n_trials, n_channels, n_samples = X.shape
        
        print(f"   [预处理] 开始处理 {n_trials} trials, {n_channels} 通道, {n_samples} 时间点")
        
        # 1. 时间窗口选择
        if self.time_window:
            start_sample = int(self.time_window[0] * self.sampling_rate)
            end_sample = int(self.time_window[1] * self.sampling_rate)
            end_sample = min(end_sample, n_samples)
            X = X[:, :, start_sample:end_sample]
            n_samples = X.shape[2]
            print(f"   [预处理] 时间窗口选择: {self.time_window[0]}-{self.time_window[1]}s")
        
        # 2. 基线校正
        if self.baseline_correction:
            X = self._baseline_correction(X)
            print(f"   [预处理] 基线校正完成")
        
        # 3. 带通滤波
        X = self._bandpass_filter(X)
        print(f"   [预处理] 带通滤波 ({self.filter_band[0]}-{self.filter_band[1]}Hz) 完成")
        
        # 4. 陷波滤波 (去除工频干扰)
        if self.notch_freq:
            X = self._notch_filter(X)
            print(f"   [预处理] 陷波滤波 ({self.notch_freq}Hz) 完成")
        
        # 5. 伪迹检测和去除
        if self.artifact_removal:
            X = self._artifact_removal(X)
            print(f"   [预处理] 伪迹去除完成")
        
        # 6. 数据标准化
        if self.standardize:
            X = self._standardize_data(X)
            print(f"   [预处理] 数据标准化完成")
        
        print(f"   [预处理] 预处理完成! 输出形状: {X.shape}")
        return X
    
    def _baseline_correction(self, X):
        """基线校正 - 去除直流偏移"""
        # 使用前0.5秒作为基线
        baseline_samples = int(0.5 * self.sampling_rate)
        baseline_samples = min(baseline_samples, X.shape[2])
        
        if baseline_samples > 0:
            baseline = np.mean(X[:, :, :baseline_samples], axis=2, keepdims=True)
            X = X - baseline
        
        return X
    
    def _bandpass_filter(self, X):
        """带通滤波"""
        n_trials, n_channels, n_samples = X.shape
        X_filtered = np.zeros_like(X)
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                X_filtered[trial_idx, ch_idx, :] = sosfilt(
                    self.bandpass_sos, X[trial_idx, ch_idx, :])
        
        return X_filtered
    
    def _notch_filter(self, X):
        """陷波滤波 - 去除工频干扰"""
        n_trials, n_channels, n_samples = X.shape
        X_filtered = np.zeros_like(X)
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                X_filtered[trial_idx, ch_idx, :] = filtfilt(
                    self.notch_b, self.notch_a, X[trial_idx, ch_idx, :])
        
        return X_filtered
    
    def _artifact_removal(self, X):
        """伪迹检测和去除"""
        n_trials, n_channels, n_samples = X.shape
        X_clean = X.copy()
        
        for trial_idx in range(n_trials):
            trial_data = X[trial_idx, :, :]  # (channels, samples)
            
            # 1. 检测异常值 (使用Z-score)
            z_scores = np.abs(zscore(trial_data, axis=1))
            outlier_mask = z_scores > 3  # 3倍标准差
            
            # 2. 检测眼动伪迹 (前额通道)
            if n_channels >= 4:
                frontal_channels = [0, 1, 2, 3]  # 假设前4个是前额通道
                frontal_data = trial_data[frontal_channels, :]
                frontal_std = np.std(frontal_data, axis=1)
                
                # 如果前额通道标准差过大，可能是眼动
                if np.any(frontal_std > 2 * np.mean(frontal_std)):
                    # 对前额通道进行平滑
                    for ch in frontal_channels:
                        X_clean[trial_idx, ch, :] = self._smooth_signal(
                            X_clean[trial_idx, ch, :])
            
            # 3. 检测肌电伪迹 (高频噪声)
            high_freq_power = np.var(np.diff(trial_data, axis=1), axis=1)
            if np.any(high_freq_power > 2 * np.mean(high_freq_power)):
                # 对异常通道进行平滑
                for ch_idx in range(n_channels):
                    if high_freq_power[ch_idx] > 2 * np.mean(high_freq_power):
                        X_clean[trial_idx, ch_idx, :] = self._smooth_signal(
                            X_clean[trial_idx, ch_idx, :])
        
        return X_clean
    
    def _smooth_signal(self, signal, window_size=5):
        """信号平滑"""
        if len(signal) < window_size:
            return signal
        
        # 使用移动平均
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed
    
    def _standardize_data(self, X):
        """数据标准化"""
        n_trials, n_channels, n_samples = X.shape
        
        # 对每个通道进行标准化
        for ch_idx in range(n_channels):
            channel_data = X[:, ch_idx, :].flatten()
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            if std_val > 0:
                X[:, ch_idx, :] = (X[:, ch_idx, :] - mean_val) / std_val
        
        return X
    
    def get_preprocessing_info(self):
        """返回预处理信息"""
        info = {
            'sampling_rate': self.sampling_rate,
            'filter_band': self.filter_band,
            'notch_freq': self.notch_freq,
            'baseline_correction': self.baseline_correction,
            'artifact_removal': self.artifact_removal,
            'time_window': self.time_window,
            'standardize': self.standardize
        }
        return info

def quick_preprocess(X, sampling_rate=250):
    """
    快速预处理函数
    
    参数:
    ------
    X : array-like, shape (n_trials, n_channels, n_samples)
        原始EEG数据
    sampling_rate : int
        采样率
        
    返回:
    ------
    X_processed : array-like
        预处理后的数据
    """
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        filter_band=(8, 30),
        notch_freq=50,
        baseline_correction=True,
        artifact_removal=True,
        time_window=(0.5, 4.0),
        standardize=True
    )
    
    return preprocessor.preprocess(X)

if __name__ == "__main__":
    # 测试预处理
    print("🧪 测试EEG预处理模块...")
    
    # 生成模拟数据
    n_trials = 10
    n_channels = 22
    n_samples = 1000
    sampling_rate = 250
    
    # 模拟EEG数据 (包含噪声)
    X = np.random.randn(n_trials, n_channels, n_samples) * 0.1
    X += np.sin(2 * np.pi * 10 * np.arange(n_samples) / sampling_rate) * 0.05  # 10Hz信号
    X += np.sin(2 * np.pi * 50 * np.arange(n_samples) / sampling_rate) * 0.02  # 50Hz工频干扰
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始数据范围: [{X.min():.3f}, {X.max():.3f}]")
    
    # 预处理
    X_processed = quick_preprocess(X, sampling_rate)
    
    print(f"预处理后形状: {X_processed.shape}")
    print(f"预处理后范围: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
    print("✅ 预处理测试完成!")
