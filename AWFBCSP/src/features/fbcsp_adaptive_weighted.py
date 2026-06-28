"""
Adaptive Weighted Filter Bank CSP (AWFBCSP)
自适应加权滤波器组CSP - 针对运动想象优化

创新点：
1. 自适应频段权重学习 - 不同频段对分类的贡献不同
2. 时间窗口分段分析 - 捕捉MI的时间演化特性
3. ERD/ERS特征增强 - 运动想象特有的能量变化模式
4. 频段互信息优化 - 自动选择最佳频段组合
5. 多尺度特征融合 - 粗粒度+细粒度频段
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from .csp import CSP
from .base_feature import BaseFeature


class AdaptiveWeightedFBCSP(BaseFeature):
    """
    自适应加权FBCSP - 专为运动想象设计
    
    与传统FBCSP的区别：
    - 频段权重自适应学习（基于互信息）
    - 时间窗口分段特征提取
    - ERD/ERS能量特征
    - 多尺度频段组合
    """
    
    def __init__(self, m_filters=3, sampling_rate=250, 
                 freq_bands=None,              # 新增：允许外部传入频段
                 use_adaptive_weights=True,
                 use_temporal_windows=False,   # 默认关闭，减少复杂度
                 use_erd_features=False,       # 默认关闭
                 use_multiscale=False):        # 默认关闭
        """
        参数:
        ------
        m_filters : int
            每个频段的CSP滤波器对数
        sampling_rate : int
            采样率
        freq_bands : list of tuples, optional
            自定义频段列表，如 [(8,12), (12,16), ...]
            如果不提供，使用默认的5个或7个频段
        use_adaptive_weights : bool
            是否使用自适应频段权重
        use_temporal_windows : bool
            是否使用时间窗口分段
        use_erd_features : bool
            是否提取ERD/ERS特征
        use_multiscale : bool
            是否使用多尺度频段
        """
        super().__init__()
        self.m_filters = m_filters
        self.sampling_rate = sampling_rate
        self.use_adaptive_weights = use_adaptive_weights
        self.use_temporal_windows = use_temporal_windows
        self.use_erd_features = use_erd_features
        self.use_multiscale = use_multiscale
        
        # 定义频段（针对运动想象优化）
        # 粗粒度频段 - 捕捉主要节律
        self.coarse_bands = [
            (8, 12),   # Mu节律
            (12, 30),  # Beta节律
        ]
        
        # 细粒度频段 - 捕捉细节变化（默认6个频段）
        self.fine_bands = [
            (8, 12),   # Alpha/Mu
            (12, 16),  # Low Beta
            (16, 20),  # Mid Beta  
            (20, 24),  # High Beta
            (24, 30),  # Low Gamma
        ]
        
        # 优先使用外部传入的频段
        if freq_bands is not None:
            self.freq_bands = list(freq_bands)
        elif self.use_multiscale:
            self.freq_bands = self.coarse_bands + self.fine_bands
        else:
            self.freq_bands = self.fine_bands
        
        self.n_fbanks = len(self.freq_bands)
        
        # 时间窗口设置（运动想象通常在1-3秒最明显）
        self.temporal_windows = [
            (0.0, 1.0),    # 早期
            (1.0, 2.5),    # 中期（MI最强）
            (2.5, 4.0),    # 后期
        ]
        
        # 训练后的参数
        self.fbcsp_filters_multi = []
        self.band_weights = None  # 频段权重
        self.baseline_power = None  # 基线功率（用于ERD/ERS）
        
    def fit(self, X, y):
        """训练自适应加权FBCSP"""
        X = self._validate_input(X)
        y = np.asarray(y)
        
        print(f"   [AWFBCSP] 训练自适应加权FBCSP...")
        
        # 1. 准备多频段数据
        x_train_fb = self._prepare_multiband_data(X)
        
        # 2. 学习频段权重（基于互信息）
        if self.use_adaptive_weights:
            print(f"   [AWFBCSP] 学习频段重要性权重...")
            self.band_weights = self._compute_band_weights(x_train_fb, y)
            print(f"   [AWFBCSP] 频段权重: {self.band_weights}")
        else:
            self.band_weights = np.ones(self.n_fbanks) / self.n_fbanks
        
        # 3. 计算基线功率（用于ERD/ERS）
        if self.use_erd_features:
            self.baseline_power = self._compute_baseline_power(X)
        
        # 4. 训练CSP滤波器
        y_classes_unique = np.unique(y)
        n_classes = len(y_classes_unique)
        
        self.csp = CSP(n_components=self.m_filters * 2)
        
        def get_csp(x_train_fb, y_train_cls):
            """为每个频段训练CSP"""
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters
        
        # 为每个类别训练
        self.fbcsp_filters_multi = []
        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y))
            
            fbcsp_filters = get_csp(x_train_fb, y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)
        
        self.is_fitted = True
        print(f"   [AWFBCSP] 训练完成！")
        return self
    
    def transform(self, X, class_idx=0):
        """提取增强特征（改进版）"""
        self._check_is_fitted()
        X = self._validate_input(X)
        
        n_trials = X.shape[0]
        feature_list = []
        
        # 1. 基础FBCSP特征（带权重）
        fbcsp_features = self._extract_weighted_fbcsp_features(X, class_idx)
        feature_list.append(fbcsp_features)
        
        # 2. 时间窗口特征
        if self.use_temporal_windows:
            temporal_features = self._extract_temporal_features(X, class_idx)
            feature_list.append(temporal_features)
        
        # 3. ERD/ERS特征
        if self.use_erd_features:
            erd_features = self._extract_erd_features(X)
            feature_list.append(erd_features)
        
        # 4. 添加频段间的交互特征（只用高权重频段）
        if self.use_adaptive_weights:
            interaction_features = self._extract_band_interaction_features(X, class_idx)
            feature_list.append(interaction_features)
        
        # 合并所有特征
        all_features = np.concatenate(feature_list, axis=1)
        
        return all_features
    
    def _extract_weighted_fbcsp_features(self, X, class_idx):
        """提取加权的FBCSP特征（改进版）"""
        x_data = self._prepare_multiband_data(X)
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        
        # 提取特征并应用权重
        x_features = np.zeros((n_trials, self.m_filters * 2 * self.n_fbanks), dtype=float)
        
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            
            for k in range(n_trials):
                x_trial = np.copy(x_data[i, k, :, :])
                csp_feat = self.csp._transform_single_trial(x_trial, eig_vectors)
                
                # 改进：使用平方根权重（避免过度抑制低权重频段）
                weight_factor = np.sqrt(self.band_weights[i])
                weighted_feat = csp_feat * weight_factor
                
                for j in range(self.m_filters * 2):
                    x_features[k, i * self.m_filters * 2 + j] = weighted_feat[j]
        
        return x_features
    
    def _extract_temporal_features(self, X, class_idx):
        """提取时间窗口特征（简化版 - 只用关键窗口）"""
        n_trials = X.shape[0]
        features_per_window = self.m_filters * 2 * self.n_fbanks
        
        # 只使用MI最强的时间窗口（1-2.5s）
        start_sample = int(1.0 * self.sampling_rate)
        end_sample = int(2.5 * self.sampling_rate)
        
        if end_sample > X.shape[2]:
            end_sample = X.shape[2]
        
        X_window = X[:, :, start_sample:end_sample]
        
        # 提取FBCSP特征
        temporal_features = self._extract_weighted_fbcsp_features(X_window, class_idx)
        
        return temporal_features
    
    def _extract_erd_features(self, X):
        """
        提取ERD/ERS特征
        ERD (Event-Related Desynchronization): 能量降低
        ERS (Event-Related Synchronization): 能量增加
        """
        n_trials, n_channels, n_samples = X.shape
        n_bands = len(self.freq_bands)
        
        # 为每个频段计算ERD/ERS
        erd_features = np.zeros((n_trials, n_bands * 4))  # 4个统计量
        
        for band_idx, freq_band in enumerate(self.freq_bands):
            # 滤波
            X_filtered = self._filter_data(X, freq_band)
            
            # 计算瞬时功率（使用Hilbert变换）
            for trial_idx in range(n_trials):
                trial_powers = []
                for ch_idx in range(n_channels):
                    analytic_signal = hilbert(X_filtered[trial_idx, ch_idx, :])
                    instantaneous_power = np.abs(analytic_signal) ** 2
                    trial_powers.append(instantaneous_power)
                
                # 平均所有通道
                avg_power = np.mean(trial_powers, axis=0)
                
                # 归一化（相对于基线）
                if self.baseline_power is not None:
                    baseline = self.baseline_power[band_idx]
                    erd_ers = (avg_power - baseline) / baseline
                else:
                    erd_ers = avg_power
                
                # 提取统计特征
                erd_features[trial_idx, band_idx * 4 + 0] = np.mean(erd_ers)      # 平均ERD/ERS
                erd_features[trial_idx, band_idx * 4 + 1] = np.std(erd_ers)       # 标准差
                erd_features[trial_idx, band_idx * 4 + 2] = np.min(erd_ers)       # 最小值（最大ERD）
                erd_features[trial_idx, band_idx * 4 + 3] = np.max(erd_ers)       # 最大值（最大ERS）
        
        return erd_features
    
    def _compute_band_weights(self, x_train_fb, y):
        """
        计算频段权重（改进版 - 使用多个特征）
        互信息越高，说明该频段对分类越重要
        """
        n_fbanks = x_train_fb.shape[0]
        n_trials = x_train_fb.shape[1]
        
        mi_scores = np.zeros(n_fbanks)
        
        for band_idx in range(n_fbanks):
            # 提取该频段的多个统计特征
            band_data = x_train_fb[band_idx, :, :, :]  # (trials, channels, samples)
            
            # 提取更丰富的特征
            features_list = []
            
            # 1. 平均功率
            power = np.mean(band_data ** 2, axis=(1, 2))
            features_list.append(power)
            
            # 2. 方差
            variance = np.var(band_data, axis=(1, 2))
            features_list.append(variance)
            
            # 3. 半球侧化指数（左右手MI的关键）
            if band_data.shape[1] >= 10:  # 确保有足够的通道
                # BNCI2014数据集：C3≈索引7, Cz≈索引9, C4≈索引11
                # 计算左右半球功率差异
                left_channels = [6, 7, 8]  # C3及周围
                right_channels = [10, 11, 12]  # C4及周围
                
                left_power = np.mean(band_data[:, left_channels, :] ** 2, axis=(1, 2))
                right_power = np.mean(band_data[:, right_channels, :] ** 2, axis=(1, 2))
                
                # 侧化指数：(左-右)/(左+右)
                laterality = (left_power - right_power) / (left_power + right_power + 1e-10)
                features_list.append(laterality)
                
                # 4. Mu/Beta抑制比率（运动想象特有）
                # 低频段（Mu）应该被抑制，Beta可能增强
                if band_idx < 2:  # Mu频段
                    mu_suppression = -power  # 负值表示抑制
                    features_list.append(mu_suppression)
            
            # 合并特征
            band_features = np.column_stack(features_list)
            
            # 计算与标签的平均互信息
            mi_values = mutual_info_classif(band_features, y, random_state=42)
            mi = np.mean(mi_values)
            mi_scores[band_idx] = mi
        
        # 标准化互信息分数
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-10)
        
        # 归一化为权重（使用温度参数的Softmax）
        # 温度越低，权重差异越大
        temperature = 0.5  # 降低温度使权重分布更有区分度
        weights = np.exp(mi_scores / temperature) / np.sum(np.exp(mi_scores / temperature))
        
        return weights
    
    def _compute_baseline_power(self, X):
        """计算基线功率（用于ERD/ERS计算）"""
        # 使用前0.5秒作为基线
        baseline_samples = int(0.5 * self.sampling_rate)
        X_baseline = X[:, :, :baseline_samples]
        
        baseline_powers = []
        
        for freq_band in self.freq_bands:
            # 滤波
            X_filtered = self._filter_data(X_baseline, freq_band)
            
            # 计算平均功率
            power = np.mean(X_filtered ** 2)
            baseline_powers.append(power)
        
        return np.array(baseline_powers)
    
    def _prepare_multiband_data(self, X):
        """准备多频段数据"""
        n_trials, n_channels, n_samples = X.shape
        n_fbanks = len(self.freq_bands)
        
        x_train_fb = np.zeros((n_fbanks, n_trials, n_channels, n_samples))
        
        for band_idx, freq_band in enumerate(self.freq_bands):
            X_filtered = self._filter_data(X, freq_band)
            x_train_fb[band_idx, :, :, :] = X_filtered
        
        return x_train_fb
    
    def _filter_data(self, X, freq_band):
        """带通滤波"""
        low, high = freq_band
        nyquist = self.sampling_rate / 2
        
        b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
        
        n_trials, n_channels, n_samples = X.shape
        X_filtered = np.zeros_like(X)
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                X_filtered[trial_idx, ch_idx, :] = filtfilt(
                    b, a, X[trial_idx, ch_idx, :])
        
        return X_filtered
    
    def _extract_band_interaction_features(self, X, class_idx):
        """
        提取频段间交互特征
        高权重频段之间的协同作用
        """
        # 选择权重最高的两个频段
        top_bands = np.argsort(self.band_weights)[-2:]
        
        x_data = self._prepare_multiband_data(X)
        n_trials = x_data.shape[1]
        
        # 提取两个高权重频段的特征
        band1_data = x_data[top_bands[0], :, :, :]
        band2_data = x_data[top_bands[1], :, :, :]
        
        interaction_features = np.zeros((n_trials, 4))
        
        for k in range(n_trials):
            # 计算两个频段的功率
            power1 = np.mean(band1_data[k, :, :] ** 2)
            power2 = np.mean(band2_data[k, :, :] ** 2)
            
            # 交互特征
            interaction_features[k, 0] = power1 * power2  # 乘积
            interaction_features[k, 1] = power1 + power2  # 和
            interaction_features[k, 2] = abs(power1 - power2)  # 差
            interaction_features[k, 3] = power1 / (power2 + 1e-10)  # 比率
        
        return interaction_features
    
    def get_feature_importance(self):
        """返回频段重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        importance_dict = {}
        for idx, band in enumerate(self.freq_bands):
            band_name = f"{band[0]}-{band[1]}Hz"
            importance_dict[band_name] = float(self.band_weights[idx])
        
        return importance_dict

