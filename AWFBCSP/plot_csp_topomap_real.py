"""
使用真实CSP空间模式绘制脑地形图
用于可视化AWFBCSP中高权重和低权重子频带的CSP空间模式差异
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import sys
import os

sys.path.insert(0, os.path.join('src'))

from src.features.csp import CSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP

# ==================== BCI IV 2a 22通道位置（10-20系统） ====================
# 通道顺序：Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, 
#          CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
CHANNEL_NAMES_22 = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
    'P1', 'Pz', 'P2', 'POz'
]

# 22通道的大致2D位置（单位化头皮坐标）
CHAN_POS_22 = {
    'Fz':  ( 0.00,  0.55),
    'FC3': (-0.55,  0.30), 'FC1': (-0.25,  0.30), 'FCz': ( 0.00,  0.30), 
    'FC2': ( 0.25,  0.30), 'FC4': ( 0.55,  0.30),
    'C5':  (-0.85,  0.00), 'C3':  (-0.55,  0.00), 'C1':  (-0.25,  0.00), 
    'Cz':  ( 0.00,  0.00), 'C2':  ( 0.25,  0.00), 'C4':  ( 0.55,  0.00), 
    'C6':  ( 0.85,  0.00),
    'CP3': (-0.55, -0.30), 'CP1': (-0.25, -0.30), 'CPz': ( 0.00, -0.30), 
    'CP2': ( 0.25, -0.30), 'CP4': ( 0.55, -0.30),
    'P1':  (-0.25, -0.55), 'Pz':  ( 0.00, -0.60), 'P2':  ( 0.25, -0.55), 
    'POz': ( 0.00, -0.80),
}

# ==================== 3通道（BCI IV 2b）位置 ====================
CHANNEL_NAMES_3 = ['C3', 'Cz', 'C4']
CHAN_POS_3 = {
    'C3': (-0.55, 0.0),
    'Cz': ( 0.00, 0.0),
    'C4': ( 0.55, 0.0),
}


def load_and_preprocess(subject_id, dataset='2a'):
    """加载并预处理数据"""
    if dataset == '2a':
        data_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_data.npy'
        label_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_label.npy'
    else:
        data_path = f'dataset/bci_iv_2b/raw/B{subject_id:02d}T_data.npy'
        label_path = f'dataset/bci_iv_2b/raw/B{subject_id:02d}T_label.npy'
    
    X = np.load(data_path)
    y = np.load(label_path)
    
    # 只取二分类数据（左右手：标签1和2）
    binary_mask = (y == 1) | (y == 2)
    X = X[binary_mask]
    y = y[binary_mask] - 1  # 转换为0/1
    
    # 预处理：基线校正 + 带通滤波
    sampling_rate = 250
    baseline_samples = int(0.5 * sampling_rate)
    X = X - X[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    
    # 8-30Hz带通滤波
    nyquist = sampling_rate / 2
    b, a = butter(4, [8/nyquist, 30/nyquist], btype='band')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, :] = filtfilt(b, a, X[i, j, :])
    
    return X, y


def filter_data(X, freq_band, sampling_rate=250):
    """对数据进行带通滤波"""
    low, high = freq_band
    nyquist = sampling_rate / 2
    b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
    
    n_trials, n_channels, n_samples = X.shape
    X_filtered = np.zeros_like(X)
    
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            X_filtered[trial_idx, ch_idx, :] = filtfilt(b, a, X[trial_idx, ch_idx, :])
    
    return X_filtered


def compute_csp_patterns(X_filtered, y, n_components=6):
    """计算CSP空间模式（patterns）"""
    csp = CSP(n_components=n_components)
    csp.fit(X_filtered, y)
    
    # 空间模式 = 滤波器的伪逆
    # patterns_ 已经在CSP类中计算了
    patterns = csp.patterns_  # shape: (n_channels, n_components)
    filters = csp.filters_    # shape: (n_components, n_channels)
    eig_values = csp.eig_values
    
    return patterns, filters, eig_values


def interpolate_topomap(values, chan_pos, labels, grid_res=100, radius=1.0, power=2.0):
    """反距离加权插值，生成头皮网格"""
    xs = np.array([chan_pos[ch][0] for ch in labels])
    ys = np.array([chan_pos[ch][1] for ch in labels])
    
    xi = np.linspace(-radius, radius, grid_res)
    yi = np.linspace(-radius, radius, grid_res)
    X, Y = np.meshgrid(xi, yi)
    
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = np.zeros_like(Xf)
    
    for i, (gx, gy) in enumerate(zip(Xf, Yf)):
        d = np.sqrt((gx - xs)**2 + (gy - ys)**2)
        # 头外面设为 NaN
        if np.sqrt(gx**2 + gy**2) > radius:
            Zf[i] = np.nan
            continue
        d[d < 1e-6] = 1e-6
        w = 1.0 / (d**power)
        Zf[i] = np.sum(w * values) / np.sum(w)
    
    Z = Zf.reshape(X.shape)
    return X, Y, Z


def plot_topomap(ax, values, chan_pos, labels, title, vmin=None, vmax=None, cmap='RdBu_r'):
    """绘制单个地形图"""
    xs = np.array([chan_pos[ch][0] for ch in labels])
    ys = np.array([chan_pos[ch][1] for ch in labels])
    
    X, Y, Z = interpolate_topomap(values, chan_pos, labels, grid_res=200, radius=1.0, power=2.0)
    
    # 填色等高线
    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    # 等高线轮廓
    ax.contour(X, Y, Z, levels=10, colors='k', linewidths=0.3, alpha=0.5)
    
    # 头皮轮廓
    head = plt.Circle((0, 0), 1.0, edgecolor='k', facecolor='none', linewidth=1.5)
    ax.add_patch(head)
    
    # 鼻子
    nose_y = 1.0
    nose = plt.Polygon(
        [[-0.08, nose_y], [0.08, nose_y], [0.00, nose_y + 0.12]],
        closed=True, edgecolor='k', facecolor='k'
    )
    ax.add_patch(nose)
    
    # 耳朵
    ear_left = plt.Circle((-1.05, 0.0), 0.08, edgecolor='k', facecolor='none', linewidth=1.0)
    ear_right = plt.Circle((1.05, 0.0), 0.08, edgecolor='k', facecolor='none', linewidth=1.0)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # 电极位置和名称
    ax.scatter(xs, ys, s=30, color='k', zorder=5)
    for ch, x, y in zip(labels, xs, ys):
        # 只标注关键通道
        if ch in ['C3', 'Cz', 'C4', 'CP3', 'CP4', 'FC3', 'FC4']:
            ax.annotate(ch, (x, y), xytext=(3, 3), textcoords='offset points', 
                       fontsize=7, fontweight='bold')
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.3)
    
    return cf


def main():
    """主函数：提取真实CSP空间模式并绘制地形图"""
    
    print("=" * 70)
    print("📊 使用真实CSP空间模式绘制地形图")
    print("=" * 70)
    
    # ==================== 选择数据集 ====================
    # 2a有22通道，2b只有3通道，建议使用2a
    dataset = '2a'
    subject_id = 1
    
    if dataset == '2a':
        chan_names = CHANNEL_NAMES_22
        chan_pos = CHAN_POS_22
        n_channels = 22
    else:
        chan_names = CHANNEL_NAMES_3
        chan_pos = CHAN_POS_3
        n_channels = 3
    
    print(f"\n📌 数据集: BCI Competition IV Dataset {dataset}")
    print(f"📌 被试: {subject_id}")
    print(f"📌 通道数: {n_channels}")
    
    # ==================== 加载数据 ====================
    print("\n🔄 加载数据...")
    X, y = load_and_preprocess(subject_id, dataset=dataset)
    print(f"   数据形状: {X.shape}")
    print(f"   标签分布: {np.bincount(y)}")
    
    # ==================== 定义频段 ====================
    # 模拟AWFBCSP中的频段
    freq_bands = [
        (8, 12),   # Alpha/Mu
        (12, 16),  # Low Beta  (通常高权重)
        (16, 20),  # Mid Beta
        (20, 24),  # High Beta
        (24, 28),  # Low Gamma (通常低权重)
    ]
    
    # ==================== 计算每个频段的CSP模式和"重要性" ====================
    print("\n🔄 计算各频段CSP空间模式...")
    
    band_patterns = {}
    band_importance = {}
    
    for band in freq_bands:
        # 滤波到特定频段
        X_band = filter_data(X, band)
        
        # 计算CSP模式
        patterns, filters, eig_values = compute_csp_patterns(X_band, y, n_components=6)
        
        # 计算"重要性"（基于特征值的离散程度）
        # 特征值越极端，说明该频段对分类越有判别力
        eig_real = np.real(eig_values)
        importance = np.std(eig_real) / (np.mean(np.abs(eig_real)) + 1e-10)
        
        band_patterns[band] = {'patterns': patterns, 'filters': filters, 'eig_values': eig_values}
        band_importance[band] = importance
        
        print(f"   频段 {band[0]}-{band[1]}Hz: 重要性={importance:.4f}")
    
    # ==================== 找出高权重和低权重频段 ====================
    sorted_bands = sorted(band_importance.items(), key=lambda x: x[1], reverse=True)
    high_weight_band = sorted_bands[0][0]
    low_weight_band = sorted_bands[-1][0]
    
    print(f"\n📌 高权重频段: {high_weight_band[0]}-{high_weight_band[1]}Hz (重要性={band_importance[high_weight_band]:.4f})")
    print(f"📌 低权重频段: {low_weight_band[0]}-{low_weight_band[1]}Hz (重要性={band_importance[low_weight_band]:.4f})")
    
    # ==================== 提取用于绘图的空间模式 ====================
    # CSP模式第一列对应最大特征值（类别1特有），最后一列对应最小特征值（类别2特有）
    high_patterns = band_patterns[high_weight_band]['patterns']
    low_patterns = band_patterns[low_weight_band]['patterns']
    
    # CSP 1: 第一个pattern（对类别1最敏感）
    # CSP 2: 最后一个pattern（对类别2最敏感）
    csp1_high = high_patterns[:, 0]
    csp2_high = high_patterns[:, -1]
    csp1_low = low_patterns[:, 0]
    csp2_low = low_patterns[:, -1]
    
    # ==================== 绘制2x2地形图 ====================
    print("\n🎨 绘制地形图...")
    
    patterns_to_plot = [
        (f"High-weight Band ({high_weight_band[0]}-{high_weight_band[1]}Hz) — CSP 1 (Left Hand)", csp1_high),
        (f"High-weight Band ({high_weight_band[0]}-{high_weight_band[1]}Hz) — CSP 2 (Right Hand)", csp2_high),
        (f"Low-weight Band ({low_weight_band[0]}-{low_weight_band[1]}Hz) — CSP 1 (Left Hand)", csp1_low),
        (f"Low-weight Band ({low_weight_band[0]}-{low_weight_band[1]}Hz) — CSP 2 (Right Hand)", csp2_low),
    ]
    
    # 统一色标范围
    all_vals = np.concatenate([p[1] for p in patterns_to_plot])
    vlim = np.max(np.abs(all_vals))
    vmin, vmax = -vlim, vlim
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, (title, vals) in zip(axes.ravel(), patterns_to_plot):
        plot_topomap(ax, vals, chan_pos, chan_names, title, vmin=vmin, vmax=vmax)
    
    fig.suptitle(
        f'Real CSP Spatial Patterns: High-weight vs Low-weight Sub-bands\n'
        f'(BCI IV 2a, Subject {subject_id})',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # 共享 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cbar_ax, label='CSP Spatial Pattern Amplitude')
    
    plt.tight_layout(rect=[0.03, 0.05, 0.90, 0.95])
    
    # 保存
    output_path = 'results/awfbcsp_csp_topomap_real.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"\n✅ 地形图已保存: {output_path}")
    
    plt.show()
    
    # ==================== 额外：打印频段权重排序 ====================
    print("\n" + "=" * 50)
    print("📊 频段重要性排序（用于AWFBCSP权重参考）:")
    print("=" * 50)
    for i, (band, importance) in enumerate(sorted_bands, 1):
        print(f"  {i}. {band[0]:2d}-{band[1]:2d}Hz: {importance:.4f}")


if __name__ == "__main__":
    main()

