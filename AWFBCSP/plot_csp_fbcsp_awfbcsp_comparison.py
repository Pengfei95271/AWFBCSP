"""
CSP vs FBCSP vs AWFBCSP 空间模式对比可视化

展示三种方法的空间模式差异：
- CSP: 单频段（8-30Hz）
- FBCSP: 多频段等权重
- AWFBCSP: 多频段自适应权重
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
import sys
import os

sys.path.insert(0, 'src')

from src.features.csp import CSP
from src.features.fbcsp import FBCSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP

# ==================== BCI IV 2a 22通道位置 ====================
CHANNEL_NAMES_22 = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
    'P1', 'Pz', 'P2', 'POz'
]

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


def load_and_preprocess(subject_id):
    """加载并预处理BCI IV 2a数据"""
    data_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_data.npy'
    label_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_label.npy'
    
    X = np.load(data_path)
    y = np.load(label_path)
    
    # 二分类（左右手：标签1和2）
    binary_mask = (y == 1) | (y == 2)
    X = X[binary_mask]
    y = y[binary_mask] - 1  # 转换为0/1
    
    # 基线校正
    sampling_rate = 250
    baseline_samples = int(0.5 * sampling_rate)
    X = X - X[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    
    return X, y


def filter_data(X, freq_band, sampling_rate=250):
    """带通滤波"""
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
    """计算CSP空间模式"""
    csp = CSP(n_components=n_components)
    csp.fit(X_filtered, y)
    patterns = csp.patterns_  # (n_channels, n_components)
    return patterns, csp.eig_values


def interpolate_topomap(values, chan_pos, labels, grid_res=100, radius=1.0, power=2.0):
    """反距离加权插值"""
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
        if np.sqrt(gx**2 + gy**2) > radius:
            Zf[i] = np.nan
            continue
        d[d < 1e-6] = 1e-6
        w = 1.0 / (d**power)
        Zf[i] = np.sum(w * values) / np.sum(w)
    
    Z = Zf.reshape(X.shape)
    return X, Y, Z


def plot_single_topomap(ax, values, chan_pos, labels, title, vmin=None, vmax=None, 
                        cmap='RdBu_r', show_colorbar=False):
    """绘制单个地形图"""
    xs = np.array([chan_pos[ch][0] for ch in labels])
    ys = np.array([chan_pos[ch][1] for ch in labels])
    
    X, Y, Z = interpolate_topomap(values, chan_pos, labels, grid_res=150, radius=1.0, power=2.0)
    
    # 填色
    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.contour(X, Y, Z, levels=10, colors='k', linewidths=0.2, alpha=0.4)
    
    # 头皮轮廓
    head = plt.Circle((0, 0), 1.0, edgecolor='k', facecolor='none', linewidth=1.5)
    ax.add_patch(head)
    
    # 鼻子
    nose = plt.Polygon([[-0.06, 1.0], [0.06, 1.0], [0.00, 1.1]], 
                       closed=True, edgecolor='k', facecolor='k')
    ax.add_patch(nose)
    
    # 耳朵
    ear_left = plt.Circle((-1.03, 0.0), 0.06, edgecolor='k', facecolor='none', linewidth=0.8)
    ear_right = plt.Circle((1.03, 0.0), 0.06, edgecolor='k', facecolor='none', linewidth=0.8)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # 电极位置
    ax.scatter(xs, ys, s=15, color='k', zorder=5)
    
    # 标注关键通道
    for ch, x, y in zip(labels, xs, ys):
        if ch in ['C3', 'C4']:
            ax.annotate(ch, (x, y), xytext=(2, 2), textcoords='offset points', 
                       fontsize=6, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
    
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.2)
    
    return cf


def main():
    """主函数"""
    print("=" * 70)
    print("📊 CSP vs FBCSP vs AWFBCSP 空间模式对比")
    print("=" * 70)
    
    # 配置
    subject_id = 1
    sampling_rate = 250
    freq_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    
    print(f"\n📌 被试: A{subject_id:02d}")
    print(f"📌 频段: {freq_bands}")
    
    # ==================== 加载数据 ====================
    print("\n🔄 加载数据...")
    X_raw, y = load_and_preprocess(subject_id)
    print(f"   数据形状: {X_raw.shape}, 标签: {np.bincount(y)}")
    
    # ==================== 1. CSP（单频段8-30Hz） ====================
    print("\n🔄 计算CSP（8-30Hz单频段）...")
    X_broadband = filter_data(X_raw, (8, 30), sampling_rate)
    csp_patterns, csp_eig = compute_csp_patterns(X_broadband, y)
    
    # CSP空间模式：第一列(左手)，最后一列(右手)
    csp_left = csp_patterns[:, 0]
    csp_right = csp_patterns[:, -1]
    
    print(f"   CSP特征值范围: {np.real(csp_eig).min():.3f} ~ {np.real(csp_eig).max():.3f}")
    
    # ==================== 2. FBCSP（多频段等权重） ====================
    print("\n🔄 计算FBCSP（多频段等权重）...")
    
    # 对每个频段计算CSP模式
    fbcsp_band_patterns = []
    fbcsp_band_eig = []
    
    for band in freq_bands:
        X_band = filter_data(X_raw, band, sampling_rate)
        patterns, eig_vals = compute_csp_patterns(X_band, y)
        fbcsp_band_patterns.append(patterns)
        fbcsp_band_eig.append(eig_vals)
        print(f"   频段 {band[0]:2d}-{band[1]:2d}Hz: 特征值范围 {np.real(eig_vals).min():.3f} ~ {np.real(eig_vals).max():.3f}")
    
    # FBCSP: 等权重平均各频段模式
    equal_weights = np.ones(len(freq_bands)) / len(freq_bands)
    
    fbcsp_left = np.zeros(22)
    fbcsp_right = np.zeros(22)
    for i, patterns in enumerate(fbcsp_band_patterns):
        fbcsp_left += equal_weights[i] * patterns[:, 0]
        fbcsp_right += equal_weights[i] * patterns[:, -1]
    
    print(f"   FBCSP权重（等权重）: {equal_weights}")
    
    # ==================== 3. AWFBCSP（自适应权重） ====================
    print("\n🔄 计算AWFBCSP（自适应权重）...")
    
    # 使用真实的AWFBCSP来获取权重
    awfbcsp = AdaptiveWeightedFBCSP(
        m_filters=3,
        sampling_rate=sampling_rate,
        use_adaptive_weights=True,
        use_temporal_windows=False,
        use_erd_features=False,
        use_multiscale=False
    )
    # 临时修改频段
    awfbcsp.freq_bands = freq_bands
    awfbcsp.n_fbanks = len(freq_bands)
    
    awfbcsp.fit(X_raw, y)
    adaptive_weights = awfbcsp.band_weights
    
    print(f"   AWFBCSP自适应权重: {adaptive_weights}")
    
    # 使用自适应权重计算加权平均模式
    awfbcsp_left = np.zeros(22)
    awfbcsp_right = np.zeros(22)
    for i, patterns in enumerate(fbcsp_band_patterns):
        awfbcsp_left += adaptive_weights[i] * patterns[:, 0]
        awfbcsp_right += adaptive_weights[i] * patterns[:, -1]
    
    # ==================== 绘图 ====================
    print("\n🎨 绘制对比图...")
    
    # 统一色标
    all_vals = np.concatenate([
        csp_left, csp_right, 
        fbcsp_left, fbcsp_right, 
        awfbcsp_left, awfbcsp_right
    ])
    vlim = np.max(np.abs(all_vals))
    vmin, vmax = -vlim, vlim
    
    # 创建图形：3行2列
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 0.3], width_ratios=[1, 1, 0.05])
    
    # 第一行：CSP
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    plot_single_topomap(ax1, csp_left, CHAN_POS_22, CHANNEL_NAMES_22, 
                        'CSP Pattern 1\n(Left Hand, 8-30Hz)', vmin, vmax)
    plot_single_topomap(ax2, csp_right, CHAN_POS_22, CHANNEL_NAMES_22, 
                        'CSP Pattern 2\n(Right Hand, 8-30Hz)', vmin, vmax)
    
    # 第二行：FBCSP
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    plot_single_topomap(ax3, fbcsp_left, CHAN_POS_22, CHANNEL_NAMES_22, 
                        'FBCSP Pattern 1\n(Left Hand, Equal Weights)', vmin, vmax)
    plot_single_topomap(ax4, fbcsp_right, CHAN_POS_22, CHANNEL_NAMES_22, 
                        'FBCSP Pattern 2\n(Right Hand, Equal Weights)', vmin, vmax)
    
    # 第三行：AWFBCSP
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    
    plot_single_topomap(ax5, awfbcsp_left, CHAN_POS_22, CHANNEL_NAMES_22, 
                        'AWFBCSP Pattern 1\n(Left Hand, Adaptive Weights)', vmin, vmax)
    cf = plot_single_topomap(ax6, awfbcsp_right, CHAN_POS_22, CHANNEL_NAMES_22, 
                             'AWFBCSP Pattern 2\n(Right Hand, Adaptive Weights)', vmin, vmax)
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[:3, 2])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cbar_ax, label='Spatial Pattern Amplitude')
    
    # 第四行：权重对比条形图
    ax_weights = fig.add_subplot(gs[3, :2])
    
    x_pos = np.arange(len(freq_bands))
    width = 0.35
    
    bars1 = ax_weights.bar(x_pos - width/2, equal_weights, width, label='FBCSP (Equal)', 
                           color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax_weights.bar(x_pos + width/2, adaptive_weights, width, label='AWFBCSP (Adaptive)', 
                           color='coral', alpha=0.7, edgecolor='black')
    
    ax_weights.set_xlabel('Frequency Band', fontsize=10, fontweight='bold')
    ax_weights.set_ylabel('Weight', fontsize=10, fontweight='bold')
    ax_weights.set_title('Band Weights Comparison', fontsize=11, fontweight='bold')
    ax_weights.set_xticks(x_pos)
    ax_weights.set_xticklabels([f'{b[0]}-{b[1]}Hz' for b in freq_bands])
    ax_weights.legend(loc='upper right')
    ax_weights.grid(axis='y', alpha=0.3)
    
    # 标注最高权重
    max_idx = np.argmax(adaptive_weights)
    ax_weights.annotate(f'Max: {adaptive_weights[max_idx]:.2f}', 
                       xy=(max_idx + width/2, adaptive_weights[max_idx]),
                       xytext=(max_idx + 0.5, adaptive_weights[max_idx] + 0.05),
                       fontsize=9, fontweight='bold', color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # 总标题
    fig.suptitle(f'CSP vs FBCSP vs AWFBCSP Spatial Patterns Comparison\n'
                 f'(BCI Competition IV 2a, Subject A{subject_id:02d})',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 添加方法说明文字
    fig.text(0.02, 0.92, '① CSP: Single broadband (8-30Hz)', fontsize=9, 
             fontweight='bold', color='darkblue', transform=fig.transFigure)
    fig.text(0.02, 0.64, '② FBCSP: Multi-band with equal weights', fontsize=9, 
             fontweight='bold', color='darkblue', transform=fig.transFigure)
    fig.text(0.02, 0.36, '③ AWFBCSP: Multi-band with adaptive weights', fontsize=9, 
             fontweight='bold', color='darkred', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0.03, 0.02, 0.98, 0.95])
    
    # 保存
    output_path = 'results/csp_fbcsp_awfbcsp_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"\n✅ 图片已保存: {output_path}")
    
    # ==================== 额外：各频段模式详细图 ====================
    print("\n🎨 绘制各频段模式详细图...")
    
    fig2, axes = plt.subplots(2, len(freq_bands), figsize=(2.5*len(freq_bands), 6))
    
    for i, (band, patterns) in enumerate(zip(freq_bands, fbcsp_band_patterns)):
        # 左手模式
        plot_single_topomap(axes[0, i], patterns[:, 0], CHAN_POS_22, CHANNEL_NAMES_22,
                           f'{band[0]}-{band[1]}Hz\nLeft Hand', vmin, vmax)
        # 右手模式
        plot_single_topomap(axes[1, i], patterns[:, -1], CHAN_POS_22, CHANNEL_NAMES_22,
                           f'{band[0]}-{band[1]}Hz\nRight Hand', vmin, vmax)
        
        # 显示权重
        axes[0, i].text(0, -1.0, f'w={adaptive_weights[i]:.2f}', 
                       fontsize=8, ha='center', fontweight='bold',
                       color='red' if adaptive_weights[i] == max(adaptive_weights) else 'black')
    
    fig2.suptitle(f'CSP Patterns for Each Frequency Band\n'
                  f'(with AWFBCSP adaptive weights shown below)',
                  fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    fig2.colorbar(sm, cax=cbar_ax2, label='Pattern Amplitude')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    output_path2 = 'results/csp_patterns_per_band.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 各频段详细图已保存: {output_path2}")
    
    plt.show()
    
    # ==================== 打印总结 ====================
    print("\n" + "=" * 70)
    print("📊 方法对比总结:")
    print("=" * 70)
    print(f"\n{'方法':<15} {'特点':<50}")
    print("-" * 70)
    print(f"{'CSP':<15} {'单频段(8-30Hz)，可能丢失频率信息':<50}")
    print(f"{'FBCSP':<15} {'多频段等权重，不区分频段重要性':<50}")
    print(f"{'AWFBCSP':<15} {'多频段自适应权重，强调重要频段':<50}")
    
    print(f"\n📌 AWFBCSP学习到的频段权重:")
    for band, weight in zip(freq_bands, adaptive_weights):
        importance = "⭐⭐⭐" if weight == max(adaptive_weights) else ("⭐⭐" if weight > 0.15 else "⭐")
        print(f"   {band[0]:2d}-{band[1]:2d}Hz: {weight:.4f} {importance}")


if __name__ == "__main__":
    main()

