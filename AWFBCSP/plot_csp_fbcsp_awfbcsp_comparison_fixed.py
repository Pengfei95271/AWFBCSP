"""
CSP vs FBCSP vs AWFBCSP 空间模式对比可视化（修复版）

修复的关键问题：
1. ✅ 不再跨频段加权平均patterns（改为展示权重最大的频段）
2. ✅ 不假设patterns[:,0]是左手（改为Class 0/1 dominant）
3. ✅ 使用sosfiltfilt优化滤波性能
4. ✅ 添加频段生理标签（μ、β等）
5. ✅ 明确说明visualize的是patterns而非filters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt
import sys
import os

sys.path.insert(0, 'src')

from src.features.csp import CSP
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

# 频段生理标签
BAND_LABELS = {
    (8, 12): 'μ',
    (12, 16): 'low β',
    (16, 20): 'β',
    (20, 24): 'high β',
    (24, 30): 'high β',
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
    """
    带通滤波（使用sosfiltfilt优化性能和数值稳定性）
    
    Args:
        X: (n_trials, n_channels, n_samples)
        freq_band: (low, high) Hz
    """
    low, high = freq_band
    nyquist = sampling_rate / 2
    sos = butter(4, [low/nyquist, high/nyquist], btype='band', output='sos')
    
    # 直接对所有trials和channels进行滤波（axis=-1表示时间维度）
    X_filtered = sosfiltfilt(sos, X, axis=-1)
    
    return X_filtered


def compute_csp_patterns_with_class_check(X_filtered, y, n_components=6):
    """
    计算CSP空间模式，并确定哪个pattern对应哪个类别
    
    返回:
        patterns: (n_channels, n_components)
        class0_indices: 更偏向class 0的pattern索引
        class1_indices: 更偏向class 1的pattern索引
    """
    csp = CSP(n_components=n_components)
    csp.fit(X_filtered, y)
    
    # 计算CSP特征
    X_csp = csp.transform(X_filtered)
    
    # 对每个component，计算两类的方差比
    class0_mask = (y == 0)
    class1_mask = (y == 1)
    
    variance_ratios = []
    for i in range(n_components):
        var0 = np.var(X_csp[class0_mask, i])
        var1 = np.var(X_csp[class1_mask, i])
        # 方差比：>1表示更偏向class 0，<1表示更偏向class 1
        ratio = var0 / (var1 + 1e-10)
        variance_ratios.append(ratio)
    
    variance_ratios = np.array(variance_ratios)
    
    # 找出最偏向每个类的component
    class0_indices = np.argsort(variance_ratios)[-3:][::-1]  # 前3个最偏向class 0
    class1_indices = np.argsort(variance_ratios)[:3]  # 前3个最偏向class 1
    
    patterns = csp.patterns_
    
    return patterns, csp.eig_values, class0_indices, class1_indices, csp


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


def plot_topomap(ax, values, chan_pos, labels, title, vmin=None, vmax=None, 
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
    
    # 标注关键通道（C3/C4）
    for ch, x, y in zip(labels, xs, ys):
        if ch in ['C3', 'C4']:
            ax.annotate(ch, (x, y), xytext=(2, 2), textcoords='offset points', 
                       fontsize=7, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.2)
    
    return cf


def main():
    """主函数"""
    print("=" * 80)
    print("📊 CSP vs FBCSP vs AWFBCSP 空间模式对比（修复版）")
    print("=" * 80)
    
    # 配置
    subject_id = 1
    sampling_rate = 250
    freq_bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    
    print(f"\n📌 被试: A{subject_id:02d}")
    print(f"📌 频段: {freq_bands}")
    print(f"📌 可视化对象: CSP spatial patterns (用于解释传感器级贡献)")
    
    # ==================== 加载数据 ====================
    print("\n🔄 加载数据...")
    X_raw, y = load_and_preprocess(subject_id)
    print(f"   数据形状: {X_raw.shape}")
    print(f"   类别分布: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
    
    # ==================== 1. CSP（单频段8-30Hz） ====================
    print("\n🔄 计算CSP（8-30Hz单频段）...")
    X_broadband = filter_data(X_raw, (8, 30), sampling_rate)
    patterns_csp, eig_csp, c0_idx, c1_idx, csp_obj = compute_csp_patterns_with_class_check(X_broadband, y)
    
    # 选择最能代表各类的pattern
    csp_class0 = patterns_csp[:, c0_idx[0]]
    csp_class1 = patterns_csp[:, c1_idx[0]]
    
    print(f"   特征值范围: {np.real(eig_csp).min():.3f} ~ {np.real(eig_csp).max():.3f}")
    print(f"   Class 0-dominant component: #{c0_idx[0]}")
    print(f"   Class 1-dominant component: #{c1_idx[0]}")
    
    # ==================== 2. FBCSP（展示等权重情况下选择的频段） ====================
    print("\n🔄 计算FBCSP（多频段等权重）...")
    
    # 对每个频段计算CSP
    fbcsp_band_patterns = []
    fbcsp_band_eig = []
    fbcsp_band_class_info = []
    
    for band in freq_bands:
        X_band = filter_data(X_raw, band, sampling_rate)
        patterns, eig_vals, c0_idx_band, c1_idx_band, _ = compute_csp_patterns_with_class_check(X_band, y)
        fbcsp_band_patterns.append(patterns)
        fbcsp_band_eig.append(eig_vals)
        fbcsp_band_class_info.append((c0_idx_band, c1_idx_band))
        
        band_label = BAND_LABELS.get(band, '')
        print(f"   频段 {band[0]:2d}-{band[1]:2d}Hz ({band_label:>6}): "
              f"特征值 {np.real(eig_vals).min():.3f}~{np.real(eig_vals).max():.3f}")
    
    # 等权重：选择中间频段作为代表（展示FBCSP的"中庸"策略）
    equal_weights = np.ones(len(freq_bands)) / len(freq_bands)
    fbcsp_representative_band_idx = len(freq_bands) // 2  # 中间频段 (16-20Hz β)
    
    print(f"\n   FBCSP代表频段（等权重下选择中间频段）: {freq_bands[fbcsp_representative_band_idx]} "
          f"({BAND_LABELS[freq_bands[fbcsp_representative_band_idx]]})")
    
    fbcsp_c0_idx, fbcsp_c1_idx = fbcsp_band_class_info[fbcsp_representative_band_idx]
    fbcsp_class0 = fbcsp_band_patterns[fbcsp_representative_band_idx][:, fbcsp_c0_idx[0]]
    fbcsp_class1 = fbcsp_band_patterns[fbcsp_representative_band_idx][:, fbcsp_c1_idx[0]]
    
    # ==================== 3. AWFBCSP（展示自适应选择的最优频段） ====================
    print("\n🔄 计算AWFBCSP（自适应权重）...")
    
    awfbcsp = AdaptiveWeightedFBCSP(
        m_filters=3,
        sampling_rate=sampling_rate,
        use_adaptive_weights=True,
        use_temporal_windows=False,
        use_erd_features=False,
        use_multiscale=False
    )
    awfbcsp.freq_bands = freq_bands
    awfbcsp.n_fbanks = len(freq_bands)
    
    awfbcsp.fit(X_raw, y)
    adaptive_weights = awfbcsp.band_weights
    
    print(f"   自适应权重: {adaptive_weights}")
    
    # 选择权重最大的频段作为AWFBCSP的代表
    awfbcsp_best_band_idx = np.argmax(adaptive_weights)
    awfbcsp_best_band = freq_bands[awfbcsp_best_band_idx]
    
    print(f"\n   AWFBCSP最优频段: {awfbcsp_best_band} "
          f"({BAND_LABELS[awfbcsp_best_band]}, 权重={adaptive_weights[awfbcsp_best_band_idx]:.3f})")
    
    awfbcsp_c0_idx, awfbcsp_c1_idx = fbcsp_band_class_info[awfbcsp_best_band_idx]
    awfbcsp_class0 = fbcsp_band_patterns[awfbcsp_best_band_idx][:, awfbcsp_c0_idx[0]]
    awfbcsp_class1 = fbcsp_band_patterns[awfbcsp_best_band_idx][:, awfbcsp_c1_idx[0]]
    
    # ==================== 绘图 ====================
    print("\n🎨 绘制对比图...")
    
    # 统一色标
    all_vals = np.concatenate([
        csp_class0, csp_class1,
        fbcsp_class0, fbcsp_class1,
        awfbcsp_class0, awfbcsp_class1
    ])
    vlim = np.max(np.abs(all_vals))
    vmin, vmax = -vlim, vlim
    
    # 创建图形：3行3列（调整间距让图形更清爽）
    fig = plt.figure(figsize=(14, 13))
    gs = GridSpec(4, 4, height_ratios=[1, 1, 1, 0.4], width_ratios=[1, 1, 1, 0.05], 
                  hspace=0.4, wspace=0.25)
    
    # ========== 第一行：CSP（8-30Hz broadband） ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    plot_topomap(ax1, csp_class0, CHAN_POS_22, CHANNEL_NAMES_22,
                 'CSP: Class 0\n(8-30Hz broadband)', vmin, vmax)
    plot_topomap(ax2, csp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                 'CSP: Class 1\n(8-30Hz broadband)', vmin, vmax)
    plot_topomap(ax3, csp_class0 - csp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                 'CSP: Difference\n(Class 0 - 1)', vmin, vmax, cmap='seismic')
    
    # ========== 第二行：FBCSP（代表频段：μ） ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    fbcsp_band = freq_bands[fbcsp_representative_band_idx]
    fbcsp_label = BAND_LABELS[fbcsp_band]
    
    plot_topomap(ax4, fbcsp_class0, CHAN_POS_22, CHANNEL_NAMES_22,
                 f'FBCSP: Class 0\n({fbcsp_band[0]}-{fbcsp_band[1]}Hz {fbcsp_label})',
                 vmin, vmax)
    plot_topomap(ax5, fbcsp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                 f'FBCSP: Class 1\n({fbcsp_band[0]}-{fbcsp_band[1]}Hz {fbcsp_label})',
                 vmin, vmax)
    plot_topomap(ax6, fbcsp_class0 - fbcsp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                 'FBCSP: Difference\n(Class 0 - 1)', vmin, vmax, cmap='seismic')
    
    # ========== 第三行：AWFBCSP（自适应选择的最优频段） ==========
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    awfbcsp_label = BAND_LABELS[awfbcsp_best_band]
    
    plot_topomap(ax7, awfbcsp_class0, CHAN_POS_22, CHANNEL_NAMES_22,
                 f'AWFBCSP: Class 0\n({awfbcsp_best_band[0]}-{awfbcsp_best_band[1]}Hz {awfbcsp_label}, w={adaptive_weights[awfbcsp_best_band_idx]:.2f})',
                 vmin, vmax)
    cf = plot_topomap(ax8, awfbcsp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                      f'AWFBCSP: Class 1\n({awfbcsp_best_band[0]}-{awfbcsp_best_band[1]}Hz {awfbcsp_label}, w={adaptive_weights[awfbcsp_best_band_idx]:.2f})',
                      vmin, vmax)
    plot_topomap(ax9, awfbcsp_class0 - awfbcsp_class1, CHAN_POS_22, CHANNEL_NAMES_22,
                 'AWFBCSP: Difference\n(Class 0 - 1)', vmin, vmax, cmap='seismic')
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[:3, 3])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cbar_ax, label='Pattern Amplitude (a.u.)')
    
    # ========== 第四行：频段权重对比（核心：展示AWFBCSP的特征选择策略） ==========
    ax_weights = fig.add_subplot(gs[3, :3])
    
    x_pos = np.arange(len(freq_bands))
    width = 0.35
    
    # 添加生理标签
    band_labels_full = [f'{b[0]}-{b[1]}Hz\n({BAND_LABELS[b]})' for b in freq_bands]
    
    bars1 = ax_weights.bar(x_pos - width/2, equal_weights, width, 
                           label='FBCSP (Equal)', color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax_weights.bar(x_pos + width/2, adaptive_weights, width, 
                           label='AWFBCSP (Adaptive)', color='coral', alpha=0.7, edgecolor='black')
    
    # 高亮AWFBCSP选择的最优频段
    bars2[awfbcsp_best_band_idx].set_facecolor('red')
    bars2[awfbcsp_best_band_idx].set_edgecolor('darkred')
    bars2[awfbcsp_best_band_idx].set_linewidth(2.5)
    
    ax_weights.set_xlabel('Frequency Band', fontsize=10, fontweight='bold')
    ax_weights.set_ylabel('Weight', fontsize=10, fontweight='bold')
    ax_weights.set_title('AWFBCSP Feature Re-weighting Strategy', fontsize=11, fontweight='bold')
    ax_weights.set_xticks(x_pos)
    ax_weights.set_xticklabels(band_labels_full, fontsize=8)
    ax_weights.legend(loc='upper right')
    ax_weights.grid(axis='y', alpha=0.3)
    
    # ========== 总标题 + 关键说明 ==========
    fig.suptitle(f'Spatial Pattern Analysis: CSP vs FBCSP vs AWFBCSP\n'
                 f'(Showing CSP patterns from individual frequency bands)\n'
                 f'BCI Competition IV 2a, Subject A{subject_id:02d}',
                 fontsize=13, fontweight='bold', y=0.995)
    
    # 添加关键说明文字（方案B1的核心caption）
    note_text = ("Note: AWFBCSP does not learn new spatial patterns. "
                 "It adaptively selects and re-weights\n"
                 "CSP features from multiple frequency bands. "
                 f"Red bars show the selected band ({awfbcsp_best_band[0]}-{awfbcsp_best_band[1]}Hz).")
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存（调整布局避免文字被裁剪）
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = 'results/csp_fbcsp_awfbcsp_comparison_fixed.png'
    os.makedirs('results', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.2)
    
    print(f"\n✅ 图片已保存: {output_path}")
    
    # ==================== 额外：各频段patterns详细对比（带权重可视化） ====================
    print("\n🎨 绘制各频段patterns详细图...")
    
    fig2, axes = plt.subplots(2, len(freq_bands), figsize=(2.8*len(freq_bands), 6))
    
    for i, (band, patterns, (c0_idx_band, c1_idx_band)) in enumerate(zip(
            freq_bands, fbcsp_band_patterns, fbcsp_band_class_info)):
        
        band_label = BAND_LABELS[band]
        is_best = (i == awfbcsp_best_band_idx)
        
        ax_c0 = axes[0, i]
        plot_topomap(ax_c0, patterns[:, c0_idx_band[0]], CHAN_POS_22, CHANNEL_NAMES_22,
                    f'{band[0]}-{band[1]}Hz ({band_label})\nClass 0',
                    vmin, vmax)
        
        # Class 1 pattern
        ax_c1 = axes[1, i]
        plot_topomap(ax_c1, patterns[:, c1_idx_band[0]], CHAN_POS_22, CHANNEL_NAMES_22,
                    f'Class 1',
                    vmin, vmax)
        
        # 用边框粗细表示权重（最优频段用红色粗框）
        border_color = 'red' if is_best else 'gray'
        border_width = 5 if is_best else 1
        
        for spine in ax_c0.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        
        for spine in ax_c1.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        
        # 显示权重（更清晰的位置和样式）
        weight_val = adaptive_weights[i]
        
        if is_best:
            # 最优频段：使用更醒目的样式
            ax_c0.text(0, -0.12, f'w = {weight_val:.3f}',
                      transform=ax_c0.transAxes, ha='center', va='top',
                      fontsize=11, fontweight='bold',
                      color='white',
                      bbox=dict(boxstyle='round,pad=0.6', 
                               facecolor='red', 
                               edgecolor='darkred',
                               linewidth=3, alpha=0.95))
        else:
            # 普通频段：灰色简洁样式
            weight_str = f'w={weight_val:.3f}'
            ax_c0.text(0, -0.15, weight_str,
                      transform=ax_c0.transAxes, ha='center', va='top',
                      fontsize=10, fontweight='normal',
                      color='black',
                      bbox=dict(boxstyle='round,pad=0.4', 
                               facecolor='lightgray', 
                               edgecolor='gray',
                               linewidth=1, alpha=0.8))
    
    fig2.suptitle(f'CSP Spatial Patterns for Individual Frequency Bands\n'
                  f'(Red border indicates band selected by AWFBCSP adaptive weighting)',
                  fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    fig2.colorbar(sm, cax=cbar_ax2, label='Pattern Amplitude (a.u.)')
    
    # 调整布局，给权重文字留出更多空间
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.96])
    
    output_path2 = 'results/csp_patterns_per_band_fixed.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    fig2.savefig(output_path2.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.3)
    
    print(f"✅ 各频段详细图已保存: {output_path2}")
    
    plt.show()
    
    # ==================== 打印总结 ====================
    print("\n" + "=" * 90)
    print("📊 可视化逻辑总结（方案B1：理论严谨版）:")
    print("=" * 90)
    print(f"\n{'方法':<15} {'可视化内容':<50} {'理论依据':<25}")
    print("-" * 90)
    print(f"{'CSP':<15} {'展示8-30Hz单频段的CSP patterns':<50} {'Baseline方法':<25}")
    print(f"{'FBCSP':<15} {f'展示示例频段({fbcsp_band[0]}-{fbcsp_band[1]}Hz {fbcsp_label})的CSP patterns':<50} {'等权重策略':<25}")
    print(f"{'AWFBCSP':<15} {f'展示选中频段({awfbcsp_best_band[0]}-{awfbcsp_best_band[1]}Hz {awfbcsp_label})的CSP patterns':<50} {'自适应权重选择':<25}")
    
    print(f"\n{'='*90}")
    print("🔬 核心理论说明（审稿关键）:")
    print("=" * 90)
    print("✅ AWFBCSP不学习新的spatial patterns")
    print("✅ 它通过feature-level的adaptive weighting来优化FBCSP")
    print("✅ 可视化展示的是各频段的CSP patterns + AWFBCSP的权重选择策略")
    print("✅ 这避免了跨频段直接加权平均patterns的理论问题")
    
    print(f"\n📌 AWFBCSP学习到的频段权重:")
    for band, weight in zip(freq_bands, adaptive_weights):
        band_label = BAND_LABELS[band]
        is_max = (weight == adaptive_weights.max())
        importance = "⭐⭐⭐ [SELECTED]" if is_max else ("⭐⭐" if weight > 0.15 else "⭐")
        print(f"   {band[0]:2d}-{band[1]:2d}Hz ({band_label:>6}): {weight:.4f} {importance}")
    
    print("\n" + "=" * 90)
    print("✅ 方案B1关键要点（审稿安全）:")
    print("=" * 90)
    print("   1. ✅ 不跨频段加权平均patterns（理论严谨）")
    print("   2. ✅ 展示各频段独立的CSP patterns")
    print("   3. ✅ 用权重图展示AWFBCSP的特征选择策略")
    print("   4. ✅ 明确说明：AWFBCSP优化的是feature-level而非pattern-level")
    print("   5. ✅ 使用方差比确定class-dominant patterns（避免类别假设）")
    print("=" * 90)
    
    print("\n📝 论文Caption建议（可直接使用）:")
    print("-" * 90)
    print("\"AWFBCSP does not learn new spatial patterns.")
    print(" Instead, it adaptively selects and re-weights CSP features")
    print(" from multiple frequency bands. The spatial maps shown correspond")
    print(" to CSP patterns of individual bands.\"")
    print("=" * 90)


if __name__ == "__main__":
    main()

