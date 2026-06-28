"""
CSP vs FBCSP vs AWFBCSP 空间模式对比 - 论文版本
适合学术论文插图，紧凑布局，专业风格
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
import matplotlib
import sys
import os

# 论文级别的字体设置
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 12

sys.path.insert(0, 'src')

from src.features.csp import CSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP

# ==================== 22通道位置 ====================
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
    """加载并预处理数据"""
    data_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_data.npy'
    label_path = f'dataset/bci_iv_2a/A{subject_id:02d}T_label.npy'
    
    X = np.load(data_path)
    y = np.load(label_path)
    
    binary_mask = (y == 1) | (y == 2)
    X = X[binary_mask]
    y = y[binary_mask] - 1
    
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
    return csp.patterns_, csp.eig_values


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
    
    return X, Y, Zf.reshape(X.shape)


def plot_topomap_publication(ax, values, chan_pos, labels, vmin=None, vmax=None):
    """绘制论文级别的地形图"""
    xs = np.array([chan_pos[ch][0] for ch in labels])
    ys = np.array([chan_pos[ch][1] for ch in labels])
    
    X, Y, Z = interpolate_topomap(values, chan_pos, labels, grid_res=150, radius=1.0, power=2.0)
    
    # 填色
    cf = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax, extend='both')
    ax.contour(X, Y, Z, levels=8, colors='k', linewidths=0.15, alpha=0.3)
    
    # 头皮轮廓
    head = plt.Circle((0, 0), 1.0, edgecolor='k', facecolor='none', linewidth=1.2)
    ax.add_patch(head)
    
    # 鼻子
    nose = plt.Polygon([[-0.05, 1.0], [0.05, 1.0], [0.00, 1.08]], 
                       closed=True, edgecolor='k', facecolor='k', linewidth=0.8)
    ax.add_patch(nose)
    
    # 耳朵
    ear_left = plt.Circle((-1.02, 0.0), 0.05, edgecolor='k', facecolor='none', linewidth=0.8)
    ear_right = plt.Circle((1.02, 0.0), 0.05, edgecolor='k', facecolor='none', linewidth=0.8)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # 电极位置（小点）
    ax.scatter(xs, ys, s=8, color='k', zorder=5)
    
    # 只标注C3和C4
    for ch, x, y in zip(labels, xs, ys):
        if ch in ['C3', 'C4']:
            ax.annotate(ch, (x, y), xytext=(0, -8), textcoords='offset points', 
                       fontsize=7, ha='center', fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.05, 1.15)
    
    return cf


def main():
    """主函数 - 生成论文图片"""
    print("=" * 60)
    print("生成论文级别的 CSP/FBCSP/AWFBCSP 对比图")
    print("=" * 60)
    
    # 配置
    subject_id = 1
    sampling_rate = 250
    # 与实验一致的6个频段
    freq_bands = [
        (8, 12),   # Alpha
        (12, 16),  # Low Beta
        (16, 20),  # Mid Beta
        (20, 24),  # High Beta
        (24, 28),  # Low Gamma
        (28, 32)   # Mid Gamma
    ]
    
    # 加载数据
    print("\n加载数据...")
    X_raw, y = load_and_preprocess(subject_id)
    
    # ==================== 计算三种方法的模式 ====================
    
    # 1. CSP (8-30Hz)
    print("计算 CSP...")
    X_broadband = filter_data(X_raw, (8, 30), sampling_rate)
    csp_patterns, _ = compute_csp_patterns(X_broadband, y)
    csp_left, csp_right = csp_patterns[:, 0], csp_patterns[:, -1]
    
    # 2. FBCSP (等权重)
    print("计算 FBCSP...")
    fbcsp_band_patterns = []
    for band in freq_bands:
        X_band = filter_data(X_raw, band, sampling_rate)
        patterns, _ = compute_csp_patterns(X_band, y)
        fbcsp_band_patterns.append(patterns)
    
    equal_weights = np.ones(len(freq_bands)) / len(freq_bands)
    fbcsp_left = sum(w * p[:, 0] for w, p in zip(equal_weights, fbcsp_band_patterns))
    fbcsp_right = sum(w * p[:, -1] for w, p in zip(equal_weights, fbcsp_band_patterns))
    
    # 3. AWFBCSP (自适应权重)
    print("计算 AWFBCSP...")
    awfbcsp = AdaptiveWeightedFBCSP(
        m_filters=3, sampling_rate=sampling_rate,
        use_adaptive_weights=True, use_temporal_windows=False,
        use_erd_features=False, use_multiscale=False
    )
    awfbcsp.freq_bands = freq_bands
    awfbcsp.n_fbanks = len(freq_bands)
    awfbcsp.fit(X_raw, y)
    adaptive_weights = awfbcsp.band_weights
    
    awfbcsp_left = sum(w * p[:, 0] for w, p in zip(adaptive_weights, fbcsp_band_patterns))
    awfbcsp_right = sum(w * p[:, -1] for w, p in zip(adaptive_weights, fbcsp_band_patterns))
    
    print(f"\nAWFBCSP 自适应权重: {adaptive_weights}")
    
    # ==================== 图1: 主对比图 (2行3列) ====================
    print("\n生成主对比图...")
    
    all_vals = np.concatenate([csp_left, csp_right, fbcsp_left, fbcsp_right, 
                               awfbcsp_left, awfbcsp_right])
    vlim = np.max(np.abs(all_vals))
    vmin, vmax = -vlim, vlim
    
    # 创建图形 - 适合论文双栏宽度 (约17cm = 6.7inch)
    fig = plt.figure(figsize=(7, 5))
    
    # 使用GridSpec精确控制布局
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.08], 
                  height_ratios=[1, 1],
                  wspace=0.1, hspace=0.25)
    
    # 方法名称和数据
    methods = [
        ('CSP\n(8-30 Hz)', csp_left, csp_right),
        ('FBCSP\n(Equal weights)', fbcsp_left, fbcsp_right),
        ('AWFBCSP\n(Adaptive weights)', awfbcsp_left, awfbcsp_right),
    ]
    
    # 子图标签
    labels_fig = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    label_idx = 0
    
    for col, (method_name, left_pattern, right_pattern) in enumerate(methods):
        # 上行：左手
        ax_top = fig.add_subplot(gs[0, col])
        plot_topomap_publication(ax_top, left_pattern, CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        
        # 添加子图标签
        ax_top.text(-1.1, 1.1, labels_fig[label_idx], fontsize=10, fontweight='bold',
                   transform=ax_top.transData, va='top')
        label_idx += 1
        
        # 方法名称（只在上行显示）
        ax_top.set_title(method_name, fontsize=9, fontweight='bold', pad=5)
        
        # 下行：右手
        ax_bottom = fig.add_subplot(gs[1, col])
        cf = plot_topomap_publication(ax_bottom, right_pattern, CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        
        ax_bottom.text(-1.1, 1.1, labels_fig[label_idx], fontsize=10, fontweight='bold',
                      transform=ax_bottom.transData, va='top')
        label_idx += 1
    
    # 行标签（左侧）
    fig.text(0.02, 0.72, 'Left hand', fontsize=10, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.28, 'Right hand', fontsize=10, fontweight='bold', 
             rotation=90, va='center', ha='center')
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Pattern amplitude', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.98])
    
    # 保存
    fig.savefig('results/Fig_CSP_comparison_main.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('results/Fig_CSP_comparison_main.pdf', bbox_inches='tight')
    fig.savefig('results/Fig_CSP_comparison_main.eps', bbox_inches='tight', format='eps')
    print("保存: results/Fig_CSP_comparison_main.png/pdf/eps")
    
    # ==================== 图2: 频段权重对比 ====================
    print("\n生成权重对比图...")
    
    fig2, ax = plt.subplots(figsize=(4, 2.5))
    
    x_pos = np.arange(len(freq_bands))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, equal_weights, width, label='FBCSP', 
                   color='#4ECDC4', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, adaptive_weights, width, label='AWFBCSP', 
                   color='#FF6B6B', edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Frequency band (Hz)', fontsize=10)
    ax.set_ylabel('Weight', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{b[0]}-{b[1]}' for b in freq_bands], fontsize=9)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.set_ylim(0, max(adaptive_weights) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注最高权重
    max_idx = np.argmax(adaptive_weights)
    ax.annotate(f'{adaptive_weights[max_idx]:.2f}', 
               xy=(max_idx + width/2, adaptive_weights[max_idx]),
               xytext=(0, 5), textcoords='offset points',
               fontsize=8, ha='center', fontweight='bold', color='#FF6B6B')
    
    plt.tight_layout()
    fig2.savefig('results/Fig_band_weights.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig('results/Fig_band_weights.pdf', bbox_inches='tight')
    print("保存: results/Fig_band_weights.png/pdf")
    
    # ==================== 图3: 各频段详细模式 ====================
    print("\n生成各频段详细图...")
    
    fig3 = plt.figure(figsize=(12, 4))
    gs3 = GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.08], 
                   wspace=0.08, hspace=0.15)
    
    for i, (band, patterns) in enumerate(zip(freq_bands, fbcsp_band_patterns)):
        # 上行：左手
        ax_top = fig3.add_subplot(gs3[0, i])
        plot_topomap_publication(ax_top, patterns[:, 0], CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        ax_top.set_title(f'{band[0]}-{band[1]} Hz', fontsize=9, fontweight='bold', pad=3)
        
        # 下行：右手
        ax_bottom = fig3.add_subplot(gs3[1, i])
        cf = plot_topomap_publication(ax_bottom, patterns[:, -1], CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        
        # 显示权重
        weight_color = '#FF6B6B' if adaptive_weights[i] == max(adaptive_weights) else 'black'
        ax_bottom.text(0, -1.0, f'w = {adaptive_weights[i]:.2f}', fontsize=8, 
                      ha='center', fontweight='bold', color=weight_color)
    
    # 行标签
    fig3.text(0.02, 0.72, 'Left hand', fontsize=10, fontweight='bold', 
              rotation=90, va='center')
    fig3.text(0.02, 0.28, 'Right hand', fontsize=10, fontweight='bold', 
              rotation=90, va='center')
    
    # Colorbar
    cbar_ax3 = fig3.add_subplot(gs3[:, 6])
    cbar3 = fig3.colorbar(sm, cax=cbar_ax3)
    cbar3.set_label('Pattern amplitude', fontsize=9)
    cbar3.ax.tick_params(labelsize=8)
    
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    fig3.savefig('results/Fig_CSP_per_band.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig('results/Fig_CSP_per_band.pdf', bbox_inches='tight')
    print("保存: results/Fig_CSP_per_band.png/pdf")
    
    # ==================== 图4: 综合单图版本（适合单栏） ====================
    print("\n生成综合单图版本...")
    
    fig4 = plt.figure(figsize=(5.5, 7))
    gs4 = GridSpec(4, 3, height_ratios=[1, 1, 0.4, 0.05],
                   width_ratios=[1, 1, 0.08],
                   wspace=0.1, hspace=0.3)
    
    # 上面2x2: CSP和AWFBCSP对比（省略FBCSP）
    compare_methods = [
        ('CSP (8-30 Hz)', csp_left, csp_right),
        ('AWFBCSP', awfbcsp_left, awfbcsp_right),
    ]
    
    for col, (method_name, left_p, right_p) in enumerate(compare_methods):
        ax_top = fig4.add_subplot(gs4[0, col])
        plot_topomap_publication(ax_top, left_p, CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        ax_top.set_title(f'{method_name}\nLeft hand', fontsize=9, fontweight='bold', pad=3)
        ax_top.text(-1.05, 1.05, f'({chr(97+col*2)})', fontsize=10, fontweight='bold',
                   transform=ax_top.transData)
        
        ax_bottom = fig4.add_subplot(gs4[1, col])
        cf = plot_topomap_publication(ax_bottom, right_p, CHAN_POS_22, CHANNEL_NAMES_22, vmin, vmax)
        ax_bottom.set_title('Right hand', fontsize=9, pad=3)
        ax_bottom.text(-1.05, 1.05, f'({chr(98+col*2)})', fontsize=10, fontweight='bold',
                      transform=ax_bottom.transData)
    
    # Colorbar（右侧）
    cbar_ax4 = fig4.add_subplot(gs4[:2, 2])
    cbar4 = fig4.colorbar(sm, cax=cbar_ax4)
    cbar4.set_label('Amplitude', fontsize=9)
    cbar4.ax.tick_params(labelsize=8)
    
    # 底部：权重对比
    ax_weights = fig4.add_subplot(gs4[2, :2])
    
    x_pos = np.arange(len(freq_bands))
    width = 0.35
    
    bars1 = ax_weights.bar(x_pos - width/2, equal_weights, width, label='FBCSP', 
                           color='#4ECDC4', edgecolor='black', linewidth=0.6)
    bars2 = ax_weights.bar(x_pos + width/2, adaptive_weights, width, label='AWFBCSP', 
                           color='#FF6B6B', edgecolor='black', linewidth=0.6)
    
    ax_weights.set_xlabel('Frequency band (Hz)', fontsize=9)
    ax_weights.set_ylabel('Weight', fontsize=9)
    ax_weights.set_xticks(x_pos)
    ax_weights.set_xticklabels([f'{b[0]}-{b[1]}' for b in freq_bands], fontsize=8)
    ax_weights.legend(loc='upper right', fontsize=8, frameon=True)
    ax_weights.set_ylim(0, max(adaptive_weights) * 1.25)
    ax_weights.spines['top'].set_visible(False)
    ax_weights.spines['right'].set_visible(False)
    ax_weights.grid(axis='y', alpha=0.3, linestyle='--')
    ax_weights.text(-0.5, max(adaptive_weights) * 1.15, '(e)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig4.savefig('results/Fig_CSP_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig4.savefig('results/Fig_CSP_combined.pdf', bbox_inches='tight')
    print("保存: results/Fig_CSP_combined.png/pdf")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("所有论文图片生成完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. Fig_CSP_comparison_main.pdf  - 主对比图 (2×3布局)")
    print("  2. Fig_band_weights.pdf         - 频段权重对比")
    print("  3. Fig_CSP_per_band.pdf         - 各频段详细模式")
    print("  4. Fig_CSP_combined.pdf         - 综合单图版本")


if __name__ == "__main__":
    main()

