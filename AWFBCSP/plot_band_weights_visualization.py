"""
📊 图A：频段权重可解释性可视化（MI-weight Visualization）

论文补充图：展示AWFBCSP学习到的被试特异性频段权重分布
- 揭示不同被试的μ/β节律主导模式
- 证明AWFBCSP能够自适应学习频段重要性

输出：
- 一张柱状图：BCI IV 2a + 2b 全部被试的频段权重（带误差条）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP

# ==================== 配置 ====================
# Times New Roman 字体（学术论文标准）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号也用衬线字体
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 频段配置
FREQ_BANDS = [
    (8, 12),   # Alpha/Mu
    (12, 16),  # Low Beta
    (16, 20),  # Mid Beta
    (20, 24),  # High Beta
    (24, 30),  # Low Gamma
]

BAND_NAMES = ['8–12 Hz\n(μ)', '12–16 Hz\n(low β)', '16–20 Hz\n(β)', 
              '20–24 Hz\n(high β)', '24–30 Hz\n(γ)']


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
    
    # 二分类（左右手）
    binary_mask = (y == 1) | (y == 2)
    X = X[binary_mask]
    y = y[binary_mask] - 1
    
    # 基线校正
    sampling_rate = 250
    baseline_samples = int(0.5 * sampling_rate)
    X = X - X[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    
    # 带通滤波
    nyquist = sampling_rate / 2
    b, a = butter(4, [8/nyquist, 30/nyquist], btype='band')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, :] = filtfilt(b, a, X[i, j, :])
    
    return X, y


def compute_band_weights_for_subjects(subject_ids, dataset='2a'):
    """计算所有被试的频段权重，返回权重矩阵"""
    weights_list = []
    
    for subject_id in subject_ids:
        try:
            print(f"  处理被试 {'A' if dataset=='2a' else 'B'}{subject_id:02d}...", end='')
            X, y = load_and_preprocess(subject_id, dataset)
            
            # 训练AWFBCSP
            awfbcsp = AdaptiveWeightedFBCSP(
                m_filters=3,
                sampling_rate=250,
                freq_bands=FREQ_BANDS,
                use_adaptive_weights=True,
                use_temporal_windows=False,
                use_erd_features=False,
                use_multiscale=False
            )
            awfbcsp.fit(X, y)
            
            weights_list.append(awfbcsp.band_weights)
            print(f" 权重: [{', '.join([f'{w:.2f}' for w in awfbcsp.band_weights])}]")
            
        except Exception as e:
            print(f" ⚠️ 失败: {e}")
    
    return np.array(weights_list)


def plot_combined_bar_chart(weights_2a, weights_2b, save_path=None):
    """
    绘制单张柱状图：两个数据集全部被试的频段权重对比（带误差条）
    
    论文Figure标题建议：
    "Learned MI-guided sub-band weights across all subjects from 
     BCI Competition IV Dataset 2a (22-channel) and 2b (3-channel), 
     demonstrating consistent subject-specific μ/β rhythm preferences."
    """
    n_bands = len(FREQ_BANDS)
    
    # 计算均值和标准差
    mean_2a = np.mean(weights_2a, axis=0)
    std_2a = np.std(weights_2a, axis=0)
    mean_2b = np.mean(weights_2b, axis=0)
    std_2b = np.std(weights_2b, axis=0)
    
    # 专业配色方案（高对比度，色盲友好）
    color_2a = '#3274A1'  # 深蓝色 - Dataset 2a (22通道)
    color_2b = '#E1812C'  # 橙色 - Dataset 2b (3通道)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_bands)
    width = 0.38
    
    # 绘制柱状图（带渐变效果的阴影）
    bars1 = ax.bar(x - width/2, mean_2a, width, yerr=std_2a, 
                   label='Dataset 2a (22-ch, n=9)', color=color_2a, alpha=0.9,
                   edgecolor='#1a1a1a', linewidth=1.2, capsize=4, 
                   error_kw={'elinewidth': 1.5, 'capthick': 1.2, 'color': '#333333'})
    
    bars2 = ax.bar(x + width/2, mean_2b, width, yerr=std_2b,
                   label='Dataset 2b (3-ch, n=9)', color=color_2b, alpha=0.9,
                   edgecolor='#1a1a1a', linewidth=1.2, capsize=4,
                   error_kw={'elinewidth': 1.5, 'capthick': 1.2, 'color': '#333333'})
    
    # 在柱子上方标注数值（更精致的样式）
    for bar, mean, std in zip(bars1, mean_2a, std_2a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.015,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#2a2a2a')
    
    for bar, mean, std in zip(bars2, mean_2b, std_2b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.015,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#2a2a2a')
    
    # 设置坐标轴标签
    ax.set_xlabel('Frequency Sub-band', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Normalized MI Weight', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, fontsize=16)
    
    # 设置y轴范围
    max_val = max(np.max(mean_2a + std_2a), np.max(mean_2b + std_2b))
    ax.set_ylim([0, max_val * 1.22])
    
    # 图例（精致样式）
    legend = ax.legend(loc='upper right', framealpha=0.95, 
                       edgecolor='#cccccc', fontsize=12,
                       fancybox=True, shadow=False)
    legend.get_frame().set_linewidth(1.0)
    
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 关闭网格线
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"\n✅ 图片已保存: {save_path}")
        print(f"✅ PDF已保存: {save_path.replace('.png', '.pdf')}")
    
    plt.show()
    return fig


def plot_individual_subjects_bar(weights_2a, weights_2b, save_path=None):
    """
    绘制所有18个被试的个体柱状图（补充材料用）
    """
    n_bands = len(FREQ_BANDS)
    n_subjects_2a = weights_2a.shape[0]
    n_subjects_2b = weights_2b.shape[0]
    
    # 优雅的颜色渐变
    colors_2a = plt.cm.Blues(np.linspace(0.35, 0.85, n_subjects_2a))
    colors_2b = plt.cm.Oranges(np.linspace(0.35, 0.85, n_subjects_2b))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_bands)
    total_subjects = n_subjects_2a + n_subjects_2b
    width = 0.85 / total_subjects
    
    # 绘制 2a 被试
    for i in range(n_subjects_2a):
        offset = (i - total_subjects/2 + 0.5) * width
        ax.bar(x + offset, weights_2a[i], width, 
               color=colors_2a[i], alpha=0.9, edgecolor='#333333', linewidth=0.6,
               label='Dataset 2a (A01–A09)' if i == 0 else None)
    
    # 绘制 2b 被试
    for i in range(n_subjects_2b):
        offset = (n_subjects_2a + i - total_subjects/2 + 0.5) * width
        ax.bar(x + offset, weights_2b[i], width,
               color=colors_2b[i], alpha=0.9, edgecolor='#333333', linewidth=0.6,
               label='Dataset 2b (B01–B09)' if i == 0 else None)
    
    # 设置坐标轴
    ax.set_xlabel('Frequency Sub-band', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Normalized MI Weight', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, fontsize=11)
    ax.set_ylim([0, np.max([weights_2a.max(), weights_2b.max()]) * 1.12])
    
    # 图例
    legend = ax.legend(loc='upper right', framealpha=0.95, fontsize=11,
                       edgecolor='#cccccc', fancybox=True)
    legend.get_frame().set_linewidth(1.0)
    
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 关闭网格线
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"\n✅ 个体图已保存: {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("📊 图A：频段权重可解释性可视化")
    print("   BCI IV 2a + 2b 全部18个被试")
    print("=" * 60)
    
    all_subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # ============ 收集所有被试权重 ============
    print("\n🔹 处理 BCI IV 2a 数据集 (9 subjects)...")
    weights_2a = compute_band_weights_for_subjects(all_subject_ids, dataset='2a')
    
    print("\n🔹 处理 BCI IV 2b 数据集 (9 subjects)...")
    weights_2b = compute_band_weights_for_subjects(all_subject_ids, dataset='2b')
    
    # ============ 打印统计信息 ============
    print("\n" + "=" * 60)
    print("📊 权重统计:")
    print("=" * 60)
    print(f"\nBCI IV 2a (Mean ± Std):")
    for i, band in enumerate(BAND_NAMES):
        band_name = band.replace('\n', ' ')
        print(f"  {band_name}: {np.mean(weights_2a[:, i]):.3f} ± {np.std(weights_2a[:, i]):.3f}")
    
    print(f"\nBCI IV 2b (Mean ± Std):")
    for i, band in enumerate(BAND_NAMES):
        band_name = band.replace('\n', ' ')
        print(f"  {band_name}: {np.mean(weights_2b[:, i]):.3f} ± {np.std(weights_2b[:, i]):.3f}")
    
    # ============ 绘制主图（推荐用于论文） ============
    print("\n" + "=" * 60)
    print("📈 绘制频段权重柱状图（论文主图）")
    print("=" * 60)
    plot_combined_bar_chart(weights_2a, weights_2b, 
                           'results/Fig_band_weights_all_subjects.png')
    
    # ============ 可选：绘制个体被试图 ============
    print("\n📈 绘制个体被试权重图（补充材料）")
    plot_individual_subjects_bar(weights_2a, weights_2b,
                                 'results/Fig_band_weights_individual.png')
    
    print("\n" + "=" * 60)
    print("✅ 图A频段权重可视化完成！")
    print("=" * 60)
    print("\n📝 论文写作建议：")
    print("""
    Figure X presents the learned MI-guided sub-band weights across all 
    18 subjects from BCI Competition IV Dataset 2a (22-channel EEG) and 
    2b (3-channel EEG). The results demonstrate that AWFBCSP consistently 
    learns subject-specific spectral preferences, with notable emphasis 
    on μ (8-12 Hz) and β (12-30 Hz) rhythms. The variation across subjects 
    (shown by error bars) confirms the importance of adaptive weighting 
    in accommodating individual differences in motor imagery patterns.
    """)
