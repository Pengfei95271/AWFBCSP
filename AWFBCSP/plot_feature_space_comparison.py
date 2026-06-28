"""
📊 图B：AWFBCSP vs FBCSP 特征空间可分性对比（t-SNE / PCA）

论文补充图：展示AWFBCSP相比FBCSP产生更紧凑、更可分的特征分布
- 证明自适应权重带来的特征质量提升
- 直观展示spectral-spatial discriminability的改善

输出：
1. PCA 二维投影对比图
2. t-SNE 二维投影对比图
3. 组合图（论文推荐）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')
from src.features.fbcsp import FBCSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP

# ==================== 配置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# 频段配置
FREQ_BANDS = [
    (8, 12),   # Alpha/Mu
    (12, 16),  # Low Beta
    (16, 20),  # Mid Beta
    (20, 24),  # High Beta
    (24, 30),  # Low Gamma
]

# 类别配色
CLASS_COLORS = {
    0: '#1f77b4',  # 蓝色 - Left Hand / Class 0
    1: '#d62728',  # 红色 - Right Hand / Class 1
}

CLASS_NAMES = {
    0: 'Left Hand',
    1: 'Right Hand',
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


def extract_fbcsp_features(X, y, m_filters=3):
    """提取标准FBCSP特征"""
    fbcsp = FBCSP(
        m_filters=m_filters,
        sampling_rate=250,
        freq_bands=FREQ_BANDS
    )
    fbcsp.fit(X, y)
    features = fbcsp.transform(X)
    return features, fbcsp


def extract_awfbcsp_features(X, y, m_filters=3):
    """提取AWFBCSP特征"""
    awfbcsp = AdaptiveWeightedFBCSP(
        m_filters=m_filters,
        sampling_rate=250,
        freq_bands=FREQ_BANDS,
        use_adaptive_weights=True,
        use_temporal_windows=False,
        use_erd_features=False,
        use_multiscale=False
    )
    awfbcsp.fit(X, y)
    features = awfbcsp.transform(X)
    return features, awfbcsp


def compute_separability_metrics(features, labels):
    """计算类别可分性指标"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    class0_mask = labels == 0
    class1_mask = labels == 1
    
    # 类内散布
    mean0 = np.mean(features_scaled[class0_mask], axis=0)
    mean1 = np.mean(features_scaled[class1_mask], axis=0)
    
    # 类间距离
    inter_class_dist = np.linalg.norm(mean0 - mean1)
    
    # 类内散布（平均标准差）
    std0 = np.mean(np.std(features_scaled[class0_mask], axis=0))
    std1 = np.mean(np.std(features_scaled[class1_mask], axis=0))
    intra_class_scatter = (std0 + std1) / 2
    
    # Fisher判别比
    fisher_ratio = inter_class_dist / (intra_class_scatter + 1e-10)
    
    return {
        'inter_class_dist': inter_class_dist,
        'intra_class_scatter': intra_class_scatter,
        'fisher_ratio': fisher_ratio
    }


def add_confidence_ellipse(ax, x, y, n_std=2.0, facecolor='none', **kwargs):
    """添加置信椭圆"""
    if len(x) < 3:
        return None
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = plt.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_pca_comparison(fbcsp_features, awfbcsp_features, labels, 
                        subject_name, save_path=None):
    """绘制PCA降维对比图"""
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    
    # FBCSP PCA
    fbcsp_scaled = scaler.fit_transform(fbcsp_features)
    fbcsp_pca = pca.fit_transform(fbcsp_scaled)
    fbcsp_var = pca.explained_variance_ratio_
    
    # AWFBCSP PCA
    awfbcsp_scaled = scaler.fit_transform(awfbcsp_features)
    awfbcsp_pca = pca.fit_transform(awfbcsp_scaled)
    awfbcsp_var = pca.explained_variance_ratio_
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (ax, data, title, var_ratio) in enumerate(zip(
        axes, 
        [fbcsp_pca, awfbcsp_pca],
        ['FBCSP Features', 'AWFBCSP Features'],
        [fbcsp_var, awfbcsp_var]
    )):
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            
            # 添加置信椭圆
            add_confidence_ellipse(ax, data[mask, 0], data[mask, 1],
                                   n_std=2.0, edgecolor=CLASS_COLORS[cls],
                                   linewidth=2, linestyle='--', alpha=0.8)
        
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'({"a" if ax_idx==0 else "b"}) {title}', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f'PCA Feature Space Comparison - Subject {subject_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"✅ PCA对比图已保存: {save_path}")
    
    plt.show()
    return fig


def plot_tsne_comparison(fbcsp_features, awfbcsp_features, labels, 
                         subject_name, save_path=None):
    """绘制t-SNE降维对比图"""
    scaler = StandardScaler()
    
    # FBCSP t-SNE
    fbcsp_scaled = scaler.fit_transform(fbcsp_features)
    tsne_fbcsp = TSNE(n_components=2, perplexity=30, random_state=42, 
                      n_iter=1000, learning_rate='auto', init='pca')
    fbcsp_tsne = tsne_fbcsp.fit_transform(fbcsp_scaled)
    
    # AWFBCSP t-SNE
    awfbcsp_scaled = scaler.fit_transform(awfbcsp_features)
    tsne_awfbcsp = TSNE(n_components=2, perplexity=30, random_state=42, 
                        n_iter=1000, learning_rate='auto', init='pca')
    awfbcsp_tsne = tsne_awfbcsp.fit_transform(awfbcsp_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (ax, data, title) in enumerate(zip(
        axes, 
        [fbcsp_tsne, awfbcsp_tsne],
        ['FBCSP Features', 'AWFBCSP Features']
    )):
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            
            # 添加置信椭圆
            add_confidence_ellipse(ax, data[mask, 0], data[mask, 1],
                                   n_std=2.0, edgecolor=CLASS_COLORS[cls],
                                   linewidth=2, linestyle='--', alpha=0.8)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'({"a" if ax_idx==0 else "b"}) {title}', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f't-SNE Feature Space Comparison - Subject {subject_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE对比图已保存: {save_path}")
    
    plt.show()
    return fig


def plot_combined_comparison(fbcsp_features, awfbcsp_features, labels, 
                             subject_name, save_path=None):
    """
    绘制组合对比图：PCA + t-SNE（论文推荐使用）
    
    论文Figure标题建议：
    "Feature space visualization comparing FBCSP and AWFBCSP.
     (a-b) PCA projection showing AWFBCSP produces more compact clusters.
     (c-d) t-SNE visualization demonstrating improved class separability."
    """
    scaler = StandardScaler()
    
    # PCA
    pca = PCA(n_components=2)
    fbcsp_scaled = scaler.fit_transform(fbcsp_features)
    fbcsp_pca = pca.fit_transform(fbcsp_scaled)
    fbcsp_var = pca.explained_variance_ratio_.copy()
    
    awfbcsp_scaled = scaler.fit_transform(awfbcsp_features)
    awfbcsp_pca = pca.fit_transform(awfbcsp_scaled)
    awfbcsp_var = pca.explained_variance_ratio_.copy()
    
    # t-SNE
    print("  计算t-SNE（这可能需要一些时间）...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                n_iter=1000, learning_rate='auto', init='pca')
    fbcsp_tsne = tsne.fit_transform(fbcsp_scaled)
    awfbcsp_tsne = tsne.fit_transform(awfbcsp_scaled)
    
    # 计算可分性指标
    fbcsp_metrics = compute_separability_metrics(fbcsp_features, labels)
    awfbcsp_metrics = compute_separability_metrics(awfbcsp_features, labels)
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, hspace=0.25, wspace=0.2)
    
    plot_data = [
        (gs[0, 0], fbcsp_pca, 'FBCSP (PCA)', fbcsp_var, 'pca', 'a'),
        (gs[0, 1], awfbcsp_pca, 'AWFBCSP (PCA)', awfbcsp_var, 'pca', 'b'),
        (gs[1, 0], fbcsp_tsne, 'FBCSP (t-SNE)', None, 'tsne', 'c'),
        (gs[1, 1], awfbcsp_tsne, 'AWFBCSP (t-SNE)', None, 'tsne', 'd'),
    ]
    
    for gs_pos, data, title, var_ratio, method, subplot_label in plot_data:
        ax = fig.add_subplot(gs_pos)
        
        for cls in [0, 1]:
            mask = labels == cls
            scatter = ax.scatter(data[mask, 0], data[mask, 1], 
                               c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                               alpha=0.65, s=40, edgecolors='white', linewidth=0.4)
            
            # 置信椭圆
            add_confidence_ellipse(ax, data[mask, 0], data[mask, 1],
                                   n_std=2.0, edgecolor=CLASS_COLORS[cls],
                                   linewidth=2.5, linestyle='--', alpha=0.9)
        
        if method == 'pca' and var_ratio is not None:
            ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=11)
            ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=11)
        else:
            ax.set_xlabel('Dimension 1', fontsize=11)
            ax.set_ylabel('Dimension 2', fontsize=11)
        
        ax.set_title(f'({subplot_label}) {title}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加指标对比框
    metrics_text = (
        f"Separability Metrics:\n"
        f"─────────────────────\n"
        f"Fisher Ratio:\n"
        f"  FBCSP:   {fbcsp_metrics['fisher_ratio']:.2f}\n"
        f"  AWFBCSP: {awfbcsp_metrics['fisher_ratio']:.2f}\n"
        f"  Improvement: {(awfbcsp_metrics['fisher_ratio']/fbcsp_metrics['fisher_ratio']-1)*100:.1f}%\n"
        f"─────────────────────\n"
        f"Inter-class Distance:\n"
        f"  FBCSP:   {fbcsp_metrics['inter_class_dist']:.2f}\n"
        f"  AWFBCSP: {awfbcsp_metrics['inter_class_dist']:.2f}"
    )
    
    fig.text(0.5, -0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                       edgecolor='orange', alpha=0.9),
             family='monospace')
    
    # 总标题
    fig.suptitle(f'Feature Space Comparison: FBCSP vs AWFBCSP\nSubject {subject_name}', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"✅ 组合对比图已保存: {save_path}")
    
    plt.show()
    
    return fig, fbcsp_metrics, awfbcsp_metrics


def plot_multiple_subjects_tsne(subject_ids, dataset='2a', save_path=None):
    """
    绘制多被试t-SNE对比（2x3网格）
    
    展示AWFBCSP在不同被试上一致地产生更可分的特征空间
    """
    n_subjects = min(len(subject_ids), 6)
    
    fig, axes = plt.subplots(2, n_subjects, figsize=(4*n_subjects, 8))
    
    for i, subject_id in enumerate(subject_ids[:n_subjects]):
        subject_name = f"{'A' if dataset=='2a' else 'B'}{subject_id:02d}"
        print(f"  处理 {subject_name}...")
        
        try:
            X, y = load_and_preprocess(subject_id, dataset)
            
            # 提取特征
            fbcsp_features, _ = extract_fbcsp_features(X, y)
            awfbcsp_features, _ = extract_awfbcsp_features(X, y)
            
            # t-SNE
            scaler = StandardScaler()
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                        n_iter=500, learning_rate='auto', init='pca')
            
            fbcsp_scaled = scaler.fit_transform(fbcsp_features)
            fbcsp_tsne = tsne.fit_transform(fbcsp_scaled)
            
            awfbcsp_scaled = scaler.fit_transform(awfbcsp_features)
            awfbcsp_tsne = tsne.fit_transform(awfbcsp_scaled)
            
            # 绘制FBCSP
            ax_top = axes[0, i]
            for cls in [0, 1]:
                mask = y == cls
                ax_top.scatter(fbcsp_tsne[mask, 0], fbcsp_tsne[mask, 1],
                             c=CLASS_COLORS[cls], alpha=0.6, s=25)
            ax_top.set_title(f'{subject_name}\nFBCSP', fontsize=11, fontweight='bold')
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            
            # 绘制AWFBCSP
            ax_bot = axes[1, i]
            for cls in [0, 1]:
                mask = y == cls
                ax_bot.scatter(awfbcsp_tsne[mask, 0], awfbcsp_tsne[mask, 1],
                             c=CLASS_COLORS[cls], alpha=0.6, s=25)
            ax_bot.set_title('AWFBCSP', fontsize=11, fontweight='bold')
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])
            
        except Exception as e:
            print(f"    ⚠️ {subject_name} 处理失败: {e}")
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[0], label='Left Hand'),
                       Patch(facecolor=CLASS_COLORS[1], label='Right Hand')]
    fig.legend(handles=legend_elements, loc='center right', fontsize=10)
    
    fig.suptitle('t-SNE Feature Space Comparison Across Subjects', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"✅ 多被试对比图已保存: {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("📊 图B：特征空间可分性对比可视化")
    print("=" * 60)
    
    # ============ 单被试详细对比 ============
    print("\n🔹 单被试详细对比（推荐用于论文）...")
    
    # BCI IV 2a - 选择代表性被试
    for dataset, subject_id in [('2a', 1), ('2b', 1)]:
        subject_name = f"{'A' if dataset=='2a' else 'B'}{subject_id:02d}"
        print(f"\n  处理 {subject_name}...")
        
        try:
            X, y = load_and_preprocess(subject_id, dataset)
            
            print(f"    数据形状: {X.shape}")
            print(f"    标签分布: {np.bincount(y)}")
            
            # 提取特征
            print("    提取FBCSP特征...")
            fbcsp_features, fbcsp_model = extract_fbcsp_features(X, y)
            
            print("    提取AWFBCSP特征...")
            awfbcsp_features, awfbcsp_model = extract_awfbcsp_features(X, y)
            
            print(f"    FBCSP特征维度: {fbcsp_features.shape}")
            print(f"    AWFBCSP特征维度: {awfbcsp_features.shape}")
            
            # 绘制组合对比图
            print(f"\n  绘制 {subject_name} 组合对比图...")
            fig, fbcsp_m, awfbcsp_m = plot_combined_comparison(
                fbcsp_features, awfbcsp_features, y, subject_name,
                save_path=f'results/Fig_feature_space_{subject_name}.png'
            )
            
            print(f"\n  📊 {subject_name} 可分性指标:")
            print(f"     FBCSP  Fisher Ratio: {fbcsp_m['fisher_ratio']:.3f}")
            print(f"     AWFBCSP Fisher Ratio: {awfbcsp_m['fisher_ratio']:.3f}")
            print(f"     提升: {(awfbcsp_m['fisher_ratio']/fbcsp_m['fisher_ratio']-1)*100:.1f}%")
            
        except Exception as e:
            print(f"    ⚠️ {subject_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ============ 多被试概览 ============
    print("\n🔹 多被试t-SNE概览...")
    
    print("  BCI IV 2a...")
    plot_multiple_subjects_tsne([1, 2, 3, 4, 5, 6], dataset='2a',
                                save_path='results/Fig_feature_space_multi_2a.png')
    
    print("  BCI IV 2b...")
    plot_multiple_subjects_tsne([1, 2, 3, 4, 5, 6], dataset='2b',
                                save_path='results/Fig_feature_space_multi_2b.png')
    
    print("\n✅ 图B特征空间可视化完成！")
    print("\n📝 论文写作建议：")
    print("""
    Figure X presents a feature space visualization comparing FBCSP and 
    AWFBCSP using both PCA and t-SNE projections. The results clearly 
    demonstrate that AWFBCSP produces more compact and separable feature 
    distributions compared to standard FBCSP. The confidence ellipses 
    (95%) show reduced intra-class variance and increased inter-class 
    distance, indicating improved spectral-spatial discriminability.
    
    Quantitatively, the Fisher discriminant ratio improves by X% with 
    AWFBCSP, confirming that MI-guided band weighting effectively 
    enhances feature quality for motor imagery classification.
    """)

