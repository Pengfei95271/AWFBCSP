"""
快速测量：只测特征提取时间
"""
import numpy as np
import time
import sys
sys.path.append('src')
from features.csp import CSP
from features.fbcsp import FBCSP
from features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP as AWFBCSP
from utils.data_loader import load_single_subject

print("⏱️  快速特征提取时间测量\n")

# 测试单个被试
subject_id = 1
X_2a, y_2a = load_single_subject('dataset/bci_iv_2a', 1, '2a')
X_2b, y_2b = load_single_subject('dataset/bci_iv_2b/raw', 1, '2b')
y_2b = y_2b - 1

# 简单分割
split = int(0.8 * len(X_2a))

results = []

for dataset_name, X, y in [('2A', X_2a, y_2a), ('2B', X_2b, y_2b)]:
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"📊 BCI-IV-{dataset_name}:")
    
    freq_bands = [[4, 8], [8, 12], [12, 16], [16, 20], 
                  [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
    
    for method_name, Method in [('CSP', CSP), ('FBCSP', FBCSP), ('AWFBCSP', AWFBCSP)]:
        times = []
        for _ in range(10):  # 重复10次
            start = time.time()
            if method_name == 'CSP':
                model = Method(n_components=4)
            elif method_name == 'FBCSP':
                model = Method(m_filters=2, freq_bands=freq_bands, sampling_rate=250)
            else:  # AWFBCSP
                model = Method(m_filters=2, sampling_rate=250)
            model.fit(X_train, y_train)
            _ = model.transform(X_test)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # 转换为ms
        std_time = np.std(times) * 1000
        
        print(f"  {method_name:8s}: {avg_time:6.1f} ± {std_time:4.1f} ms")
        results.append((dataset_name, method_name, avg_time))

print("\n✅ 测量完成！")
print("\n相对比较:")
for dataset in ['2A', '2B']:
    print(f"\n{dataset}:")
    csp_time = [r[2] for r in results if r[0]==dataset and r[1]=='CSP'][0]
    for r in results:
        if r[0] == dataset:
            ratio = r[2] / csp_time
            print(f"  {r[1]:8s}: {ratio:.2f}× vs CSP")

