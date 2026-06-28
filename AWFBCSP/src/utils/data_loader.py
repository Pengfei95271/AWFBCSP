import os
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery, MotorImagery

def load_moabb_data(subjects=[1], paradigm_type='binary'):
    """
    Load and preprocess data from MOABB BNCI2014_001 dataset
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to load
    paradigm_type : str
        'binary' for left vs right hand (2 classes)
        'four_class' for left hand, right hand, feet, tongue (4 classes)
    """
    dataset = BNCI2014_001()
    
    if paradigm_type == 'binary':
        paradigm = LeftRightImagery()
    elif paradigm_type == 'four_class':
        paradigm = MotorImagery()
    else:
        raise ValueError("paradigm_type must be 'binary' or 'four_class'")

    X_list = []
    y_list = []

    for subject in subjects:
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
        
        if paradigm_type == 'binary':
            # 二分类：左手 vs 右手
            y = (labels == 'left_hand').astype(int)
        else:
            # 四分类：左手、右手、双脚、舌头
            label_mapping = {
                'left_hand': 0,
                'right_hand': 1, 
                'feet': 2,
                'tongue': 3
            }
            y = np.array([label_mapping[label] for label in labels])
        
        X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)

        X_list.append(X_reshaped)
        y_list.append(y)

    return X_list, y_list

def load_single_subject(subject_id=1, paradigm_type='binary', dataset='2a'):
    """Load data for a single subject
    
    Parameters:
    -----------
    subject_id : int
        被试编号 (1-9)
    paradigm_type : str
        'binary' for left vs right hand (2 classes)
        'four_class' for left hand, right hand, feet, tongue (4 classes)
    dataset : str
        数据集类型: '2a' 或 '2b'
    """
    if paradigm_type == 'four_class':
        # 使用本地数据加载四分类数据
        return load_local_four_class_data(subject_id)
    elif paradigm_type == 'binary':
        # 使用本地数据加载二分类数据（左右手）
        return load_local_binary_data(subject_id, dataset=dataset)
    else:
        # 使用MOABB加载数据
        X_list, y_list = load_moabb_data([subject_id], paradigm_type=paradigm_type)
        return X_list[0], y_list[0]

def load_local_four_class_data(subject_id=1):
    """加载本地四分类数据（训练+评估）"""
    dataset_path = 'dataset/bci_iv_2a'
    subject_name = f"A{subject_id:02d}"
    
    all_data = []
    all_labels = []
    
    # 加载训练数据 (T) - 有标签的数据
    train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
    train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
    
    if os.path.exists(train_data_path) and os.path.exists(train_label_path):
        train_data = np.load(train_data_path)
        train_labels = np.load(train_label_path)
        all_data.append(train_data)
        all_labels.append(train_labels)
        print(f"   加载训练数据: {train_data.shape[0]} trials")
    else:
        raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
    
    # 加载评估数据 (E) - 生成标签
    eval_data_path = os.path.join(dataset_path, f"{subject_name}E_data.npy")
    
    if os.path.exists(eval_data_path):
        eval_data = np.load(eval_data_path)
        # 评估数据标签：每类72个，按顺序排列
        eval_labels = np.repeat([1, 2, 3, 4], 72)  # 每类72个trials
        all_data.append(eval_data)
        all_labels.append(eval_labels)
        print(f"   加载评估数据: {eval_data.shape[0]} trials")
    else:
        print(f"   ⚠️  评估数据文件不存在: {eval_data_path}")
    
    # 合并数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    
    # 转换标签从1-4到0-3
    y = y - 1
    
    print(f"✅ 被试 {subject_name} 数据加载完成!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 标签分布: {np.bincount(y)}")
    
    return X, y

def load_local_binary_data(subject_id=1, dataset='2a'):
    """加载本地二分类数据（左右手，只使用训练数据）
    
    Parameters:
    -----------
    subject_id : int
        被试编号 (1-9)
    dataset : str
        数据集类型: '2a' 或 '2b'
    """
    if dataset == '2a':
        dataset_path = 'dataset/bci_iv_2a'
        subject_name = f"A{subject_id:02d}"
    elif dataset == '2b':
        dataset_path = 'dataset/bci_iv_2b/raw'
        subject_name = f"B{subject_id:02d}"
    else:
        raise ValueError("dataset must be '2a' or '2b'")
    
    # 只加载训练数据 (T) - 有标签的数据
    train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
    train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
    
    if os.path.exists(train_data_path) and os.path.exists(train_label_path):
        train_data = np.load(train_data_path)
        train_labels = np.load(train_label_path)
        
        # 检查是否是二分类数据（标签只有1和2）
        unique_labels = np.unique(train_labels)
        if len(unique_labels) == 2 and set(unique_labels) == {1, 2}:
            X = train_data
            y = train_labels
            print(f"   加载训练数据: {train_data.shape[0]} trials (二分类)")
        else:
            # 如果不是二分类，只取前两类
            print(f"   训练数据包含 {len(unique_labels)} 类，只取前两类用于二分类")
            binary_mask = (train_labels == 1) | (train_labels == 2)
            X = train_data[binary_mask]
            y = train_labels[binary_mask]
            print(f"   加载训练数据: {X.shape[0]} trials (二分类)")
    else:
        raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
    
    # 转换标签从1-2到0-1
    y = y - 1
    
    print(f"✅ 被试 {subject_name} 二分类数据加载完成!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 标签分布: {np.bincount(y)}")
    
    return X, y

def load_all_subjects_binary_data(subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """加载所有被试的二分类数据（左右手，只使用训练数据）"""
    dataset_path = 'dataset/bci_iv_2a'
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    print("📥 加载所有被试的二分类训练数据...")
    
    for subject_id in subject_ids:
        subject_name = f"A{subject_id:02d}"
        
        # 只加载训练数据 (T) - 有标签的数据
        train_data_path = os.path.join(dataset_path, f"{subject_name}T_data.npy")
        train_label_path = os.path.join(dataset_path, f"{subject_name}T_label.npy")
        
        if os.path.exists(train_data_path) and os.path.exists(train_label_path):
            train_data = np.load(train_data_path)
            train_labels = np.load(train_label_path)
            
            # 只取前两类用于二分类
            binary_mask = (train_labels == 1) | (train_labels == 2)
            train_data_binary = train_data[binary_mask]
            train_labels_binary = train_labels[binary_mask]
            
            all_data.append(train_data_binary)
            all_labels.append(train_labels_binary)
            all_subjects.extend([subject_id] * len(train_labels_binary))
            
            print(f"   被试 {subject_name}: {train_data_binary.shape[0]} trials (二分类)")
        else:
            print(f"   ⚠️  被试 {subject_name} 训练数据文件不存在")
    
    if not all_data:
        raise ValueError("没有找到有效的训练数据文件")
    
    # 合并所有被试数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    subjects = np.array(all_subjects)
    
    # 转换标签从1-2到0-1
    y = y - 1
    
    print(f"✅ 所有被试二分类数据加载完成!")
    print(f"   - 总样本数: {X.shape[0]}")
    print(f"   - 标签分布: {np.bincount(y)}")
    print(f"   - 被试分布: {np.bincount(subjects)}")
    
    return X, y, subjects

def load_multiple_subjects(subject_ids=[1, 2, 3]):
    """Load and concatenate data from multiple subjects"""
    X_list, y_list = load_moabb_data(subject_ids)

    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)

    return X_combined, y_combined

def get_dataset_info():
    """Get basic information about the dataset"""
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery()

    info = {
        'n_subjects': len(dataset.subject_list),
        'subject_list': dataset.subject_list,
        'paradigm': paradigm.__class__.__name__,
        'description': 'Motor imagery left vs right hand'
    }

    return info

def preprocess_data(X, filter_low=8.0, filter_high=30.0, baseline_correct=True):
    """Basic preprocessing of EEG data"""
    from scipy.signal import butter, filtfilt

    # Apply bandpass filter
    if filter_low is not None and filter_high is not None:
        nyquist = 250 / 2  # Assuming 250 Hz sampling rate
        b, a = butter(4, [filter_low/nyquist, filter_high/nyquist], btype='band')

        n_trials, n_channels, n_samples = X.shape
        X_filtered = np.zeros_like(X)

        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                X_filtered[trial_idx, ch_idx, :] = filtfilt(
                    b, a, X[trial_idx, ch_idx, :])

        X = X_filtered

    # Baseline correction (subtract mean of first 1 second)
    if baseline_correct:
        baseline_samples = int(250 * 1.0)  # 1 second at 250 Hz
        baseline_mean = np.mean(X[:, :, :baseline_samples], axis=2, keepdims=True)
        X = X - baseline_mean

    return X