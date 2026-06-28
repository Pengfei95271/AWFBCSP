import numpy as np
import scipy.linalg
from .base_feature import BaseFeature

class CSP(BaseFeature):
    """Common Spatial Patterns - Improved Implementation"""

    def __init__(self, n_components=6, reg=None, log=True, norm_trace=False):
        super().__init__()
        self.n_components = n_components
        self.m_filters = n_components // 2  # Number of filter pairs
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.filters_ = None
        self.patterns_ = None
        self.eig_values = None
        self.eig_vectors = None

    def fit(self, X, y):
        """Fit CSP filters - supports multi-class classification"""
        X = self._validate_input(X)
        y = np.asarray(y)

        x_data = np.copy(X)
        y_labels = np.copy(y)
        n_trials, n_channels, n_samples = x_data.shape
        
        # Get unique classes
        unique_classes = np.unique(y_labels)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            raise ValueError("CSP requires at least 2 classes")
        
        # For multi-class, use One-vs-Rest approach
        if n_classes > 2:
            # Compute class-specific covariance matrices
            cov_x = np.zeros((n_classes, n_channels, n_channels), dtype=float)
            class_counts = np.zeros(n_classes)
            
            for i in range(n_trials):
                x_trial = x_data[i, :, :]
                y_trial = y_labels[i]
                cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
                cov_x_trial /= np.trace(cov_x_trial)
                class_idx = np.where(unique_classes == y_trial)[0][0]
                cov_x[class_idx, :, :] += cov_x_trial
                class_counts[class_idx] += 1

            # Average covariance matrices for each class
            for cls in range(n_classes):
                if class_counts[cls] > 0:
                    cov_x[cls] /= class_counts[cls]
            
            # For multi-class, use the most discriminative class vs rest
            # Find the class with maximum variance difference
            class_variances = []
            for cls in range(n_classes):
                if class_counts[cls] > 0:
                    class_var = np.trace(cov_x[cls])
                    class_variances.append(class_var)
                else:
                    class_variances.append(0)
            
            # Use the two most different classes for CSP
            class_pairs = []
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    if class_counts[i] > 0 and class_counts[j] > 0:
                        diff = abs(class_variances[i] - class_variances[j])
                        class_pairs.append((i, j, diff))
            
            if len(class_pairs) == 0:
                # Fallback: use first two classes
                cls1, cls2 = 0, 1
            else:
                # Sort by difference and take the most discriminative pair
                class_pairs.sort(key=lambda x: x[2], reverse=True)
                best_pair = class_pairs[0]
                cls1, cls2 = best_pair[0], best_pair[1]
            
            # Use the two most discriminative classes
            cov_class1 = cov_x[cls1]
            cov_class2 = cov_x[cls2]
            
        else:
            # Binary classification - original method
            cov_x = np.zeros((2, n_channels, n_channels), dtype=float)
            for i in range(n_trials):
                x_trial = x_data[i, :, :]
                y_trial = y_labels[i]
                cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
                cov_x_trial /= np.trace(cov_x_trial)
                cov_x[y_trial, :, :] += cov_x_trial

            cov_x = np.asarray([cov_x[cls]/np.sum(y_labels==cls) for cls in range(2)])
            cov_class1 = cov_x[0]
            cov_class2 = cov_x[1]
        
        # Regularization if specified
        if self.reg is not None:
            reg_matrix = self.reg * np.eye(n_channels)
            cov_class1 += reg_matrix
            cov_class2 += reg_matrix

        # Solve generalized eigenvalue problem
        cov_combined = cov_class1 + cov_class2
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_class1)
        
        # Sort by absolute eigenvalues (descending)
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:, sort_indices]
        u_mat = np.transpose(u_mat)

        self.eig_values = eig_values
        self.eig_vectors = u_mat
        self.filters_ = u_mat
        self.patterns_ = np.linalg.pinv(self.filters_)
        self.is_fitted = True

        return eig_values, u_mat

    def transform(self, X):
        """Extract CSP features"""
        self._check_is_fitted()
        X = self._validate_input(X)

        n_trials, n_channels, n_samples = X.shape
        features = np.zeros((n_trials, self.n_components))

        for trial_idx in range(n_trials):
            x_trial = X[trial_idx]
            # Use the improved transform method
            csp_feat = self._transform_single_trial(x_trial)
            features[trial_idx] = csp_feat

        return features

    def _transform_single_trial(self, x_trial, eig_vectors=None):
        """Transform a single trial - improved method"""
        if eig_vectors is None:
            eig_vectors = self.eig_vectors
            
        # Apply CSP filters
        z_trial = np.matmul(eig_vectors, x_trial)
        
        # Select first m and last m components (most discriminative)
        z_trial_selected = z_trial[:self.m_filters, :]
        z_trial_selected = np.append(z_trial_selected, z_trial[-self.m_filters:, :], axis=0)
        
        # Compute variance features
        sum_z2 = np.sum(z_trial_selected**2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z ** 2)/z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)
        
        # Normalize by sum of variances
        sum_var_z = sum(var_z)
        
        if self.log:
            # Return log variance features (standard CSP approach)
            return np.log(var_z/sum_var_z + 1e-8)
        else:
            return var_z/sum_var_z

    def _compute_covariance(self, X):
        """Compute average covariance matrix - legacy method"""
        n_trials, n_channels, n_samples = X.shape
        cov = np.zeros((n_channels, n_channels))

        for trial_idx in range(n_trials):
            trial_data = X[trial_idx]
            if self.norm_trace:
                trial_cov = np.cov(trial_data)
                trial_cov /= np.trace(trial_cov)
            else:
                trial_cov = np.cov(trial_data)
            cov += trial_cov

        return cov / n_trials