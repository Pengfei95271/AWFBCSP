import numpy as np
from scipy.signal import butter, filtfilt
from .csp import CSP
from .base_feature import BaseFeature

class FBCSP(BaseFeature):
    """Filter Bank Common Spatial Patterns - Improved Implementation"""

    def __init__(self, m_filters=6, freq_bands=None, sampling_rate=250):
        super().__init__()
        self.m_filters = m_filters
        self.sampling_rate = sampling_rate

        if freq_bands is None:
            self.freq_bands = [[8, 12], [12, 16], [16, 20],
                              [20, 24], [24, 30]]
        else:
            self.freq_bands = freq_bands

        self.fbcsp_filters_multi = []
        self.n_fbanks = len(self.freq_bands)

    def fit(self, X, y):
        """Fit FBCSP - Improved method"""
        X = self._validate_input(X)
        y = np.asarray(y)

        # Prepare multi-band data
        x_train_fb = self._prepare_multiband_data(X)
        
        # Get unique classes
        y_classes_unique = np.unique(y)
        n_classes = len(y_classes_unique)
        
        # Initialize CSP for each frequency band
        # CSP expects n_components (total features), not m_filters
        self.csp = CSP(n_components=self.m_filters * 2)

        def get_csp(x_train_fb, y_train_cls):
            """Train CSP for each frequency band"""
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        # Train for each class (one-vs-rest approach)
        self.fbcsp_filters_multi = []
        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            # Create binary labels: 0 for target class, 1 for others
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y))
            
            # Train CSP filters for this class
            fbcsp_filters = get_csp(x_train_fb, y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

        self.is_fitted = True
        return self

    def transform(self, X, class_idx=0):
        """Extract FBCSP features"""
        self._check_is_fitted()
        X = self._validate_input(X)

        # Prepare multi-band data
        x_data = self._prepare_multiband_data(X)
        
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        # Total features: 2 * m_filters * n_fbanks (first and last m filters for each band)
        x_features = np.zeros((n_trials, self.m_filters * 2 * self.n_fbanks), dtype=float)
        
        for i in range(n_fbanks):
            # Get CSP filters for this frequency band
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            
            for k in range(n_trials):
                x_trial = np.copy(x_data[i, k, :, :])
                csp_feat = self.csp._transform_single_trial(x_trial, eig_vectors)
                
                # Store features (first m and last m components)
                # csp_feat has 2*m_filters features: first m_filters + last m_filters
                for j in range(self.m_filters * 2):
                    x_features[k, i * self.m_filters * 2 + j] = csp_feat[j]

        return x_features

    def _prepare_multiband_data(self, X):
        """Prepare multi-band filtered data"""
        n_trials, n_channels, n_samples = X.shape
        n_fbanks = len(self.freq_bands)
        
        # Initialize multi-band data array
        x_train_fb = np.zeros((n_fbanks, n_trials, n_channels, n_samples))
        
        for band_idx, freq_band in enumerate(self.freq_bands):
            # Filter data for this frequency band
            X_filtered = self._filter_data(X, freq_band)
            x_train_fb[band_idx, :, :, :] = X_filtered
            
        return x_train_fb

    def _filter_data(self, X, freq_band):
        """Apply bandpass filter"""
        low, high = freq_band
        nyquist = self.sampling_rate / 2

        # Design butterworth filter
        b, a = butter(4, [low/nyquist, high/nyquist], btype='band')

        n_trials, n_channels, n_samples = X.shape
        X_filtered = np.zeros_like(X)

        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                X_filtered[trial_idx, ch_idx, :] = filtfilt(
                    b, a, X[trial_idx, ch_idx, :])

        return X_filtered