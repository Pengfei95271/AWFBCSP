"""
Supplementary experiments for reviewer major comments.

This script focuses on reproducible add-on experiments:
1) Stronger baselines (CSP/FBCSP/FBCSP+FeatureSelection/AWFBCSP, optional Riemannian, optional EEGNet-like net).
2) Statistical tests across subjects (Wilcoxon + effect size).
3) Cross-subject robustness (LOSO).
4) AWFBCSP ablations (descriptor type, cross-band interaction, temperature tau, epsilon).
5) Scaling sensitivity analysis (to verify whether weighting effect is classifier-specific).

Important reproducibility settings (explicitly for manuscript Methods section):
- MI estimator for AW weights: plug-in mutual information on discretized descriptors.
- Discretization: equal-width binning, default n_bins=10, same setting for all subjects/datasets.
- Temperature tau and epsilon are explicit hyperparameters and logged.
- All train-only operations are strictly inside CV folds:
  covariance/CSP filters, MI weights, feature selector, standardization, classifier fitting.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.special import softmax
from scipy.stats import wilcoxon, norm
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features.csp import CSP
from src.features.fbcsp import FBCSP
from src.features.fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_trials(
    X: np.ndarray,
    fs: int = 250,
    band: Tuple[float, float] = (8.0, 30.0),
    baseline_sec: float = 0.5,
) -> np.ndarray:
    """Simple per-trial preprocessing without label leakage."""
    X_out = X.copy()
    baseline_n = int(fs * baseline_sec)
    if baseline_n > 0 and baseline_n < X_out.shape[-1]:
        X_out = X_out - X_out[:, :, :baseline_n].mean(axis=2, keepdims=True)

    nyq = fs / 2.0
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
    for i in range(X_out.shape[0]):
        for ch in range(X_out.shape[1]):
            X_out[i, ch, :] = filtfilt(b, a, X_out[i, ch, :])
    return X_out


def load_binary_subject_local(subject_id: int, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load local binary data without console-encoding side effects."""
    if dataset == "2a":
        root = Path("dataset/bci_iv_2a")
        prefix = f"A{subject_id:02d}"
    elif dataset == "2b":
        root = Path("dataset/bci_iv_2b/raw")
        prefix = f"B{subject_id:02d}"
    else:
        raise ValueError("dataset must be '2a' or '2b'")

    data_path = root / f"{prefix}T_data.npy"
    label_path = root / f"{prefix}T_label.npy"
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing files: {data_path} or {label_path}")

    X = np.load(data_path)
    y_raw = np.load(label_path)

    # Binary mapping: keep labels 1 and 2 only, then map to 0/1.
    mask = np.isin(y_raw, [1, 2])
    X = X[mask]
    y = y_raw[mask] - 1
    return X, y


def plugin_mi_discrete(x: np.ndarray, y: np.ndarray, n_bins: int, eps: float) -> float:
    """Plug-in MI estimator with equal-width discretization."""
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(int)
    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < eps:
        return 0.0
    edges = np.linspace(lo, hi + eps, n_bins + 1)
    x_bin = np.digitize(x, bins=edges[1:-1], right=False)
    return float(mutual_info_score(y, x_bin))


class ReviewReadyAWFBCSP(AdaptiveWeightedFBCSP):
    """
    AWFBCSP variant for reviewer-focused ablation control.

    Added controls:
    - descriptor_mode: "power" or "multi"
    - use_interaction: include/remove cross-band interaction features
    - temperature: tau in softmax
    - epsilon: numerical stability term
    - mi_bins: bins for plug-in MI discretization
    """

    def __init__(
        self,
        m_filters: int = 3,
        sampling_rate: int = 250,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        use_adaptive_weights: bool = True,
        use_temporal_windows: bool = False,
        use_erd_features: bool = False,
        use_multiscale: bool = False,
        descriptor_mode: str = "multi",
        use_interaction: bool = True,
        temperature: float = 0.5,
        epsilon: float = 1e-10,
        mi_bins: int = 10,
        quiet: bool = True,
    ) -> None:
        super().__init__(
            m_filters=m_filters,
            sampling_rate=sampling_rate,
            freq_bands=freq_bands,
            use_adaptive_weights=use_adaptive_weights,
            use_temporal_windows=use_temporal_windows,
            use_erd_features=use_erd_features,
            use_multiscale=use_multiscale,
        )
        if descriptor_mode not in {"power", "multi"}:
            raise ValueError("descriptor_mode must be 'power' or 'multi'")
        self.descriptor_mode = descriptor_mode
        self.use_interaction = use_interaction
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.mi_bins = int(mi_bins)
        self.quiet = bool(quiet)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                return super().fit(X, y)
        return super().fit(X, y)

    def _compute_band_weights(self, x_train_fb: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_fbanks = x_train_fb.shape[0]
        mi_scores = np.zeros(n_fbanks, dtype=float)
        y = np.asarray(y)

        for band_idx in range(n_fbanks):
            band_data = x_train_fb[band_idx]  # (trials, channels, samples)
            descriptors: List[np.ndarray] = []

            power = np.mean(band_data ** 2, axis=(1, 2))
            descriptors.append(power)

            if self.descriptor_mode == "multi":
                variance = np.var(band_data, axis=(1, 2))
                descriptors.append(variance)
                if band_data.shape[1] >= 13:
                    left = np.mean(band_data[:, [6, 7, 8], :] ** 2, axis=(1, 2))
                    right = np.mean(band_data[:, [10, 11, 12], :] ** 2, axis=(1, 2))
                    laterality = (left - right) / (left + right + self.epsilon)
                    descriptors.append(laterality)

            mi_vals = [plugin_mi_discrete(d, y, self.mi_bins, self.epsilon) for d in descriptors]
            mi_scores[band_idx] = float(np.mean(mi_vals))

        if np.allclose(mi_scores, mi_scores[0]):
            return np.ones(n_fbanks, dtype=float) / n_fbanks

        # Min-max then temperature softmax.
        mi_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + self.epsilon)
        logits = mi_norm / max(self.temperature, self.epsilon)
        return softmax(logits)

    def transform(self, X: np.ndarray, class_idx: int = 0) -> np.ndarray:
        """Override to support interaction-feature ablation."""
        self._check_is_fitted()
        X = self._validate_input(X)

        features = [self._extract_weighted_fbcsp_features(X, class_idx)]
        if self.use_temporal_windows:
            features.append(self._extract_temporal_features(X, class_idx))
        if self.use_erd_features:
            features.append(self._extract_erd_features(X))
        if self.use_adaptive_weights and self.use_interaction:
            features.append(self._extract_band_interaction_features(X, class_idx))
        return np.concatenate(features, axis=1)


@dataclass
class MethodSpec:
    name: str
    kind: str
    params: Dict
    selector_k: Optional[int] = None


def build_classifiers(seed: int) -> Dict[str, object]:
    return {
        "SVM-RBF": SVC(C=10, kernel="rbf", gamma=0.01, random_state=seed),
        "LDA": LinearDiscriminantAnalysis(),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=seed,
        ),
    }


def extract_features(
    spec: MethodSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if spec.kind == "csp":
        model = CSP(n_components=spec.params.get("n_components", 6))
        model.fit(X_train, y_train)
        return model.transform(X_train), model.transform(X_test)

    if spec.kind == "fbcsp":
        model = FBCSP(
            m_filters=spec.params.get("m_filters", 3),
            freq_bands=spec.params.get("freq_bands"),
            sampling_rate=fs,
        )
        model.fit(X_train, y_train)
        return model.transform(X_train, class_idx=0), model.transform(X_test, class_idx=0)

    if spec.kind == "awfbcsp":
        model = ReviewReadyAWFBCSP(
            m_filters=spec.params.get("m_filters", 3),
            sampling_rate=fs,
            freq_bands=spec.params.get("freq_bands"),
            use_adaptive_weights=spec.params.get("use_adaptive_weights", True),
            use_temporal_windows=spec.params.get("use_temporal_windows", False),
            use_erd_features=spec.params.get("use_erd_features", False),
            use_multiscale=spec.params.get("use_multiscale", False),
            descriptor_mode=spec.params.get("descriptor_mode", "multi"),
            use_interaction=spec.params.get("use_interaction", True),
            temperature=spec.params.get("temperature", 0.5),
            epsilon=spec.params.get("epsilon", 1e-10),
            mi_bins=spec.params.get("mi_bins", 10),
            quiet=spec.params.get("quiet", True),
        )
        model.fit(X_train, y_train)
        return model.transform(X_train, class_idx=0), model.transform(X_test, class_idx=0)

    raise ValueError(f"Unknown method kind: {spec.kind}")


def evaluate_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    method: MethodSpec,
    classifier,
    fs: int,
    n_splits: int,
    seed: int,
    use_scaler: bool = True,
) -> Dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_acc = []
    fold_train_time = []
    fold_infer_ms_per_trial = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t0 = time.perf_counter()
        F_train, F_test = extract_features(method, X_train, y_train, X_test, fs)

        if method.selector_k is not None and method.selector_k > 0:
            k = min(method.selector_k, F_train.shape[1])
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            F_train = selector.fit_transform(F_train, y_train)
            F_test = selector.transform(F_test)

        if use_scaler:
            scaler = StandardScaler()
            F_train = scaler.fit_transform(F_train)
            F_test = scaler.transform(F_test)

        clf = clone(classifier)
        clf.fit(F_train, y_train)
        t1 = time.perf_counter()

        p0 = time.perf_counter()
        pred = clf.predict(F_test)
        p1 = time.perf_counter()

        acc = accuracy_score(y_test, pred)
        fold_acc.append(float(acc))
        fold_train_time.append(float(t1 - t0))
        fold_infer_ms_per_trial.append(float((p1 - p0) * 1000.0 / len(y_test)))

    return {
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "fold_acc": fold_acc,
        "train_time_s_mean": float(np.mean(fold_train_time)),
        "infer_ms_per_trial_mean": float(np.mean(fold_infer_ms_per_trial)),
    }


def loso_evaluation(
    subjects_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    method: MethodSpec,
    classifier,
    fs: int,
    use_scaler: bool = True,
) -> Dict:
    test_subjects = sorted(subjects_data.keys())
    per_subject = {}
    for sid in test_subjects:
        X_test, y_test = subjects_data[sid]
        X_train = np.concatenate([subjects_data[k][0] for k in test_subjects if k != sid], axis=0)
        y_train = np.concatenate([subjects_data[k][1] for k in test_subjects if k != sid], axis=0)

        t0 = time.perf_counter()
        F_train, F_test = extract_features(method, X_train, y_train, X_test, fs)

        if method.selector_k is not None and method.selector_k > 0:
            k = min(method.selector_k, F_train.shape[1])
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            F_train = selector.fit_transform(F_train, y_train)
            F_test = selector.transform(F_test)

        if use_scaler:
            scaler = StandardScaler()
            F_train = scaler.fit_transform(F_train)
            F_test = scaler.transform(F_test)

        clf = clone(classifier)
        clf.fit(F_train, y_train)
        t1 = time.perf_counter()

        p0 = time.perf_counter()
        pred = clf.predict(F_test)
        p1 = time.perf_counter()
        acc = accuracy_score(y_test, pred)
        per_subject[sid] = {
            "acc": float(acc),
            "train_time_s": float(t1 - t0),
            "infer_ms_per_trial": float((p1 - p0) * 1000.0 / len(y_test)),
        }

    vals = [v["acc"] for v in per_subject.values()]
    return {
        "acc_mean": float(np.mean(vals)),
        "acc_std": float(np.std(vals)),
        "per_subject": per_subject,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size robust to non-normality."""
    x = np.asarray(x)
    y = np.asarray(y)
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    return float((gt - lt) / n)


def wilcoxon_with_effect(a: List[float], b: List[float]) -> Dict:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) != len(b_arr) or len(a_arr) < 3:
        return {"p": None, "statistic": None, "r": None, "cliffs_delta": None}

    stat, p = wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided")
    # Approximate z from p for effect size r=|z|/sqrt(n)
    z = norm.isf(max(p, 1e-16) / 2.0)
    r = float(abs(z) / np.sqrt(len(a_arr)))
    cd = cliffs_delta(a_arr, b_arr)
    return {"p": float(p), "statistic": float(stat), "r": r, "cliffs_delta": cd}


def try_riemannian_baseline(seed: int):
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
    except Exception:
        return None


def build_methods(args) -> List[MethodSpec]:
    base = [
        MethodSpec("CSP", "csp", {"n_components": 6}),
        MethodSpec("FBCSP", "fbcsp", {"m_filters": args.m_filters}),
        MethodSpec("FBCSP+MI-SelectK", "fbcsp", {"m_filters": args.m_filters}, selector_k=args.selector_k),
        MethodSpec(
            "AWFBCSP(full)",
            "awfbcsp",
            {
                "m_filters": args.m_filters,
                "descriptor_mode": "multi",
                "use_interaction": True,
                "temperature": args.tau,
                "epsilon": args.epsilon,
                "mi_bins": args.mi_bins,
                "use_adaptive_weights": True,
            },
        ),
    ]
    if args.run_ablation:
        base.extend(
            [
                MethodSpec(
                    "AWFBCSP(power_only)",
                    "awfbcsp",
                    {
                        "m_filters": args.m_filters,
                        "descriptor_mode": "power",
                        "use_interaction": True,
                        "temperature": args.tau,
                        "epsilon": args.epsilon,
                        "mi_bins": args.mi_bins,
                    },
                ),
                MethodSpec(
                    "AWFBCSP(no_interaction)",
                    "awfbcsp",
                    {
                        "m_filters": args.m_filters,
                        "descriptor_mode": "multi",
                        "use_interaction": False,
                        "temperature": args.tau,
                        "epsilon": args.epsilon,
                        "mi_bins": args.mi_bins,
                    },
                ),
                MethodSpec(
                    "AWFBCSP(tau=0.2)",
                    "awfbcsp",
                    {
                        "m_filters": args.m_filters,
                        "descriptor_mode": "multi",
                        "use_interaction": True,
                        "temperature": 0.2,
                        "epsilon": args.epsilon,
                        "mi_bins": args.mi_bins,
                    },
                ),
                MethodSpec(
                    "AWFBCSP(tau=1.0)",
                    "awfbcsp",
                    {
                        "m_filters": args.m_filters,
                        "descriptor_mode": "multi",
                        "use_interaction": True,
                        "temperature": 1.0,
                        "epsilon": args.epsilon,
                        "mi_bins": args.mi_bins,
                    },
                ),
                MethodSpec(
                    "AWFBCSP(eps=1e-8)",
                    "awfbcsp",
                    {
                        "m_filters": args.m_filters,
                        "descriptor_mode": "multi",
                        "use_interaction": True,
                        "temperature": args.tau,
                        "epsilon": 1e-8,
                        "mi_bins": args.mi_bins,
                    },
                ),
            ]
        )
    return base


def main():
    parser = argparse.ArgumentParser(description="Supplementary reviewer experiments for AWFBCSP.")
    parser.add_argument("--dataset", type=str, default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m-filters", type=int, default=3, dest="m_filters")
    parser.add_argument("--selector-k", type=int, default=20, dest="selector_k")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--mi-bins", type=int, default=10, dest="mi_bins")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-loso", action="store_true")
    parser.add_argument("--include-riemannian", action="store_true")
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/reviewer_supplementary")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess each subject.
    subjects_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    print("=" * 88)
    print("Reviewer supplementary experiments")
    print("=" * 88)
    print(f"Dataset={args.dataset}, Subjects={args.subjects}, folds={args.folds}, seed={args.seed}")
    print(f"MI config: plug-in discrete estimator, bins={args.mi_bins}, tau={args.tau}, epsilon={args.epsilon}")
    for sid in args.subjects:
        X, y = load_binary_subject_local(subject_id=sid, dataset=args.dataset)
        X = preprocess_trials(X, fs=args.fs, band=(8.0, 30.0), baseline_sec=0.5)
        subjects_data[sid] = (X, y)

    methods = build_methods(args)
    classifiers = build_classifiers(args.seed)
    use_scaler = not args.no_scaler

    subject_cv_rows = []
    per_subject_scores: Dict[str, Dict[str, List[float]]] = {}

    # Within-subject CV.
    for method in methods:
        per_subject_scores[method.name] = {clf_name: [] for clf_name in classifiers.keys()}
        for clf_name, clf in classifiers.items():
            for sid in sorted(subjects_data.keys()):
                X, y = subjects_data[sid]
                metrics = evaluate_subject_cv(
                    X=X,
                    y=y,
                    method=method,
                    classifier=clf,
                    fs=args.fs,
                    n_splits=args.folds,
                    seed=args.seed,
                    use_scaler=use_scaler,
                )
                per_subject_scores[method.name][clf_name].append(metrics["acc_mean"])
                subject_cv_rows.append(
                    {
                        "subject": sid,
                        "method": method.name,
                        "classifier": clf_name,
                        **metrics,
                    }
                )
                print(
                    f"[CV] S{sid:02d} | {method.name:<24} | {clf_name:<8} "
                    f"| acc={metrics['acc_mean']*100:.2f}%±{metrics['acc_std']*100:.2f}% "
                    f"| train={metrics['train_time_s_mean']:.3f}s | infer={metrics['infer_ms_per_trial_mean']:.3f}ms/trial"
                )

    cv_df = pd.DataFrame(subject_cv_rows)
    cv_df.to_csv(out_dir / "within_subject_cv_results.csv", index=False)

    # Statistical tests: AWFBCSP(full) vs baselines for each classifier.
    stat_rows = []
    target = "AWFBCSP(full)"
    if target in per_subject_scores:
        for clf_name in classifiers.keys():
            aw = per_subject_scores[target][clf_name]
            for method in methods:
                if method.name == target:
                    continue
                base = per_subject_scores[method.name][clf_name]
                if len(aw) == len(base) and len(aw) >= 3:
                    test = wilcoxon_with_effect(aw, base)
                    stat_rows.append(
                        {
                            "classifier": clf_name,
                            "compare": f"{target} vs {method.name}",
                            "aw_mean": float(np.mean(aw)),
                            "base_mean": float(np.mean(base)),
                            "delta_mean": float(np.mean(np.array(aw) - np.array(base))),
                            **test,
                        }
                    )
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / "statistical_tests_awfbcsp_vs_baselines.csv", index=False)

    # LOSO robustness.
    loso_rows = []
    if args.run_loso:
        loso_classifier = classifiers["LDA"]
        for method in methods:
            res = loso_evaluation(
                subjects_data=subjects_data,
                method=method,
                classifier=loso_classifier,
                fs=args.fs,
                use_scaler=use_scaler,
            )
            loso_rows.append(
                {
                    "method": method.name,
                    "classifier": "LDA",
                    "acc_mean": res["acc_mean"],
                    "acc_std": res["acc_std"],
                }
            )
            print(f"[LOSO] {method.name:<24} | LDA | acc={res['acc_mean']*100:.2f}%±{res['acc_std']*100:.2f}%")

            for sid, m in res["per_subject"].items():
                loso_rows.append(
                    {
                        "method": method.name,
                        "classifier": "LDA",
                        "test_subject": sid,
                        "acc": m["acc"],
                        "train_time_s": m["train_time_s"],
                        "infer_ms_per_trial": m["infer_ms_per_trial"],
                    }
                )
        pd.DataFrame(loso_rows).to_csv(out_dir / "loso_results.csv", index=False)

    # Scaling sensitivity for reviewer concern about z-score cancellation.
    scaling_rows = []
    compare_methods = [m for m in methods if m.name in {"FBCSP", "AWFBCSP(full)"}]
    for apply_scaler in [True, False]:
        for clf_name in ["SVM-RBF", "RF"]:
            clf = classifiers[clf_name]
            for method in compare_methods:
                subj_scores = []
                for sid in sorted(subjects_data.keys()):
                    X, y = subjects_data[sid]
                    metrics = evaluate_subject_cv(
                        X=X,
                        y=y,
                        method=method,
                        classifier=clf,
                        fs=args.fs,
                        n_splits=args.folds,
                        seed=args.seed,
                        use_scaler=apply_scaler,
                    )
                    subj_scores.append(metrics["acc_mean"])
                scaling_rows.append(
                    {
                        "method": method.name,
                        "classifier": clf_name,
                        "use_scaler": apply_scaler,
                        "acc_mean_subjectwise": float(np.mean(subj_scores)),
                        "acc_std_subjectwise": float(np.std(subj_scores)),
                    }
                )
    pd.DataFrame(scaling_rows).to_csv(out_dir / "scaling_sensitivity.csv", index=False)

    # Optional Riemannian baseline.
    riem_summary = {}
    if args.include_riemannian:
        pipe = try_riemannian_baseline(args.seed)
        if pipe is None:
            riem_summary["status"] = "skipped"
            riem_summary["reason"] = "pyriemann not installed"
            print("[INFO] Riemannian baseline skipped (pyriemann not installed).")
        else:
            riem_scores = []
            for sid in sorted(subjects_data.keys()):
                X, y = subjects_data[sid]
                skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                fold_scores = []
                for tr, te in skf.split(X, y):
                    clf = clone(pipe)
                    clf.fit(X[tr], y[tr])
                    pred = clf.predict(X[te])
                    fold_scores.append(accuracy_score(y[te], pred))
                riem_scores.append(float(np.mean(fold_scores)))
                print(f"[CV] S{sid:02d} | Riemannian+LR | acc={np.mean(fold_scores)*100:.2f}%")
            riem_summary["status"] = "ok"
            riem_summary["acc_mean_subjectwise"] = float(np.mean(riem_scores))
            riem_summary["acc_std_subjectwise"] = float(np.std(riem_scores))

    summary = {
        "config": vars(args),
        "n_subjects": len(subjects_data),
        "files": {
            "within_subject_cv": str(out_dir / "within_subject_cv_results.csv"),
            "stats": str(out_dir / "statistical_tests_awfbcsp_vs_baselines.csv"),
            "loso": str(out_dir / "loso_results.csv") if args.run_loso else None,
            "scaling_sensitivity": str(out_dir / "scaling_sensitivity.csv"),
        },
        "optional_riemannian": riem_summary,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("Done. Results saved to:", out_dir)
    print("=" * 88)


if __name__ == "__main__":
    main()

