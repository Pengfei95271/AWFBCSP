# AWFBCSP — Adaptive Weighted Filter Bank CSP for Motor Imagery EEG

Mutual Information-Guided Adaptive Filter Bank CSP (AWFBCSP) for robust motor imagery (MI)
EEG classification. AWFBCSP augments FBCSP with a mutual-information-guided sub-band weighting
mechanism (temperature-scaled softmax, sqrt(w) variance-preserving reweighting, and a 4-D
cross-band interaction feature), improving accuracy, stability, and noise robustness without
increasing inference cost.

> Paper: Mutual Information-Guided Adaptive Filter Bank CSP for Robust Motor Imagery EEG
> Classification — under review, Biomedical Signal Processing and Control (Elsevier).

## Requirements
Python 3.10. Install with: pip install -r requirements.txt

## Datasets
Public, fetched automatically via MOABB (BNCI2014_001): BCI Competition IV-2a (22 ch, 4-class)
and IV-2b (3 ch, binary).

## Structure
- src/features/fbcsp_adaptive_weighted.py  — AWFBCSP (proposed method)
- src/features/csp.py, src/features/fbcsp.py — CSP / FBCSP baselines
- src/utils/data_loader.py — MOABB loading
- csp_traditional_classifiers_bci_iv_2a.py / _2b.py — main results (Tables 2-3)
- csp_traditional_classifiers_noise_robustness*.py, plot_noise_robustness_line_charts.py — Fig 5
- plot_band_weights_visualization.py — Fig 4; plot_boxplot_from_table.py — Fig 3
- measure_training_time.py, plot_training_time_comparison.py — Fig 6

Run the scripts from the repository root.

## Citation
Cao P., Liang T., Jia H., Grau Saldes A., Bolea Monte Y.,
"Mutual Information-Guided Adaptive Filter Bank CSP for Robust Motor Imagery EEG Classification",
Biomedical Signal Processing and Control (under review), 2026.

## License
MIT — see LICENSE.

## Acknowledgements
Supported by the China Scholarship Council (CSC) under Grant No. 202208390009.
