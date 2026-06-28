"""Feature extraction: CSP, FBCSP, and Adaptive Weighted FBCSP (AWFBCSP)."""
from .base_feature import BaseFeature
from .csp import CSP
from .fbcsp import FBCSP
from .fbcsp_adaptive_weighted import AdaptiveWeightedFBCSP
__all__ = ["BaseFeature", "CSP", "FBCSP", "AdaptiveWeightedFBCSP"]
