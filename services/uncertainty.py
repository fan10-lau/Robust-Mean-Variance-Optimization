from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class UncertaintySpec:
    kind: str                   # type of uncrtainty (box/ellipsoid)
    epsilon: float              # sizing parameter (z-score for box; sqrt(chi2) for ellipsoid)
    T: int                      # sample size used to estimate mu/Q 

def theta_half_diag(Q: np.ndarray, T: int) -> np.ndarray:
    """
    Θ^{1/2} diagonal for mean uncertainty: σ_i / sqrt(T)
    (σ_i are the asset volatilities from the covariance Q)
    """
    return np.sqrt(np.clip(np.diag(Q), 0.0, None)) / np.sqrt(T)

# Common z-scores if you want to pass confidence levels instead of raw epsilon
Z_FROM_CL = {0.90: 1.645, 0.95: 1.960, 0.975: 2.241, 0.99: 2.576}
