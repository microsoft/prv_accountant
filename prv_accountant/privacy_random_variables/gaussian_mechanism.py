# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import scipy
from prv_accountant import PrivacyRandomVariable


class GaussianMechanism(PrivacyRandomVariable):
    def __init__(self, noise_multiplier: float, L2_sensitivity: float = 1.0) -> None:
        self.mu = L2_sensitivity/noise_multiplier
        assert self.mu > 0

    def cdf(self, t):
        return scipy.stats.norm.cdf(t/self.mu - self.mu/2)

    def rdp(self, alpha):
        return alpha*self.mu**2/2
