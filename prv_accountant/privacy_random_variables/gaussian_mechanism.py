# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import scipy
import numpy as np
from .abstract_privacy_random_variable import PrivacyRandomVariable


class GaussianMechanism(PrivacyRandomVariable):
    def __init__(self, noise_multiplier: float, l2_sensitivity: float = 1.0) -> None:
        self.mu = l2_sensitivity/noise_multiplier
        assert self.mu > 0

    def cdf(self, t):
        return scipy.stats.norm.cdf(np.double(t/self.mu - self.mu/2))

    def rdp(self, alpha):
        return alpha*self.mu**2/2
