# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from .abstract_privacy_random_variable import PrivacyRandomVariable


class PureDPMechanism(PrivacyRandomVariable):
    def __init__(self, eps: float) -> None:
        self.eps = eps
        assert self.eps >= 0.0

    def cdf(self, t):
        return np.where(t < -self.eps, 0, np.where(t < self.eps, 1/(1+np.exp(self.eps)), 1))

    def rdp(self, alpha):
        return (alpha-1)*self.eps/alpha
