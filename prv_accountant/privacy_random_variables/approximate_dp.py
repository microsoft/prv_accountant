# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from .abstract_privacy_random_variable import PrivacyRandomVariable


class ApproximateDPMechanism(PrivacyRandomVariable):
    def __init__(self, epsilon: float, delta: float) -> None:
        self.epsilon = epsilon
        self.delta = delta

        assert self.epsilon > 0
        assert 0 <= self.delta and self.delta <= 1

    def cdf(self, t):
        return np.where(t < -self.epsilon, 0, np.where(t < self.epsilon, 1/(1+np.exp(self.epsilon)), 1))

    def rdp(self, alpha):
        """
        The Renyi divergence for the Laplace mechanism

        See Mironov, I. (2017) Renyi Differential Privacy
        """
        return 1/(alpha-1) * np.log(
            alpha/(2*alpha-1) * np.exp((alpha-1)*self.mu) + (alpha-1)/(2*alpha-1) * np.exp(-alpha*self.mu)
        )

    @property
    def pm_inf(self) -> float:
        return self.delta