# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from .abstract_privacy_random_variable import PrivacyRandomVariable


class LaplaceMechanism(PrivacyRandomVariable):
    def __init__(self, mu: float) -> None:
        self.mu = mu
        assert self.mu > 0

    def cdf(self, t):
        return np.where(
            t >= self.mu,
            1,
            np.where(
                np.logical_and(-self.mu < t, t < self.mu),
                1/2*np.exp(1/2*(t-self.mu)),
                0
            )
        )

    def rdp(self, alpha):
        """
        The Renyi divergence for the Laplace mechanism

        See Mironov, I. (2017) Renyi Differential Privacy
        """
        return 1/(alpha-1) * np.log(
            alpha/(2*alpha-1) * np.exp((alpha-1)*self.mu) + (alpha-1)/(2*alpha-1) * np.exp(-alpha*self.mu)
        )
