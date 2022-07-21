# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from abc import ABC, abstractmethod
from scipy import integrate


class PrivacyRandomVariable(ABC):
    def mean(self) -> float:
        """Compute the mean of the random variable."""
        raise NotImplementedError(f"{type(self)} has not provided an implementation for a mean computation.")

    def probability(self, a, b):
        """Compute the probability mass on the interval [a, b]."""
        return self.cdf(b) - self.cdf(a)

    def pdf(self, t):
        """
        Compute the probability density function of this privacy random variable at point t
        conditioned on the value being finite.
        """
        raise NotImplementedError(f"{type(self)} has not provided an implementation for a pdf.")

    @abstractmethod
    def cdf(self, t):
        """
        Compute the cumulative distribution function of this privacy random variable at point t
        conditioned on the value being finite.
        """
        pass

    def rdp(self, alpha: float) -> float:
        """Compute RDP of this mechanism of order alpha conditioned on the value being finite."""
        raise NotImplementedError(f"{type(self)} has not provided an implementation for rdp.")

    @property
    def pm_inf(self) -> float:
        """Probability mass at infinity."""
        return 0.0


class PrivacyRandomVariableTruncated:
    def __init__(self, prv, t_min: float, t_max: float) -> None:
        self.prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self.remaining_mass = self.prv.cdf(t_max) - self.prv.cdf(t_min)

    def mean(self) -> float:
        points = [self.t_min, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, self.t_max]
        m = 0.0
        for L, R in zip(points[:-1], points[1:]):
            I, err = integrate.quad(self.cdf, L, R, limit=500)

            m += (
                R*self.cdf(R) -
                L*self.cdf(L) -
                I
            )

        return m

    def probability(self, a, b):
        a = np.clip(a, self.t_min, self.t_max)
        b = np.clip(b, self.t_min, self.t_max)
        return self.prv.probability(a, b) / self.remaining_mass

    def pdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.pdf(t)/self.remaining_mass, 0))

    def cdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.cdf(t)/self.remaining_mass, 1))

    @property
    def pm_inf(self) -> float:
        return self.prv.pm_inf
