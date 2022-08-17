# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from dataclasses import dataclass

from .domain import Domain

from typing import Tuple


@dataclass
class DiscretePrivacyRandomVariable:
    pmf: np.ndarray
    domain: Domain
    log_pmc_inf: float = 0.0  # log(1-pm_inf) where pm_inf is the probability mass at infinity

    def __len__(self) -> int:
        assert len(self.pmf) == len(self.domain)
        return len(self.pmf)

    def compute_epsilon(self, delta: float, delta_error: float, epsilon_error: float) -> Tuple[float, float, float]:
        delta_inf = 1 - np.exp(self.log_pmc_inf)
        delta_fin = (delta - delta_inf)/(1-delta_inf)

        if delta_fin <= 0:
            return (np.inf, np.inf, np.inf)

        if np.finfo(np.longdouble).eps*len(self.domain) > delta_fin - delta_error:
            raise ValueError("Floating point errors will dominate for such small values of delta. "
                             "Increase delta or reduce domain size.")

        t = self.domain.ts()
        p = self.pmf
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p*np.exp(-t)).cumsum())
        ndelta = np.exp(t) * d2-d1

        def find_epsilon(delta_target):
            i = np.searchsorted(ndelta, -delta_target, side='left')
            if i <= 0:
                raise RuntimeError("Cannot compute epsilon")
            return np.log((d1[i]-delta_target)/d2[i])

        eps_upper = find_epsilon(delta_fin - delta_error) + epsilon_error
        eps_lower = find_epsilon(delta_fin + delta_error) - epsilon_error
        eps_estimate = find_epsilon(delta_fin)
        return float(eps_lower), float(eps_estimate), float(eps_upper)

    def compute_delta_estimate(self, epsilon: float) -> float:
        t = self.domain.ts()
        delta_fin = float(np.where(t >= epsilon, self.pmf*(1.0 - np.exp(epsilon)*np.exp(-t)), 0.0).sum())
        delta_inf = 1 - np.exp(self.log_pmc_inf)
        return delta_fin*(1-delta_inf) + delta_inf
