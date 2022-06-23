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

    def __len__(self) -> int:
        assert len(self.pmf) == len(self.domain)
        return len(self.pmf)

    def compute_epsilon(self, delta: float, delta_error: float, epsilon_error: float) -> Tuple[float, float, float]:
        if np.finfo(np.longdouble).eps*len(self.domain) > delta - delta_error:
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
            return np.log((d1[i-1]-delta_target)/d2[i-1])

        eps_upper = find_epsilon(delta - delta_error) + epsilon_error
        eps_lower = find_epsilon(delta + delta_error) - epsilon_error
        eps_estimate = find_epsilon(delta)
        return float(eps_lower), float(eps_estimate), float(eps_upper)

    def compute_epsilon_delta_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        t = self.domain.ts()
        p = self.pmf[t>=0.0]
        t = t[t>=0.0]
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p*np.exp(-t)).cumsum())
        deltas = d1 - np.exp(t) * d2
        return t, deltas

    def compute_f_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute trade-off function 

        See Proposition 2.11 in https://arxiv.org/pdf/1905.02383.pdf
        """
        epss, deltas = self.compute_epsilon_delta_estimates()
        fprs_seg = (deltas[:-1] - deltas[1:])/(np.exp(epss[1:]) - np.exp(epss[:-1]))
        fnrs_seg = -np.exp(epss)[:-1]*fprs_seg + 1-deltas[:-1]
        fnrs = np.concatenate([np.flip(fnrs_seg), fprs_seg])
        fprs = np.concatenate([np.flip(fprs_seg), fnrs_seg])
        return fprs, fnrs

    def compute_delta_estimate(self, epsilon: float) -> float:
        t = self.domain.ts()
        return float(np.where(t >= epsilon, self.pmf*(1.0 - np.exp(epsilon)*np.exp(-t)), 0.0).sum())

    def compute_membership_inference_advantage(self) -> Tuple[float, float, float]:
        deltas = self.compute_delta(epsilon=0.0)  # We know that eps=0 maximises MIAdv
        return tuple(reversed(deltas))
