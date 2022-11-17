# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import warnings
from typing import Tuple, Sequence, Optional, Union

from .other_accountants import RDP
from . import discretisers
from . import composers
from .domain import Domain
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .privacy_random_variables import PrivacyRandomVariableTruncated, PrivacyRandomVariable
from . import privacy_random_variables


def compute_safe_domain_size(prvs: Sequence[PrivacyRandomVariable], max_self_compositions: Sequence[int],
                             eps_error: float, delta_error: float) -> float:
    """
    Compute a safe domain size for the discretisation of the PRVs

    For details about this algorithm see remark 5.6 in
    https://www.microsoft.com/en-us/research/publication/numerical-composition-of-differential-privacy/
    """
    total_compositions = sum(max_self_compositions)

    rdp = RDP(prvs=prvs)
    _, _, L_max = rdp.compute_epsilon(delta=delta_error/4, num_self_compositions=max_self_compositions)

    for prv in prvs:
        rdp = RDP(prvs=[prv])
        _, _, L = rdp.compute_epsilon(delta=delta_error/8/total_compositions, num_self_compositions=[1])
        L_max = max(L_max, L)
    L_max = max(L_max, eps_error) + 3
    return L_max


class PRVAccountant:
    def __init__(self, prvs: Union[PrivacyRandomVariable, Sequence[PrivacyRandomVariable]],
                 eps_error: float, delta_error: float,
                 max_self_compositions: Sequence[int] = None,
                 eps_max: Optional[float] = None):
        """
        Privacy Random Variable Accountant for heterogenous composition

        :param prvs: Sequence of `PrivacyRandomVariable` to be composed.
        :type prvs: `Sequence[PrivacyRandomVariable]`
        :param max_self_compositions: Maximum number of compositions of the PRV with itself.
        :type max_self_compositions: Sequence[int]
        :param eps_error: Maximum error allowed in $\varepsilon$. Typically around 0.1
        :param delta_error: Maximum error allowed in $\\delta$. typically around $10^{-3} \times \\delta$
        :param Optional[float] eps_max: Maximum number of valid epsilon. If the true epsilon exceeds this value the
                                        privacy calculation may be off. Setting `eps_max` to `None` automatically computes
                                        a suitable `eps_max` if the PRV supports it.
        """
        if isinstance(prvs, PrivacyRandomVariable):
            prvs = [prvs]
            if max_self_compositions is not None:
                max_self_compositions = [max_self_compositions]

        if max_self_compositions is None:
            max_self_compositions = [1]*len(prvs)

        self.eps_error = eps_error
        self.eps_error = eps_error
        self.delta_error = delta_error
        self.prvs = prvs
        self.max_self_compositions = max_self_compositions

        if len(max_self_compositions) != len(prvs):
            raise ValueError()

        if eps_max is not None:
            L = eps_max
            warnings.warn(f"Assuming that true epsilon < {eps_max}. If this is not a valid assumption set `eps_max=None`.")
        else:
            L = compute_safe_domain_size(self.prvs, max_self_compositions, eps_error=self.eps_error,
                                         delta_error=self.delta_error)

        total_max_self_compositions = sum(max_self_compositions)

        # See Theorem 5.5 in https://arxiv.org/pdf/2106.02848.pdf
        mesh_size = self.eps_error / np.sqrt(total_max_self_compositions/2*np.log(12/self.delta_error))
        domain = Domain.create_aligned(-L, L, mesh_size)

        tprvs = [PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max()) for prv in prvs]
        dprvs = [discretisers.CellCentred().discretise(tprv, domain) for tprv in tprvs]
        self.composer = composers.Heterogeneous(dprvs)

    def compute_composition(self, num_self_compositions: Union[Optional[int], Sequence[int]]) -> DiscretePrivacyRandomVariable:
        """
        Compute the composition of the PRVs

        :param Sequence[int] num_self_compositions: Number of compositions for each PRV with itself
        :return Composed PRV
        :rtype: DiscretePrivacyRandomVariable
        """
        if num_self_compositions is None:
            num_self_compositions = [1]*len(self.prvs)

        if isinstance(num_self_compositions, int):
            num_self_compositions = [num_self_compositions]

        if (np.array(self.max_self_compositions) < np.array(num_self_compositions)).any():
            raise ValueError("Requested number of compositions exceeds the maximum number of compositions")

        return self.composer.compute_composition(num_self_compositions=num_self_compositions)

    def compute_delta(self, epsilon: float, num_self_compositions: Sequence[int]) -> Tuple[float, float, float]:
        """
        Compute delta bounds for a given epsilon

        :param float epsilon: Target epsilon
        :param Sequence[int] num_self_compositions: Number of compositions for each PRV with itself
        :return: Return lower bound for $\\delta$, estimate for $\\delta$ and upper bound for $\\delta$
        :rtype: Tuple[float,float,float]
        """
        f_n = self.compute_composition(num_self_compositions)
        delta_lower = float(f_n.compute_delta_estimate(epsilon+self.eps_error)-self.delta_error)
        delta_estim = float(f_n.compute_delta_estimate(epsilon))
        delta_upper = float(f_n.compute_delta_estimate(epsilon-self.eps_error)+self.delta_error)
        return (delta_lower, delta_estim, delta_upper)

    def compute_epsilon(self, delta: float, num_self_compositions: Sequence[int]) -> Tuple[float, float, float]:
        """
        Compute epsilon bounds for a given delta

        :param float delta: Target delta
        :param Sequence[int] num_self_compositions: Number of compositions for each PRV with itself
        :return: Return lower bound for $\varepsilon$, estimate for $\varepsilon$ and upper bound for $\varepsilon$
        :rtype: Tuple[float,float,float]
        """
        f_n = self.compute_composition(num_self_compositions)
        return f_n.compute_epsilon(delta, self.delta_error, self.eps_error)


class Accountant:
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, max_compositions: int, eps_error: float = None,
                 mesh_size: float = None, verbose: bool = False) -> None:
        warnings.warn("`Accountant` will be deprecated. Use `PRVAccountant` with `PoissonSubsampledGaussianMechanism` "
                      "PRV instead.", DeprecationWarning)
        assert mesh_size is None

        prv = privacy_random_variables.PoissonSubsampledGaussianMechanism(
            sampling_probability=sampling_probability, noise_multiplier=noise_multiplier
        )
        self.delta = delta
        self.accountant = PRVAccountant(prvs=[prv], eps_error=eps_error, delta_error=self.delta/1000,
                                        max_self_compositions=[max_compositions])

    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        """
        Compute bounds for epsilon

        :param int num_compositions: Number of DP-SGD steps.
        :return: Return lower bound for $\varepsilon$, estimate for $\varepsilon$ and upper bound for $\varepsilon$
        :rtype: Tuple[float,float,float]
        """
        return self.accountant.compute_epsilon(self.delta, [num_compositions])
