# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import warnings
from typing import Tuple, Sequence, Optional

from .other_accountants import RDP
from . import discretisers
from . import composers
from .domain import Domain
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .privacy_random_variables import PrivacyRandomVariableTruncated, PrivacyRandomVariable
from . import privacy_random_variables


def compute_safe_domain_size(prvs: Sequence[PrivacyRandomVariable], max_compositions: Sequence[int],
                             eps_error: float, delta_error: float) -> float:
    """
    Compute a safe domain size for PRVS
    """
    total_compositions = sum(max_compositions)

    rdp = RDP(prvs=prvs)
    L = rdp.compute_epsilon(delta=delta_error/4, num_compositions=max_compositions)[2]

    for prv in prvs:
        rdp = RDP(prvs=[prv])
        L = max(L, rdp.compute_epsilon(delta=delta_error/8/total_compositions, num_compositions=[1])[2])
    L = max(L, eps_error) + 3
    return L


class PRVAccountant:
    def __init__(self, prvs: Sequence[PrivacyRandomVariable], eps_error: float, delta_error:float,
                 max_compositions: Sequence[int], eps_max: Optional[float] = None):
        """
        Privacy Random Variable Accountant for heterogenous composition

        :param prvs: Sequence of `PrivacyRandomVariable` to be composed. If passed as part of a tuple the
                     second element of the tuple indicates how often this PRV is to be composed
        :type prvs: `Union[Sequence[Tuple[PrivacyRandomVariable, int]], Sequence[PrivacyRandomVariable]]
        :param eps_error: Maximum error allowed in $\varepsilon$. Typically around 0.1
        :param delta_error: Maximum error allowed in $\delta$. typically around $10^{-3} \times \delta$
        :param Optional[float] eps_max: Maximum number of valid epsilon. If the true epsilon exceeds this value the
                                        privacy calculation may be off. Setting `eps_max` to `None` automatically computes
                                        a suitable `eps_max` if the PRV supports it.
        """
        self.eps_error = eps_error
        self.eps_error = eps_error
        self.delta_error = delta_error
        self.prvs = prvs
        self.max_compositions = max_compositions

        if len(max_compositions) != len(prvs):
            raise ValueError()

        if eps_max:
            L = eps_max
            warnings.warn(f"Assuming that true epsilon < {eps_max}. If this is not a valid assumption set `eps_max=None`.")
        else:
            L = compute_safe_domain_size(self.prvs, self.eps_error, self.delta_error, max_compositions)

        total_max_compositions = sum(max_compositions)

        eta0 = self.delta_error/3
        mesh_size = 2*eps_error / np.sqrt(2*total_max_compositions*np.log(2/eta0))
        domain = Domain.create_aligned(-L, L, mesh_size)

        tprvs = [PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max()) for prv in prvs]
        dprvs = [discretisers.Discretiser().discretise(tprv, domain) for tprv in tprvs]
        self.homogeneous_composers = [composers.Fourier(dprv) for dprv in dprvs]

    def compute_composition(self, num_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        if len(self.homogeneous_composers) != len(num_compositions):
            raise ValueError()

        if (np.array(self.max_compositions) < np.array(num_compositions)).any():
            raise ValueError()

        f_n_s = [composer.compute_composition(n) for composer, n in zip(self.homogeneous_composers, num_compositions)]
        f_n = composers.Heterogenous(f_n_s).compute_composition([1]*len(f_n_s))
        return f_n

    def compute_delta(self, epsilon: float, num_compositions: Sequence[int]) -> Tuple[float, float, float]:
        f_n = self.compute_composition(num_compositions)
        delta_lower = f_n.compute_delta_estimate(epsilon+self.eps_error)-self.delta_error
        delta_estim = f_n.compute_delta_estimate(epsilon)
        delta_upper = f_n.compute_delta_estimate(epsilon-self.eps_error)+self.delta_error
        return (delta_lower, delta_estim, delta_upper)

    def compute_epsilon(self, delta:float, num_compositions: Sequence[int]) -> Tuple[float, float, float]:
        """
        Compute bounds for epsilon

        :param float delta: Target delta
        :return: Return lower bound on $\varepsilon$, estimate for $\varepsilon$ and upper bound on $\varepsilon$
        :rtype: Tuple[float,float,float]
        """
        f_n = self.compute_composition(num_compositions)
        return f_n.compute_epsilon(delta, self.delta_error, self.eps_error)


class DPSGDAccountant:
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, max_compositions: int, eps_error: float = None) -> None:
        """
        Create an PRV accountant for DP-SGD

        For more details see https://arxiv.org/abs/2106.02848

        :param float noise_multiplier: Noise multiplier of the DP-SGD training
        :param float sampling_probability: Sampling probability of the training
        :param float delta: Target delta value
        :param int max_compositions: Max number of compositions this accountant is
                                     used for. This value is used to estimate a
                                     automatically determine a mesh size which
                                     influences the accuracy of the privacy budget.
        :param float eps_error: Allowed error in epsilon
        :param float mesh_size: Mesh size of the pdf discretisation.
                                (This is an upper bound the actual mesh size
                                could be smaller.)
        """

        prv = privacy_random_variables.PoissonSubsampledGaussianMechanism(
            sampling_probability=sampling_probability, noise_multiplier=noise_multiplier
        )
        self.delta = delta
        self.accountant = PRVAccountant(prvs=[prv], eps_error=eps_error, delta_error=self.delta/1000,
                                        max_compositions=[max_compositions])

    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        return self.accountant.compute_epsilon(self.delta, [num_compositions])



class Accountant(DPSGDAccountant):
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, max_compositions: int, eps_error: float = None,
                 mesh_size: float = None, verbose: bool = False) -> None:
        warnings.warn("`Accountant` will be deprecated. Use `DPSGDAccountant` instead.", DeprecationWarning)
        assert mesh_size is None
        super().__init__(noise_multiplier=noise_multiplier, sampling_probability=sampling_probability,
                         delta=delta, max_compositions=max_compositions, eps_error=eps_error)
      
