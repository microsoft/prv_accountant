# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import warnings
from typing import Tuple, Sequence, Optional, Union

from scipy import optimize
from abc import ABC, abstractmethod

from .other_accountants import RDP
from . import discretisers
from . import composers
from .domain import Domain
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .privacy_random_variables import PrivacyRandomVariableTruncated, PrivacyRandomVariable
from . import privacy_random_variables


def compute_safe_domain_size(prvs: Sequence[PrivacyRandomVariable], eps_error: float, delta_error: float, max_compositions: int) -> float:
    L = 0
    for prv in prvs:
        rdp = RDP(prv=prv, delta=delta_error/4)
        L = eps_error + rdp.compute_epsilon(max_compositions)[2]
        rdp = RDP(prv=prv, delta=delta_error/8/max_compositions)
        L = max(L, rdp.compute_epsilon(1)[2])
    L += eps_error + 3
    return L


class AbstractAccountant(ABC):
    @abstractmethod
    def compute_epsilon(self, num_compositions: Optional[int] = None)  -> Tuple[float, float, float]:
        pass


class PRVAccountantHomogeneous:
    def __init__(self, prv: PrivacyRandomVariable, delta: float, eps_error: float, max_compositions: int,
                 eps_max: Optional[float] = None):
        self.eps_error = eps_error
        self.delta = delta
        self.delta_error = delta/1000.0

        eta0 = self.delta_error/3
        mesh_size = 2*eps_error / np.sqrt(2*max_compositions*np.log(2/eta0))

        if eps_max:
            L = eps_max
            warnings.warn(f"Assuming that true epsilon < {eps_max}. If this is not a valid assumption set `eps_max=None`.")
        else:
            L = compute_safe_domain_size([prv], self.eps_error, self.delta_error, max_compositions)

        domain = Domain.create_aligned(-L, L, mesh_size)
        prv_trunc = PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max())
        dprv = discretisers.CellCentred().discretise(prv_trunc, domain)
        self.composer = composers.Fourier(dprv)


    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        return self.composer.compute_composition(num_compositions).compute_epsilon(self.delta, self.delta_error, self.eps_error)

    def compute_delta_upper(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return f_n.compute_delta_estimate(epsilon-self.eps_error)+self.delta_error

    def compute_delta_lower(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return f_n.compute_delta_estimate(epsilon+self.eps_error)-self.delta_error


class PRVAccountantHeterogenous(AbstractAccountant):
    def __init__(self, prvs: Union[Sequence[Tuple[PrivacyRandomVariable, int]], Sequence[PrivacyRandomVariable]],
                 delta: float, eps_error: float, eps_max: Optional[float] = None) -> None:
        if isinstance(prvs[0], PrivacyRandomVariable):
            prvs = [(prv, 1) for prv in prvs]

        self.delta = delta
        self.eps_error = eps_error
        self.delta_error = delta/1000.0

        self.prvs = [prv for prv, _ in prvs]
        self.n_comp = [n for _, n in prvs]
        max_compositions = sum(self.n_comp)

        if eps_max:
            L = eps_max
            warnings.warn(f"Assuming that true epsilon < {eps_max}. If this is not a valid assumption set `eps_max=None`.")
        else:
            L = compute_safe_domain_size(self.prvs, self.eps_error, self.delta_error, max_compositions)

        eta0 = self.delta_error/3
        mesh_size = 2*eps_error / np.sqrt(2*max_compositions*np.log(2/eta0))
        domain = Domain.create_aligned(-L, L, mesh_size)
        prvs_trunc = [ PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max()) for prv in self.prvs ]
        self.dprvs = [ discretisers.CellCentred().discretise(prv, domain) for prv in prvs_trunc ]

    def compute_epsilon(self, num_compositions: Optional[int] = None) -> Tuple[float, float, float]:
        """
        :param Optional[int] num_compositions: This value cannot be used with heterogenous composition
        """
        assert num_compositions is None
        f_n = self.compute_composition()
        return f_n.compute_epsilon(self.delta, self.delta_error, self.eps_error)

    def compute_composition(self) -> DiscretePrivacyRandomVariable:
        f_n_s = [composers.Fourier(dprv).compute_composition(n) for dprv, n in zip(self.dprvs, self.n_comp)]
        f_n = composers.Heterogenous(f_n_s).compute_composition()
        return f_n

    def compute_delta_upper(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return f_n.compute_delta_estimate(epsilon-self.eps_error)+self.delta_error

    def compute_delta_lower(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return f_n.compute_delta_estimate(epsilon+self.eps_error)-self.delta_error


class DPSGDAccountant(PRVAccountantHomogeneous):
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
        super().__init__(prv=prv, delta=delta, max_compositions=max_compositions, eps_error=eps_error)
                        

class Accountant(DPSGDAccountant):
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, max_compositions: int, eps_error: float = None,
                 mesh_size: float = None, verbose: bool = False) -> None:
        warnings.warn("`Accountant` will be deprecated. Use `DPSGDAccountant` instead.", DeprecationWarning)
        assert mesh_size is None
        super().__init__(noise_multiplier=noise_multiplier, sampling_probability=sampling_probability,
                         delta=delta, max_compositions=max_compositions, eps_error=eps_error)
      
