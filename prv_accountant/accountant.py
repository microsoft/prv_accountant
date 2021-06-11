import numpy as np
from typing import Tuple

from scipy import optimize

from .other_accountants import RDP
from . import discretisers
from . import composers
from .domain import Domain
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .privacy_random_variables import PrivacyRandomVariableTruncated
from . import privacy_random_variables


class Accountant:
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 delta: float, max_iterations: int, eps_error: float = None,
                 mesh_size: float = None, verbose: bool = False) -> None:
        """
        Create an PRV accountant

        For more details see https://arxiv.org/abs/2106.02848

        :param float noise_multiplier: Noise multiplier of the DP-SGD training
        :param float sampling_probability: Sampling probability of the training
        :param float delta: Target delta value
        :param int max_iterations: Max number of iterations this accountant is
                                   used for. This value is used to estimate a
                                   automatically determine a mesh size which
                                   influences the accuracy of the privacy budget.
        :param float eps_error: Allowed error in epsilon
        :param float mesh_size: Mesh size of the pdf discretisation.
                                (This is an upper bound the actual mesh size
                                could be smaller.)
        """
        self.noise_multiplier = noise_multiplier
        self.sampling_probability = sampling_probability
        self.delta = delta
        self.max_iterations = max_iterations
        self.delta_error = delta/1000.0

        eta0 = self.delta_error/3
        if mesh_size:
            if eps_error:
                raise ValueError("Cannot specify `eps_error` when `mesh_size` is specified.")
            mesh_size = mesh_size
            self.eps_error = mesh_size*np.sqrt(2*max_iterations*np.log(2/eta0))/2
        else:
            if not eps_error:
                raise ValueError("Need to specify either `eps_error` or `mesh_size`.")
            self.eps_error = eps_error
            mesh_size = 2*eps_error / np.sqrt(2*max_iterations*np.log(2/eta0))

        rdp = RDP(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            delta=self.delta_error/4)
        L = self.eps_error + rdp.compute_epsilon(max_iterations)[2]
        rdp = RDP(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            delta=self.delta_error/8/max_iterations)
        L = 3 + max(L, rdp.compute_epsilon(1)[2])

        domain = Domain.create_aligned(-L, L, mesh_size)
        if verbose:
            print(f"Initialising FDP accountant")
            print(f"Domain = {domain}")

        prv = privacy_random_variables.PoissonSubsampledGaussianMechanism(sampling_probability, noise_multiplier)
        
        prv = PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max())

        self.f_0 = discretisers.CellCentred().discretise(prv, domain)

        self.composer = composers.Fourier(self.f_0)

    def compute_compositions(self, num_compositions: int) -> DiscretePrivacyRandomVariable:
        if num_compositions > self.max_iterations:
            raise ValueError("Requested number of compositions exceeds the maximum number of compositions")
        return self.composer.compute_composition(num_compositions)

    def compute_delta_upper(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return self.compute_delta(f_n, epsilon-self.eps_error)+self.delta_error

    def compute_delta_lower(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        return self.compute_delta(f_n, epsilon+self.eps_error)-self.delta_error

    def compute_delta(self, f_n: DiscretePrivacyRandomVariable, epsilon: float) -> float:
        delta = 0.0
        for t_i, p_n_i in zip(f_n.domain, f_n.pmf):
            if t_i >= epsilon:
                delta += p_n_i*(1.0 - np.exp(epsilon)*np.exp(-t_i))
        return np.float64(delta)

    def compute_epsilon(self, iterations: int) -> Tuple[float, float, float]:
        """
        Compute epsilon bounds

        :param int iterations:
        :return Tuple[float, float, float] lower bound of true epsilon,
                                           approximation of true epsilon,
                                           upper bound of true epsilon
        """
        f_n = self.composer.compute_composition(iterations)
        eps_lower = optimize.root_scalar(lambda e: self.compute_delta_lower(f_n, e) - self.delta, bracket=(0, f_n.domain.t_max())).root
        eps_upper = optimize.root_scalar(lambda e: self.compute_delta_upper(f_n, e) - self.delta, bracket=(0, f_n.domain.t_max())).root
        eps_avg = 0.5*(eps_lower+eps_upper)
        return eps_lower, eps_avg, eps_upper
