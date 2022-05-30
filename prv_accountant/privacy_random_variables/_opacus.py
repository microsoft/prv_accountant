from opacus.accountants import IAccountant
from typing import Optional, Dict

from prv_accountant import PoissonSubsampledGaussianMechanism, PRVAccountant


class OpacusAccountantHomogeneous(IAccountant):
    def __init__(self, max_steps: int, eps_error: float, delta_error: float):
        self.prv: Optional[PoissonSubsampledGaussianMechanism] = None
        self.accountant: Optional[PRVAccountant] = None
        self.num_compositions = 0
        self.max_compositions = max_steps
        self.delta_error = delta_error
        self.eps_error = eps_error


    def step(self, noise_multiplier: float, sample_rate: float):
        """
        Signal one optimization step

        Args:
            noise_multiplier: Current noise multiplier
            sample_rate: Current sample rate
        """
        if self.prv is None:
            self.prv = PoissonSubsampledGaussianMechanism(
                sampling_probability=sample_rate,
                noise_multiplier=noise_multiplier
            )
            self.accountant = PRVAccountant(
                prvs=self.prv,
                max_self_compositions=self.max_compositions,
                delta_error=self.delta_error,
                eps_error=self.eps_error
            )
        else:
            if self.prv.sigma != noise_multiplier:
                raise ValueError("Noise multiplier has to stay constant in OpacusAccountantHomogeneous.")
            if self.prv.p != sample_rate:
                raise ValueError("Sample rate has to stay constant in OpacusAccountantHomogeneous.")

        self.num_compositions += 1
        

    def get_epsilon(self, delta: float) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
        """
        assert self.accountant is not None
        _, _, eps = self.accountant.compute_epsilon(delta=delta, num_self_compositions=self.num_compositions)
        return eps

    def __len__(self) -> int:
        """
        Number of optimization steps taken so far
        """
        return self.num_compositions

    @classmethod
    def mechanism(cls) -> str:
        """
        Accounting mechanism name
        """
        return "prv-homogeneous"


class OpacusAccountant(IAccountant):
    def __init__(self, int, eps_error: float, delta_error: float):
        self.prvs_num_comp: Dict[PoissonSubsampledGaussianMechanism, int] = {}
        self.delta_error = delta_error
        self.eps_error = eps_error

    def step(self, noise_multiplier: float, sample_rate: float):
        """
        Signal one optimization step

        Args:
            noise_multiplier: Current noise multiplier
            sample_rate: Current sample rate
        """
        prv = PoissonSubsampledGaussianMechanism(
            sampling_probability=sample_rate,
            noise_multiplier=noise_multiplier
        )
        self.prvs_num_comp[prv] = self.prvs_num_comp.get(prv, 0) + 1
        

    def get_epsilon(self, delta: float) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
        """
        accountant = PRVAccountant(prvs=self.prvs_num_comp.keys(), max_self_compositions=self.prvs_num_comp.values(),
                                   delta_error=self.delta_error, eps_error=self.eps_error)
        _, _, eps = accountant.compute_epsilon(delta=delta, num_self_compositions=self.prvs_num_comp.values())
        return eps

    def __len__(self) -> int:
        """
        Number of optimization steps taken so far
        """
        return sum(self.prvs_num_comp.values())

    @classmethod
    def mechanism(cls) -> str:
        """
        Accounting mechanism name
        """
        return "prv"