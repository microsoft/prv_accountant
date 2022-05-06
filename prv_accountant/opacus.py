from typing import Dict, Tuple
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
from dataclasses import dataclass

try:
    from opacus.accountants.accountant import IAccountant
except ImportError:
    raise ImportError("Could not import opacus. Have you installed the PRV accountant with the opacus option like `pip install prv-accountant[opacus]?")


@dataclass
class Mechanism:
    sample_rate: float
    noise_multiplier: float


class Accountant(IAccountant):
    def __init__(self):
        self.steps: Dict[Mechanism, int] = dict()
    
    def step(self, *, noise_multiplier: float, sample_rate: float):
        m = Mechanism(sample_rate=sample_rate, noise_multiplier=noise_multiplier)
        self.steps[m] = self.steps.get(m, 0) + 1

    def get_epsilon(self, delta: float) -> float:
        prvs = [PoissonSubsampledGaussianMechanism() for m in self.steps.keys()]
        accountant = PRVAccountant(prvs=prvs)
        _, _, eps = accountant.compute_epsilon(delta=delta, num_self_compositions=self.steps.values())
        return eps