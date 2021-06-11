import numpy as np
from dataclasses import dataclass

from .domain import Domain


@dataclass
class DiscretePrivacyRandomVariable:
    pmf: np.ndarray
    domain: Domain

    def __len__(self) -> int:
        assert len(self.pmf) == len(self.domain)
        return len(self.pmf)
