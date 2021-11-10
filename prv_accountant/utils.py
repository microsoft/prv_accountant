from dataclasses import dataclass
from typing import Iterable, List, Dict, Callable

from .privacy_random_variables import PrivacyRandomVariable

@dataclass
class PRVSequence:
    prvs: Dict[PrivacyRandomVariable, int]

    def total_num_compositions(self) -> int:
        return sum(self.prvs.values())

    def append(self, prv:PrivacyRandomVariable, num_compositions: int = 1) -> None:
        self.prvs[prv] = self.prvs.get(prv, 0) + num_compositions

    def __iter__(self) -> Iterable[PrivacyRandomVariable]:
        return iter(self.prvs.keys())

    def map(self, fn: Callable[[PrivacyRandomVariable], PrivacyRandomVariable]) -> "PRVSequence":
        return PRVSequence({ fn(prv): n for prv, n in self.prvs.items() })
