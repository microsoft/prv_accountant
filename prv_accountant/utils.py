from typing import Iterable, Callable, Tuple
from collections import OrderedDict

from .privacy_random_variables import PrivacyRandomVariable, PrivacyRandomVariableTruncated

class PRVSequence:
    def __init__(self, prvs: OrderedDict) -> None:
        self.prvs: OrderedDict[PrivacyRandomVariable, int] = OrderedDict(prvs)

    def total_num_compositions(self) -> int:
        return sum(self.prvs.values())

    def append(self, prv:PrivacyRandomVariable, num_compositions: int = 1) -> None:
        self.prvs[prv] = self.prvs.get(prv, 0) + num_compositions

    def __iter__(self) -> Iterable[PrivacyRandomVariable]:
        return iter(self.prvs.keys())

    def items(self) -> Tuple[PrivacyRandomVariable, int]:
        return self.prvs.items()

    def map(self, fn: Callable[[PrivacyRandomVariable], PrivacyRandomVariable]) -> "PRVSequence":
        return PRVSequence({ fn(prv): n for prv, n in self.prvs.items() })
