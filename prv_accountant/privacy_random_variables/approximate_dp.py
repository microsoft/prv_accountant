# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .abstract_privacy_random_variable import PrivacyRandomVariable
from .pure_dp_mechanism import PureDPMechanism


class ApproximateDPMechanism(PrivacyRandomVariable):
    def __init__(self, epsilon: float, delta: float) -> None:
        self.delta = delta
        self.pure_dp = PureDPMechanism(eps=epsilon)

    def cdf(self, t):
        return self.pure_dp.cdf(t)

    def rdp(self, alpha):
        return self.pure_dp.rdp(alpha)

    @property
    def pm_inf(self) -> float:
        """Probability mass at infinity."""
        return self.delta
