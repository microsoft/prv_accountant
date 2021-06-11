# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from abc import ABC, abstractmethod

from .privacy_random_variables import PrivacyRandomVariable
from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from .domain import Domain


class Discretiser(ABC):
    @abstractmethod
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        pass


class LeftNode(Discretiser):
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        tC = domain.ts()
        tL = tC
        tR = tC + domain.dt()
        f = prv.probability(tL, tR)

        mean_d = (tC*f).sum()
        mean_c = prv.mean()

        mean_shift = mean_c - mean_d

        print(f"Discrete probablity mass {f.sum()}")
        print(f"mean_d = {mean_d}, mean_c = {mean_c}, d_mean = {mean_shift}")

        assert np.abs(mean_shift) < domain.dt()

        domain_shifted = domain.shift_right(mean_shift)

        return DiscretePrivacyRandomVariable(f, domain_shifted)


class CellCentred(Discretiser):
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        print(domain)
        tC = domain.ts()
        tL = tC - domain.dt()/2.0
        tR = tC + domain.dt()/2.0
        f = prv.probability(tL, tR)

        mean_d = sum(sorted(tC*f))
        mean_c = prv.mean()

        mean_shift = mean_c - mean_d

        print(f"Discrete probablity mass {f.sum()}")
        print(f"mean_d = {mean_d}, mean_c = {mean_c}, d_mean = {mean_shift}")

        if not (np.abs(mean_shift) < domain.dt()/2):
            raise RuntimeError("Discrete mean differs from continous mean by too much.")

        domain_shifted = domain.shift_right(mean_shift)


        return DiscretePrivacyRandomVariable(f, domain_shifted)

def kahansum(inputs):
    summ = c = 0
    for num in inputs:
        y = num - c
        t = summ + y
        c = (t - summ) - y
        summ = t
    return summ
