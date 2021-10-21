# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from abc import ABC, abstractmethod
from scipy.fft import rfft, irfft
from scipy.signal import convolve
from typing import Sequence

from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable


class Composer(ABC):
    @abstractmethod
    def compute_composition(self, num_compositions: int) -> DiscretePrivacyRandomVariable:
        pass


class Fourier(Composer):
    def __init__(self, prv: DiscretePrivacyRandomVariable) -> None:
        """
        Compute the composition of the PRVs using convolutions in Fourier space

        :param DiscretePrivacyRandomVariable prv: PRV to compose
        """
        if len(prv) % 2 != 0:
            raise ValueError("Can only compose evenly sized discrete PRVs")

        self.F = rfft(prv.pmf)
        self.domain = prv.domain

    def compute_composition(self, num_compositions: int) -> DiscretePrivacyRandomVariable:
        f_n = irfft(self.F**num_compositions)

        m = num_compositions-1
        if num_compositions % 2 == 0:
            m += len(f_n)//2
        f_n = np.roll(f_n, m)

        domain = self.domain.shift_right(self.domain.shifts()*(num_compositions-1))

        return DiscretePrivacyRandomVariable(f_n, domain)


class Heterogenous(Composer):
    def __init__(self, prvs: Sequence[DiscretePrivacyRandomVariable]) -> None:
        self.prvs = prvs

    def compute_composition(self) -> DiscretePrivacyRandomVariable:
        prvs = self.prvs
        while len(prvs) > 1:
            if len(prvs) % 2 == 1:
                prvs_conv = [prvs.pop(0)]
            else:
                prvs_conv = []
            for prv_L, prv_R in zip(prvs[:-1:2], prvs[1::2]):
                prvs_conv.append(self.add_prvs(prv_L, prv_R))
            prvs = prvs_conv
        return prvs[0]

    def add_prvs(self, prv_L: DiscretePrivacyRandomVariable, prv_R: DiscretePrivacyRandomVariable) -> DiscretePrivacyRandomVariable:
        f = convolve(prv_L.pmf, prv_R.pmf, mode="same")
        domain = prv_L.domain.shift_right(prv_R.domain.shifts())
        return DiscretePrivacyRandomVariable(f, domain)
