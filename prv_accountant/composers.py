# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from abc import ABC, abstractmethod
from scipy.fft import rfft, irfft
from typing import Sequence

from .discrete_privacy_random_variable import DiscretePrivacyRandomVariable


class Composer(ABC):
    def __init__(self, prvs: Sequence[DiscretePrivacyRandomVariable]) -> None:
        """
        Abstract base class for a composing mechanism.

        :param Sequence[DiscretePrivacyRandomVariable] prvs: Sequence of discrete PRVs to compose
        """
        self.prvs = prvs

    @abstractmethod
    def compute_composition(self, num_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """
        Abstract method to compute the composition of PRVs

        :param Sequence[int] num_composition: The number of composition for each PRV with itself.
                                              The length of this sequence needs to match `self.prvs`.
        """
        pass


class Fourier(Composer):
    def __init__(self, prvs: Sequence[DiscretePrivacyRandomVariable]) -> None:
        """
        Compute the composition of the PRVs using convolutions in Fourier space

        :param prvs: PRVs to compose.
                     The Fourier comopser only handles homogeneous composition and therefore prvs lengths of 1.
        """
        super().__init__(prvs=prvs)

        if len(self.prvs) != 1:
            raise ValueError("Fourier composer can only handle homogeneous composition")

        prv = self.prvs[0]

        if len(prv) % 2 != 0:
            raise ValueError("Can only compose evenly sized discrete PRVs")

        self.F = rfft(prv.pmf)
        self.domain = prv.domain

    def compute_composition(self, num_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """Compute the composition of the PRV `num_composition` times with itself."""
        if len(num_compositions) != 1:
            raise ValueError("Fourier composer can only handle homogeneous composition")
        num_compositions = num_compositions[0]

        f_n = irfft(self.F**num_compositions)

        m = num_compositions-1
        if num_compositions % 2 == 0:
            m += len(f_n)//2
        f_n = np.roll(f_n, m)

        domain = self.domain.shift_right(self.domain.shifts()*(num_compositions-1))

        return DiscretePrivacyRandomVariable(f_n, domain)
