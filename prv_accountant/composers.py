# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from abc import ABC, abstractmethod
from scipy.fft import rfft, irfft
from scipy.signal import convolve
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
    def compute_composition(self, num_self_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """
        Abstract method to compute the composition of PRVs

        :param num_self_compositions: The number of compositions for each PRV with itself. The length of this sequence needs to
                                      match `self.prvs`. The total number of compositions is the sum of
                                      `num_self_compositions`.
        :type num_self_compositions: Sequence[int]
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

    def compute_composition(self, num_self_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """Compute the composition of the PRV `num_self_compositions` times with itself."""
        if len(num_self_compositions) != 1:
            raise ValueError("Length of `num_self_compositions` needs to match length of PRVs.")
        num_compositions = num_self_compositions[0]

        f_n = irfft(self.F**num_compositions)

        m = num_compositions-1
        if num_compositions % 2 == 0:
            m += len(f_n)//2
        f_n = np.roll(f_n, m)

        domain = self.domain.shift_right(self.domain.shifts()*(num_compositions-1))

        log_pmc_inf = sum(prv.log_pmc_inf*num_comp for prv, num_comp in zip(self.prvs, num_self_compositions))

        return DiscretePrivacyRandomVariable(f_n, domain, log_pmc_inf=log_pmc_inf)


class ConvolutionTree(Composer):
    def __init__(self, prvs: Sequence[DiscretePrivacyRandomVariable]) -> None:
        """
        Create a composer for efficiently composing heterogeneous PRVs.

        This composer runs in $n \\log n$ where $n$ is the total number of compositions. Hence, it isn't optimal for
        homogeneous composition. Use the Fourier composer for homogeneous composition instead.

        :param Sequence[DiscretePrivacyRandomVariable] prvs: Sequence of discrete PRVs to compose
        """
        super().__init__(prvs=prvs)

    def compute_composition(self, num_self_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """
        Compute the composition of PRVs.

        Since this composer is not efficient for homogeneous composition, each entry `num_self_compositions` is expected
        to be 1.

        :param num_self_compositions: The number of compositions for each PRV with itself.

        :type num_self_compositions: Sequence[int]
        """
        if (np.array(num_self_compositions) != 1).any():
            raise ValueError("Cannot handle homogeneous composition. Use Fourier composer for that.")
        prvs = self.prvs
        while len(prvs) > 1:
            if len(prvs) % 2 == 1:
                prvs_conv = [prvs.pop(0)]
            else:
                prvs_conv = []
            for prv_L, prv_R in zip(prvs[:-1:2], prvs[1::2]):
                prvs_conv.append(self._add_prvs(prv_L, prv_R))
            prvs = prvs_conv
        return prvs[0]

    def _add_prvs(self, prv_L: DiscretePrivacyRandomVariable,
                  prv_R: DiscretePrivacyRandomVariable) -> DiscretePrivacyRandomVariable:
        f = convolve(prv_L.pmf, prv_R.pmf, mode="same")
        domain = prv_L.domain.shift_right(prv_R.domain.shifts())
        log_pmc_inf = prv_L.log_pmc_inf + prv_R.log_pmc_inf
        return DiscretePrivacyRandomVariable(f, domain, log_pmc_inf=log_pmc_inf)


class Heterogeneous(Composer):
    def __init__(self, prvs: Sequence[DiscretePrivacyRandomVariable]) -> None:
        """
        Create a heterogeneous composer.

        This composer first composes identical PRVs with itself using Fourier composition followed by pairwise convolution for
        the remaining PRVs.

        :param Sequence[DiscretePrivacyRandomVariable] prvs: Sequence of discrete PRVs to compose
        """
        super().__init__(prvs)

        self.self_composers = [Fourier([prv]) for prv in prvs]

    def compute_composition(self, num_self_compositions: Sequence[int]) -> DiscretePrivacyRandomVariable:
        """
        Compute the composition of PRVs

        :param num_self_compositions: The number of composition for each PRV with itself. The length of this sequence needs to
                                      match `self.prvs`. The total number of compositions is the sum of
                                      `num_self_compositions`.

        :type num_self_compositions: Sequence[int]
        """
        if len(num_self_compositions) != len(self.prvs):
            raise ValueError("Length of `num_self_compositions` need to match number of PRVs passed to the composer.")
        self_composed_prvs = [sc.compute_composition([n]) for sc, n in zip(self.self_composers, num_self_compositions)]
        return ConvolutionTree(self_composed_prvs).compute_composition([1]*len(num_self_compositions))
