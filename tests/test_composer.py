# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import binom, norm

from prv_accountant import composers
from prv_accountant.discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from prv_accountant.domain import Domain


class TestFourier:
    def test_unity(self):
        domain = Domain(0, 100, 100)
        pmf = binom.pmf(domain.ts().astype(np.double), 100, 0.4)
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        composer = composers.Fourier([prv])

        f_n = composer.compute_composition([1])

        assert_array_almost_equal(f_n.pmf, prv.pmf)

    def test_gaussian_analytical(self):
        """
        Test that the convolution of Gaussians gives the analytical solution
        """

        domain = Domain(-40, 40, 1_000_000)
        pmf = norm.pdf(domain.ts(), 1, 1)*domain.dt()
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        assert pmf.sum() == pytest.approx(1)

        composer = composers.Fourier([prv])

        f_n = composer.compute_composition([3])

        mean = np.dot(f_n.pmf, domain.ts())
        var = np.dot(f_n.pmf, (domain.ts()-mean)**2)
        assert mean == pytest.approx(3*1, abs=1e-4)
        assert var == pytest.approx(3*(1**2), abs=1e-6)

    def test_raise_heterogeneous(self):
        domain = Domain(-20, 20, 100000)
        pmf = norm.pdf(domain.ts(), 1, 1)*domain.dt()
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        with pytest.raises(ValueError):
            composer = composers.Fourier([prv, prv])


class TestConvolutionTree:
    def test_unity(self):
        domain = Domain(0, 100, 100)
        pmf = binom.pmf(domain.ts().astype(np.double), 100, 0.4)
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        composer = composers.ConvolutionTree([prv])

        f_n = composer.compute_composition([1])

        assert_array_almost_equal(f_n.pmf, prv.pmf)

    def test_gaussian_analytical(self):
        domain = Domain(-40, 40, 1_000_000)
        prv_1 = DiscretePrivacyRandomVariable(norm.pdf(domain.ts(), 2, 1)*domain.dt(), domain)
        prv_2 = DiscretePrivacyRandomVariable(norm.pdf(domain.ts(), -1, 3/2)*domain.dt(), domain)

        composer = composers.ConvolutionTree([prv_1, prv_2])

        f_n = composer.compute_composition([1, 1])

        mean = np.dot(f_n.pmf, domain.ts())
        var = np.dot(f_n.pmf, (domain.ts()-mean)**2)
        assert mean == pytest.approx(2-1, abs=1e-4)
        assert var == pytest.approx(1**2+(3/2)**2, abs=1e-6)

    def test_raise_homogeneous(self):
        domain = Domain(-20, 20, 100000)
        pmf = norm.pdf(domain.ts(), 1, 1)*domain.dt()
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        with pytest.raises(ValueError):
            composer = composers.ConvolutionTree([prv, prv])
            composer.compute_composition([2, 1])


class TestHeterogeneous:
    def test_unity(self):
        domain = Domain(0, 100, 100)
        pmf = binom.pmf(domain.ts().astype(np.double), 100, 0.4)
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        composer = composers.Heterogeneous([prv])

        f_n = composer.compute_composition([1])

        assert_array_almost_equal(f_n.pmf, prv.pmf)

    def test_gaussian_analytical(self):
        domain = Domain(-40, 40, 1_000_000)
        prv_1 = DiscretePrivacyRandomVariable(norm.pdf(domain.ts(), 2, 3)*domain.dt(), domain)
        prv_2 = DiscretePrivacyRandomVariable(norm.pdf(domain.ts(), -1, 4)*domain.dt(), domain)

        composer = composers.Heterogeneous([prv_1, prv_2])

        f_n = composer.compute_composition([1, 2])

        mean = np.dot(f_n.pmf, domain.ts())
        var = np.dot(f_n.pmf, (domain.ts()-mean)**2)
        assert mean == pytest.approx(2-2*1, abs=1e-4)
        assert var == pytest.approx(3**2+2*(4**2), abs=1e-6)
