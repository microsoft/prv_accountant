# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
from scipy import stats

from prv_accountant import DPSGDAccountant, privacy_random_variables, PRVAccountant


def compute_delta_exact(eps, mu):
    return stats.norm.cdf(-eps/mu+mu/2)-np.exp(eps)*stats.norm.cdf(-eps/mu-mu/2)


class TestDPSGDAccountant:
    def test_analytic_solution(self):
        accountant = DPSGDAccountant(
            noise_multiplier=100.0,
            sampling_probability=1.0,
            delta=1e-8,
            eps_error=0.01,
            max_compositions=10000
        )

        f_n = accountant.compute_composition(10000)
        delta_upper = accountant.compute_delta_upper(f_n, 4+0.01)
        delta_lower = accountant.compute_delta_lower(f_n, 4-0.01)

        delta_exact = compute_delta_exact(4, 10000, 100.0)
        assert delta_upper == pytest.approx(delta_exact, rel=1e-3)
        assert delta_lower == pytest.approx(delta_exact, rel=1e-3)

    def test_throw_exceeding_max_compositions(self):
        with pytest.raises(ValueError):
            accountant = DPSGDAccountant(
                noise_multiplier=100.0,
                sampling_probability=1.0,
                delta=1e-8,
                eps_error=0.01,
                max_compositions=10000
            )

            accountant.compute_compositions(10001)

    def test_throw_error_small_delta(self):
        with pytest.raises(ValueError):
            accountant = DPSGDAccountant(
                noise_multiplier=4,
                sampling_probability=0.00038,
                delta=1.13e-18,
                eps_error=0.01,
                max_compositions=10000
            )
            accountant.compute_epsilon(1000)

    def test_invariance_max_compositions(self):
        noise_multiplier = 0.9
        sampling_probability = 256/100000
        target_delta = 1e-5

        eps_hi_target = DPSGDAccountant(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=4900,
            eps_error=0.1
        ).compute_epsilon(4900)[2]
        for m_c in range(4900, 5000):
            eps_hi = DPSGDAccountant(
                noise_multiplier=noise_multiplier,
                sampling_probability=sampling_probability,
                delta=target_delta,
                max_compositions=m_c,
                eps_error=0.1
            ).compute_epsilon(4900)[2]
            assert eps_hi == pytest.approx(eps_hi_target, 1e-3)



class TestPRVAccountant:
    @pytest.mark.parametrize("eps_error", [1e0, 1e-1, 1e-2])
    @pytest.mark.parametrize("delta_error", [1e-9, 1e-10, 1e-11])
    def test_gaussian_mechanism_analytic_homogeneous(self, eps_error, delta_error):
        noise_multiplier = 10.0
        compositions = 100
        prv = privacy_random_variables.PoissonSubsampledGaussianMechanism(sampling_probability=1.0, noise_multiplier=10.0)
        accountant = PRVAccountant(prvs=[prv], max_self_compositions=[compositions], eps_error=eps_error, delta_error=delta_error)

        delta_lower, _, delta_upper = accountant.compute_delta(4, [compositions])

        mu = np.sqrt(compositions)/noise_multiplier
        delta_exact = compute_delta_exact(4, mu)
        assert delta_lower <= delta_exact
        assert delta_exact <= delta_upper

    
    @pytest.mark.parametrize("eps_error", [1e0, 1e-1, 1e-2])
    @pytest.mark.parametrize("delta_error", [1e-9, 1e-10, 1e-11])
    def test_gaussian_mechanism_analytic_heterogeneous(self, eps_error, delta_error):
        prv_1 = privacy_random_variables.PoissonSubsampledGaussianMechanism(sampling_probability=1.0, noise_multiplier=10.0)
        prv_2 = privacy_random_variables.PoissonSubsampledGaussianMechanism(sampling_probability=1.0, noise_multiplier=5.0)
        accountant = PRVAccountant(prvs=[prv_1, prv_2], max_self_compositions=[50, 50], eps_error=eps_error, delta_error=delta_error)

        delta_lower, _, delta_upper = accountant.compute_delta(4, [50, 50])

        mu = np.sqrt(50*(10)**(-2) + 50*(5)**(-2))
        delta_exact = compute_delta_exact(4, mu)
        assert delta_upper == pytest.approx(delta_exact, rel=1e-3)
        assert delta_lower == pytest.approx(delta_exact, rel=1e-3)

    