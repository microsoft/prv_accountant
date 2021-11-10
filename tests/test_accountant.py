# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
from scipy import stats

from prv_accountant import Accountant


def compute_delta_exact(eps, iters, sigma):
    mu = np.sqrt(iters)/sigma
    return stats.norm.cdf(-eps/mu+mu/2)-np.exp(eps)*stats.norm.cdf(-eps/mu-mu/2)


class TestAccountant:
    def test_analytic_solution(self):
        accountant = Accountant(
            noise_multiplier=100.0,
            sampling_probability=1.0,
            delta=1e-8,
            eps_error=0.01,
            max_compositions=10000
        )

        f_n = accountant.compute_compositions(10000)
        delta_upper = accountant.compute_delta_upper(f_n, 4+0.01)
        delta_lower = accountant.compute_delta_lower(f_n, 4-0.01)

        delta_exact = compute_delta_exact(4, 10000, 100.0)
        assert delta_upper == pytest.approx(delta_exact, rel=1e-3)
        assert delta_lower == pytest.approx(delta_exact, rel=1e-3)

<<<<<<< HEAD
    def test_throw_error_small_delta(self):
        with pytest.raises(ValueError):
            accountant = Accountant(
                noise_multiplier=4,
                sampling_probability=0.00038,
                delta=1.13e-18,
                eps_error=0.01,
                max_compositions=10000
            )
            accountant.compute_epsilon(1000)

=======
    def test_invariance_max_compositions(self):
        noise_multiplier = 0.9
        sampling_probability = 256/100000
        target_delta = 1e-5

        eps_hi_target = Accountant(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=4900,
            eps_error=0.1
        ).compute_epsilon(4900)[2]
        for m_c in range(4900, 5000):
            eps_hi = Accountant(
                noise_multiplier=noise_multiplier,
                sampling_probability=sampling_probability,
                delta=target_delta,
                max_compositions=m_c,
                eps_error=0.1
            ).compute_epsilon(4900)[2]
            assert eps_hi == pytest.approx(eps_hi_target, 1e-3)
>>>>>>> main
