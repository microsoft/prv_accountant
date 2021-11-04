# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
from scipy import stats

from prv_accountant import DPSGDAccountant, privacy_random_variables, PRVAccountantHeterogenous


def compute_delta_exact(eps, iters, sigma):
    mu = np.sqrt(iters)/sigma
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


class TestPRVAccountantHeterogenous:
    def test_analytic_solution(self):
        prv = privacy_random_variables.PoissonSubsampledGaussianMechanism(sampling_probability=1.0, noise_multiplier=10.0)
        accountant = PRVAccountantHeterogenous(prvs=[prv]*100, eps_error=0.01, delta_error=1e-12)

        f_n = accountant.compute_composition()
        delta_upper = accountant.compute_delta_upper(f_n, 4+0.01)
        delta_lower = accountant.compute_delta_lower(f_n, 4-0.01)

        delta_exact = compute_delta_exact(4, 100, 10.0)
        assert delta_upper == pytest.approx(delta_exact, rel=1e-3)
        assert delta_lower == pytest.approx(delta_exact, rel=1e-3)
