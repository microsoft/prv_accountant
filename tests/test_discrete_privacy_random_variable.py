# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np

from prv_accountant import PRVAccountant, ApproximateDPMechanism, PoissonSubsampledGaussianMechanism
from prv_accountant.discrete_privacy_random_variable import convert_epsilon_deltas_to_trade_off_curve


class TestDiscretePrivacyRandomVariable:
    @pytest.mark.parametrize("delta", [1e-5, 1e-6, 1e-7])
    def test_inverse_privacy_curve_finite(self, delta: float):
        delta_error = delta/100
        prv = PoissonSubsampledGaussianMechanism(1e-3, 1.0)
        acc = PRVAccountant(prv, 0.1, delta_error=delta_error, max_self_compositions=100)

        f_n = acc.compute_composition(100)

        _, epsilon_estimate, _ = f_n.compute_epsilon(delta, delta_error, 0.1)
        delta_estimate = f_n.compute_delta_estimate(epsilon_estimate)

        assert delta_estimate == pytest.approx(delta)

    @pytest.mark.parametrize("delta", [1e-3, 1e-4, 1e-5])
    def test_inverse_privacy_curve_infinite(self, delta: float):
        delta_error = delta/100
        prv = ApproximateDPMechanism(epsilon=0.01, delta=1e-8)
        acc = PRVAccountant(prv, 0.1, delta_error=delta_error, max_self_compositions=100)

        f_n = acc.compute_composition(100)

        _, epsilon_estimate, _ = f_n.compute_epsilon(delta, delta_error, 0.1)
        delta_estimate = f_n.compute_delta_estimate(epsilon_estimate)

        assert delta_estimate == pytest.approx(delta)

    def test_error_terms(self):
        delta_error = 1e-5
        eps_error = 0.1
        prv = PoissonSubsampledGaussianMechanism(5e-3, noise_multiplier=1.0)
        acc = PRVAccountant(prv, eps_error=eps_error, delta_error=delta_error, max_self_compositions=1000)

        f_n = acc.compute_composition(1000)

        epss, deltas = f_n.compute_epsilon_delta_estimates()

        epss_upper = epss + eps_error
        deltas_upper = deltas + delta_error
        for eps, delta in zip(epss_upper, deltas_upper):
            if np.finfo(np.longdouble).eps*len(f_n.domain) < delta - delta_error:
                _, _, eps_direct = f_n.compute_epsilon(delta, delta_error=delta_error, epsilon_error=eps_error)
                assert eps_direct == pytest.approx(eps, abs=eps_error)

    def test_fpr_fnr_order(self):
        prv = PoissonSubsampledGaussianMechanism(1e-2, 0.7)
        acc = PRVAccountant(prvs=prv, eps_error=0.1, delta_error=1e-9, max_self_compositions=1_000)
        f_n = acc.compute_composition(1_000)
        epss, deltas = f_n.compute_epsilon_delta_estimates()
        fprs, fnrs = convert_epsilon_deltas_to_trade_off_curve(epss, deltas)
        for fnr, fpr in zip(fnrs, fprs):
            assert fnr + fpr <= 1.0
            assert 0.0 <= fnr <= 1.0
            assert 0.0 <= fpr <= 1.0
