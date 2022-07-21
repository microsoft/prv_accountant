# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from prv_accountant import PRVAccountant, ApproximateDPMechanism, PoissonSubsampledGaussianMechanism


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
