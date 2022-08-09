# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np

from prv_accountant import ApproximateDPMechanism, PRVAccountant


class TestApproximateDP:
    @pytest.mark.parametrize("epsilon", [0.01])
    @pytest.mark.parametrize("delta", [1e-7, 1e-8])
    @pytest.mark.parametrize("delta_prime", [1e-5, 1e-6])
    @pytest.mark.parametrize("num_comp", [10_000, 100_000])
    def test_advanced_composition(self, epsilon, delta, delta_prime, num_comp):
        """Check that the PRV obeys strong composition"""
        prv = ApproximateDPMechanism(epsilon=epsilon, delta=delta)
        eps_error = 0.1
        delta_error = delta/1000
        accountant = PRVAccountant(prvs=[prv], eps_error=eps_error, delta_error=delta_error, max_self_compositions=[num_comp])

        delta_tilde = num_comp*delta + delta_prime
        eps_tilde_lower, _, _ = accountant.compute_epsilon(delta_tilde, [num_comp])

        eps_tilde_sc = (
            epsilon*np.sqrt(2*num_comp*np.log(1/delta_prime)) +
            num_comp*num_comp*epsilon*(np.exp(epsilon)-1)/(np.exp(epsilon)+1)
        )

        assert eps_tilde_lower < eps_tilde_sc

    @pytest.mark.parametrize("delta_1", [1e-7, 1e-8])
    @pytest.mark.parametrize("num_comp", [10_000, 100_000])
    def test_zero_delta(self, delta_1, num_comp):
        prv = ApproximateDPMechanism(epsilon=0.0, delta=delta_1)
        eps_error = 0.1
        delta_error = delta_1/100
        accountant = PRVAccountant(prvs=[prv], eps_error=eps_error, delta_error=delta_error, max_self_compositions=[num_comp])

        _, delta, _ = accountant.compute_delta(epsilon=0.0, num_self_compositions=[num_comp])

        assert delta == pytest.approx(1 - (1-delta_1)**num_comp)
