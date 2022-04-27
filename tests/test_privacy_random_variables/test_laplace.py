# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

from scipy.integrate import quad

from prv_accountant import LaplaceMechanism, PRVAccountant


class TestLaplaceMechanism:
    @pytest.mark.parametrize("mu", [0.2, 0.5, 1.0])
    def test_normalisation(self, mu: float):
        r"""
        \E[exp(-Y)] = \int_R exp(-t) PDF_Y(t) dt == pytest.approx(1)  ## noqa: W605
        """
        prv = LaplaceMechanism(mu)
        e, _ = quad(lambda t: np.exp(-t)*prv.cdf(t), -50, 50, limit=500)
        assert e == pytest.approx(1)

    @pytest.mark.parametrize("mu", [1e-2])
    @pytest.mark.parametrize("num_comp", [10_000, 100_000])
    @pytest.mark.parametrize("delta", [1e-5, 1e-6])
    def test_advanced_composition(self, mu, num_comp, delta):
        """Check that the PRV obeys advanced composition"""
        prv = LaplaceMechanism(mu)
        eps_error = 0.1
        accountant = PRVAccountant(prvs=[prv], eps_error=eps_error, delta_error=delta*1e-3, max_self_compositions=[num_comp])
        _, _, eps_upper = accountant.compute_epsilon(delta, [num_comp])
        assert eps_upper < mu*num_comp * np.sqrt(2*np.log(1/delta)) + eps_error
