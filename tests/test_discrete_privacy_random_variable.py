# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

from prv_accountant.discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism


def test_eps_deltas():
    prv = PoissonSubsampledGaussianMechanism(1e-2, 0.7)
    acc = PRVAccountant(prvs=prv, eps_error=0.1, delta_error=1e-9, max_self_compositions=1_000)
    f_n = acc.compute_composition(1_000)
    epss, deltas = f_n.compute_epsilon_delta_estimates()
    for eps, delta in zip(epss[::1000], deltas[::1000]):
        assert f_n.compute_delta_estimate(eps) == pytest.approx(delta)

def test_fpr_fnr_order():
    prv = PoissonSubsampledGaussianMechanism(1e-2, 0.7)
    acc = PRVAccountant(prvs=prv, eps_error=0.1, delta_error=1e-9, max_self_compositions=1_000)
    f_n = acc.compute_composition(1_000)
    fnrs, fprs = f_n.compute_f_estimates()
    for fnr, fpr in zip(fnrs, fprs):
        assert fnr + fpr <= 1.0
        assert 0.0 <= fnr <= 1.0
        assert 0.0 <= fpr <= 1.0
