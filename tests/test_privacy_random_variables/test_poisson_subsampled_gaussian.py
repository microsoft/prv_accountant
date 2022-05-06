# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import scipy
import math
import pytest
import numpy as np
import sys

from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
from prv_accountant.privacy_random_variables.poisson_subsampled_gaussian_mechanism import log


def test_safe_log():
    assert np.isnan(log(-1)) == True  # noqa: E712
    assert np.isneginf(log(0)) == True  # noqa: E712
    assert log(1) == pytest.approx(0)


def reference_sf(t, sigma, p):
    def alpha(t):
        return scipy.stats.norm.cdf(-t/(1/sigma) - (1/sigma)/2)

    def oneminus_beta(t):
        return scipy.stats.norm.sf(t/(1/sigma) - (1/sigma)/2)

    # survival function (i.e., 1-cdf) for stdQ
    if t > 0:
        return p*oneminus_beta(t+math.log(1/p-(1-p)*math.exp(-t)/p))+(1-p)*alpha(t+math.log(1/p-(1-p)*math.exp(-t)/p))  # noqa: E501
    elif t > math.log(1-p):
        return p*oneminus_beta(math.log((math.exp(t)-(1-p))/p))+(1-p)*alpha(math.log((math.exp(t)-(1-p))/p))  # noqa: E501
    else:
        return 1


class TestPoissonSubsampledGaussianMechanism:
    def test_sf(self):
        p = 1e-2
        sigma = 1.0
        Q = PoissonSubsampledGaussianMechanism(p, sigma)

        t = [
            math.log(1-p) - 1,
            math.log(1-p),
            math.log(1-p)/2,
            0.0,
            1.0,
            100.0
        ]

        for t_i in t:
            assert 1 - Q.cdf(t_i) == pytest.approx(reference_sf(
                t=t_i, p=p, sigma=sigma))

    def test_normalised(self):
        p = 1e-2
        sigma = 1.0
        Q = PoissonSubsampledGaussianMechanism(p, sigma)

        t = np.linspace(-10.0, 10.0, 2000000, dtype=np.longdouble)
        dt = t[1] - t[0]

        t_L = t - dt/2.0
        t_R = t + dt/2.0

        pdf = Q.probability(t_L, t_R)
        assert pdf.sum() == pytest.approx(1.0, 1e-10)

    def test_rdp(self):
        """
        Compare to TF-privacy
        """
        # Opting out of loading all sibling packages and their dependencies.
        try:
            sys.skip_tf_privacy_import = True
            from tensorflow_privacy.privacy.analysis import rdp_accountant  # noqa: E402
        except ImportError:
            pytest.skip("Tensorflow-privacy not available.")

        p = 1e-2
        sigma = 1.0
        orders = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        Q = PoissonSubsampledGaussianMechanism(p, sigma)

        rdp = [Q.rdp(a) for a in orders]
        rdp_tf = rdp_accountant.compute_rdp(p, sigma, 1, orders)

        np.testing.assert_array_almost_equal(rdp, rdp_tf)
