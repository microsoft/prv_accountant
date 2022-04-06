# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from prv_accountant.privacy_random_variables import PrivacyRandomVariableTruncated, PoissonSubsampledGaussianMechanism


class TestPrivacyRandomVariableTruncated:
    def test_robustness(self):
        p = 0.00011878424327013022
        mu = 0.2

        prv = PrivacyRandomVariableTruncated(
            prv=PoissonSubsampledGaussianMechanism(
                sampling_probability=p, noise_multiplier=mu
            ),
            t_min=-134.4230546183631,
            t_max=134.42325451405242
        )

        assert prv.mean() > 1e-4
