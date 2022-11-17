import pytest

from prv_accountant.dpsgd import find_noise_multiplier, DPSGDAccountant


class TestFindNoiseMultiplier:
    def test_sensible_range(self):
        mu = find_noise_multiplier(sampling_probability=2e-3, num_steps=10_000, target_epsilon=4.0, target_delta=1e-7)
        assert 0 < mu and mu < 2  # Check that mu is in a sensible interval

    def test_inverse(self):
        max_steps = 10_000
        target_epsilon = 4.0
        eps_error = 0.5
        target_delta = 1e-7
        sampling_probability = 2e-3
        mu = find_noise_multiplier(sampling_probability, max_steps, target_epsilon, target_delta)
        acc = DPSGDAccountant(mu, sampling_probability, max_steps=max_steps, eps_error=eps_error)
        eps = acc.compute_epsilon(delta=target_delta, num_steps=max_steps)
        assert eps[2] == pytest.approx(target_epsilon, abs=eps_error)

    def test_robustness(self):
        with pytest.warns(None) as record:
            find_noise_multiplier(
                sampling_probability=256/50_000,
                num_steps=int(50*50_000/256),
                target_epsilon=10.0,
                target_delta=1e-5
            )
        assert len(record) == 0

    def test_robustness_2(self):
        try:
            find_noise_multiplier(
                sampling_probability=0.26058631921824105,
                num_steps=18800,
                target_delta=0.00011448277499759097,
                target_epsilon=4.0
            )
        except Exception:
            # Just test that this doesn't cause a floating point overflow
            assert False
