import pytest

from prv_accountant import Accountant


def test_accountant_warning():
    with pytest.deprecated_call():
        accountant = Accountant(
            noise_multiplier=0.8,
            sampling_probability=5e-3,
            delta=1e-6,
            eps_error=0.1,
            max_compositions=1000
        )
        accountant.compute_epsilon(num_compositions=10)
