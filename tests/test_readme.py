

def test_1():
    from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, GaussianMechanism, LaplaceMechanism

    prv_a = PoissonSubsampledGaussianMechanism(noise_multiplier=0.8, sampling_probability=5e-3)
    prv_b = GaussianMechanism(noise_multiplier=8.0)
    prv_c = LaplaceMechanism(mu=0.1)

    m = 100
    n = 200
    o = 100

    from prv_accountant.accountant import PRVAccountant
    accountant = PRVAccountant(
        prvs=[prv_a, prv_b, prv_c],
        max_self_compositions=[1_000, 1_000, 1_000],
        eps_error=0.1,
        delta_error=1e-10
    )

    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=1e-6, num_self_compositions=[m, n, o])
    print(eps_low, eps_est, eps_up)
