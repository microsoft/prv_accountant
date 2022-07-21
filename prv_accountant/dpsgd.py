import numpy as np
from scipy import optimize
from typing import Tuple

from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism


class DPSGDAccountant(PRVAccountant):
    """
    Accountant for DP-SGD or similar algorithms such as DP-Adam
    """
    def __init__(self, noise_multiplier: float, sampling_probability: float,
                 max_steps: int, eps_error: float = 0.1, delta_error: float = 1e-9) -> None:
        """
        Accountant for DP-SGD

        :param float noise_multiplier: The noise multiplier of DP-SGD.
        :param float sampling_probability: The sampling probability.
        :param int max_steps: The maximum number of DP-SGD steps.
        :param eps_error: Maximum error allowed in $\varepsilon$. Typically around 0.1
        :param delta_error: Maximum error allowed in $\\delta$. typically around $10^{-3} \times \\delta$
        """
        super().__init__(prvs=PoissonSubsampledGaussianMechanism(noise_multiplier=noise_multiplier,
                                                                 sampling_probability=sampling_probability),
                         max_self_compositions=max_steps, eps_error=eps_error, delta_error=delta_error)

    def compute_epsilon(self, delta: float, num_steps: int) -> Tuple[float, float, float]:
        """
        Compute epsilon bounds for a given delta

        :param float delta: Target delta
        :param int num_steps: Number of DP-SGD steps.
        :return: Return lower bound for $\varepsilon$, estimate for $\varepsilon$ and upper bound for $\varepsilon$
        :rtype: Tuple[float,float,float]
        """
        return super().compute_epsilon(delta=delta, num_self_compositions=num_steps)


def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float = 0.1, mu_max: float = 100.0) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    :param float mu_max: Maximum value of noise multiplier of the search.
    """
    def compute_epsilon(mu: float) -> float:
        acc = DPSGDAccountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            max_steps=num_steps,
            eps_error=eps_error/2,
            delta_error=target_delta/1000
        )
        return acc.compute_epsilon(delta=target_delta, num_steps=num_steps)

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error
    return bracket[1]
