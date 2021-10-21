# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np
import math
from scipy import integrate
from numpy import exp, sqrt
from numpy import power as pow
from scipy import special
from scipy.special import erfc

M_SQRT2 = sqrt(np.longdouble(2))
M_PI = np.pi


def log(x):
    valid = (x > 0)
    x_is_0 = (x == 0)
    return np.where(valid, np.log(np.where(valid, x, 1)), 
        np.where(x_is_0, -np.inf, np.nan))


class PrivacyRandomVariable(ABC):
    @abstractmethod
    def mean(self) -> float:
        pass

    def probability(self, a, b):
        return self.cdf(b) - self.cdf(a)

    @abstractmethod
    def pdf(self, t):
        pass

    @abstractmethod
    def cdf(self, t):
        pass

    def rdp(self, alpha: float) -> float:
        """
        Compute RDP of this mechanism of order alpha
        """
        raise NotImplementedError()

        
class PrivacyRandomVariableTruncated:
    def __init__(self, prv, t_min: float, t_max: float) -> None:
        self.prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self.remaining_mass = self.prv.cdf(t_max) - self.prv.cdf(t_min)

    def mean(self) -> float:
        points = [self.t_min, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, self.t_max]
        m = 0.0
        for L, R in zip(points[:-1], points[1:]):
            I, err = integrate.quad(self.cdf, L, R)

            m += (
                R*self.cdf(R) -
                L*self.cdf(L) -
                I
            )

        return m

    def probability(self, a, b):
        a = np.clip(a, self.t_min, self.t_max)
        b = np.clip(b, self.t_min, self.t_max)
        return self.prv.probability(a, b) / self.remaining_mass

    def pdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.pdf(t)/self.remaining_mass, 0))

    def cdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.cdf(t)/self.remaining_mass, 1))


class PoissonSubsampledGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, noise_multiplier: float) -> None:
        self.p = np.longdouble(sampling_probability)
        self.sigma = np.longdouble(noise_multiplier)

    def pdf(self, t):
        sigma = self.sigma
        p = self.p
        return np.where(t > 0, (
            (1.0/2.0) * M_SQRT2 * sigma *
            exp((
                -1.0/2.0*pow(sigma, 2)*pow(t, 2) - pow(sigma, 2)*t*log((p + exp(t) - 1)*exp(-t)/p) -
                1.0/2.0*pow(sigma, 2)*pow(log((p + exp(t) - 1)*exp(-t)/p), 2) + (3.0/2.0)*t - 1.0/8.0/pow(sigma, 2)
            )) /
            (sqrt(M_PI)*sqrt((p + exp(t) - 1)*exp(-t)/p)*(p + exp(t) - 1))
        ), np.where(t > log(1 - p), (
                (1.0/2.0) * M_SQRT2 * sigma *
                exp(-1.0/2.0*pow(sigma, 2)*pow(log((p + exp(t) - 1)/p), 2) + 2*t - 1.0/8.0/pow(sigma, 2)) /
                (sqrt(M_PI)*sqrt((p + exp(t) - 1)/p)*(p + exp(t) - 1))
            ), 0)
        )

    def cdf(self, t):
        sigma = self.sigma
        p = self.p
        z = np.where(t>0, log((p-1)/p + exp(t)/p), log((p-1)/p + exp(t)/p))
        return np.where(t > log(1 - p), (
                (1.0/2.0) * p * (-erfc(np.double((1.0/4.0)*M_SQRT2*(2*pow(sigma, 2)*z - 1)/sigma))) -
                1.0/2.0*(p - 1) * (-erfc(np.double((1.0/4.0)*M_SQRT2*(2*pow(sigma, 2)*z + 1)/sigma))) + 1
            ), 0.0)

    def mean(self):
        raise NotImplementedError("Mean computation not implemented")

    def rdp(self, alpha: float) -> float:
        """
        Compute RDP of this mechanism of order alpha
        
        Based on Google's TF Privacy: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py 
        """
        if self.p == 0:
            return 0

        if self.p == 1.:
            return alpha / (2 * self.sigma**2)

        if np.isinf(alpha):
            return np.inf

        return _compute_log_a(np.double(self.p), np.double(self.sigma), alpha) / (alpha - 1)


# The following code is based on Google's TF Privacy
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py 

def _compute_log_a(q, sigma, alpha):
    """Compute log(A_alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    else:
        return _compute_log_a_frac(q, sigma, alpha)


def _log_comb(n, k):
    return (special.gammaln(n + 1) - special.gammaln(k + 1) -
            special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
    """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q)

        s = log_coef_i + (i * i - i) / (2 * (sigma**2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _log_add(logx, logy):
    """Add two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _compute_log_a_frac(q, sigma, alpha):
    """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma**2 * math.log(1 / q - 1) + .5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _log_erfc(x):
    """Compute log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + special.log_ndtr(-x * 2**.5)
    except NameError:
        # If log_ndtr is not available, approximate as follows:
        r = special.erfc(x)
        if r == 0.0:
            # Using the Laurent series at infinity for the tail of the erfc function:
            #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
            # To verify in Mathematica:
            #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
            return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
                    .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
        else:
            return math.log(r)


def _log_sub(logx, logy):
    """Subtract two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx
