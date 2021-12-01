# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from typing import Iterable, Tuple, Sequence
from prv_accountant.privacy_random_variables import PrivacyRandomVariable


class RDP:
    def __init__(self, prvs: Sequence[PrivacyRandomVariable], orders: Iterable[float] = None) -> None:
        """
        Create a Renyi Differential Privacy accountant

        :param PrivacyRandomVariable prv:
        :param float delta:
        :param Iterable[float] orders:
        """
        if not orders:
            orders = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.orders = np.array(orders)

        self.rdps = [np.array([prv.rdp(a) for a in self.orders]) for prv in prvs]

    def compute_epsilon(self, delta: float, num_self_compositions: Sequence[int]) -> Tuple[float, float, float]:
        """
        Compute bounds on epsilon.

        This function is based on Google's TF Privacy:
        https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py
        """
        if len(num_self_compositions) != len(self.rdps):
            raise ValueError("Length of `num_self_compositions` need to match number of PRVs passed to the RDP accountant.")

        rdp_steps = sum(rdp*n for rdp, n in zip(self.rdps, num_self_compositions))
        orders_vec = np.atleast_1d(self.orders)
        rdp_vec = np.atleast_1d(rdp_steps)

        if len(orders_vec) != len(rdp_vec):
            raise ValueError("Input lists must have the same length.")

        eps = rdp_vec - np.log(delta * orders_vec) / (orders_vec - 1) + np.log1p(- 1 / orders_vec)

        idx_opt = np.nanargmin(eps)  # Ignore NaNs
        eps_opt = eps[idx_opt]
        return 0.0, eps_opt, eps_opt
