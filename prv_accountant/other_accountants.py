# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from typing import Iterable, Tuple
from prv_accountant.privacy_random_variables import PrivacyRandomVariable


class RDP:
    def __init__(self, prv: PrivacyRandomVariable, delta: float, orders: Iterable[float] = None) -> None:
        """
        Create a Renyi Differential Privacy accountant

        :param PrivacyRandomVariable prv:
        :param float delta:
        :param Iterable[float] orders:
        """
        self.delta = delta

        if not orders:
            orders = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.orders = np.array(orders)

        self.rdp = np.array([prv.rdp(a) for a in self.orders])

    def compute_epsilon(self, num_compositions: int) -> Tuple[float, float, float]:
        """
        Compute bounds on epsilon.

        This function is based on Google's TF Privacy:
        https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py 
        """
        rdp_steps = self.rdp*num_compositions
        orders_vec = np.atleast_1d(self.orders)
        rdp_vec = np.atleast_1d(rdp_steps)

        if len(orders_vec) != len(rdp_vec):
            raise ValueError("Input lists must have the same length.")

        eps = rdp_vec - np.log(self.delta) / (orders_vec - 1)

        idx_opt = np.nanargmin(eps)  # Ignore NaNs
        eps_opt = eps[idx_opt]
        return 0.0, eps_opt, eps_opt
