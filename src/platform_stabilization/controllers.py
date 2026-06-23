from collections.abc import Callable

import numpy as np


class FirstStageController(Callable[[float, float], float]):
    def __init__(
        self,
        F_theta: float,
        F_0: float,
        theta_0: float,
        i: int,
        chi: Callable[[float], float],
    ):
        self._F_theta = F_theta
        self._F_0 = F_0

        self._theta_prev = theta_0
        self._theta_0 = theta_0

        self._t_prev = 0

        self._i = i

        self._chi = chi

    def __call__(
        self,
        theta_curr: float,
        t_curr: float,
    ) -> float:
        d_t = t_curr - self._t_prev
        self._t_prev = t_curr

        d_theta = theta_curr - self._theta_prev
        self._theta_prev = theta_curr

        theta_derivative = d_theta - d_t

        F = self._F_0 + ((-1) ** (self._i + 1)) * self._F_theta * \
            np.sign(theta_derivative + self._chi(theta_curr - self._theta_0))

        return F

class SecondStageController(Callable[[float, float], float]):
    def __init__(
        self,
        F_theta: float,
        F_0: float,
        mu: float,
        theta_0: float,
        i: int,
        chi: Callable[[float], float],
    ):
        self._F_theta = F_theta
        self._F_0 = F_0
        self._mu = mu

        self._theta_prev = theta_0

        self._t_prev = 0

        self._i = i

        self._chi = chi

    def __call__(
        self,
        theta_curr: float,
        t_curr: float,
    ) -> float:
        d_t = t_curr - self._t_prev
        self._t_prev = t_curr

        d_theta = theta_curr - self._theta_prev
        self._theta_prev = theta_curr

        theta_derivative = d_theta - d_t

        F = self._F_0 + ((-1) ** (self._i + 1)) * self._F_theta * \
            np.sign(theta_derivative + self._mu * self._chi(theta_curr))

        return F
