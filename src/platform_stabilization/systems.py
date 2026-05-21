from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp

from platform_stabilization.constants import g
from platform_stabilization.chi_functions import ChiFunction


@dataclass
class Parameters:
    M: float
    J: float
    L_1: float
    L_2: float
    alpha_1: float
    alpha_2: float


@dataclass
class InitialState:
    theta: float
    d_theta: float
    y: float
    d_y: float


class DefaultSystem:
    def __init__(
        self,
        M: float,
        J: float,
        L1: float,
        L2: float,
        alpha_1: float,
        alpha_2: float,
    ):
        self._M = M
        self._J = J
        self._L1 = L1
        self._L2 = L2
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2

    def simulate(
        self,
        F_0: float,
        F_theta: float,
        mu: float,
        chi_function: ChiFunction,
        y_0: np.ndarray,
        t_span: tuple[float, float],
        rtol: float = 1e-3,
        atol: float = 1e-4,
        method: str = "RK45",
    ) -> tuple[np.ndarray, np.ndarray]:
        def system(
            t: float,
            y: np.ndarray,
        ) -> np.ndarray:
            F_0_term = F_0 * (self._L2 * np.cos(y[0] + self._alpha_2) - self._L1 * np.cos(y[0] + self._alpha_1))
            F_theta_term = F_theta * np.sign(y[1] + mu * chi_function.chi(y[0])) * \
                (self._L2 * np.cos(y[0] + self._alpha_2) + self._L1 * np.cos(y[0] + self._alpha_1))

            return np.array(
                [
                    y[1],
                    (F_0_term - F_theta_term) / self._J,
                    y[3],
                    2 * F_0 / self._M - g,
                ]
            )

        solution = solve_ivp(
            fun=system,
            t_span=t_span,
            y0=y_0,
            rtol=rtol,
            atol=atol,
            method=method,
        )

        return (
            solution.t,
            solution.y,
        )
