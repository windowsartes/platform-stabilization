from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp

from platform_stabilization.constants import g
from platform_stabilization.chi_functions import ChiFunction
from platform_stabilization.controllers import FirstStageController


class System:
    def __init__(
        self,
        M: float,
        J: float,
        L_1: float,
        L_2: float,
        alpha_1: float,
        alpha_2: float,
        **kwargs,
    ):
        self._M = M
        self._J = J
        self._L_1 = L_1
        self._L_2 = L_2
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2

    def simulate(
        self,
        controller_1: FirstStageController,
        controller_2: FirstStageController,
        y_0: np.ndarray,
        t_span: tuple[float, float],
        early_stopping_criterion: Callable[[float, np.ndarray], float],
        rtol: float = 1e-3,
        atol: float = 1e-4,
        method: str = "RK45",
    ) -> tuple[np.ndarray, np.ndarray]:
        def system(
            t: float,
            y: np.ndarray,
        ) -> np.ndarray:
            F_1 = controller_1(
                y[0],
                t,
            )

            F_2 = controller_2(
                y[0],
                t,
            )

            return np.array(
                [
                    y[1],
                    (F_2 * self._L_2 * np.cos(y[0] + self._alpha_2) - F_1 * self._L_1 * np.cos(y[0] + self._alpha_1)) / self._J,
                    y[3],
                    (F_1 + F_2) / self._M - g,
                ]
            )

        solution = solve_ivp(
            fun=system,
            t_span=t_span,
            y0=y_0,
            rtol=rtol,
            atol=atol,
            method=method,
            events=early_stopping_criterion,
        )

        return (
            solution.t,
            solution.y,
        )
