from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ChiFunction:
    chi: Callable[[float], float]
    chi_upper: float
    chi_dot_upper: float


class ChiFunctionFactory:
    @staticmethod
    def construct(function_type: str) -> ChiFunction:
        if function_type == "tanh":
            return ChiFunction(
                np.tanh,
                1,
                1,
            )
        else:
            raise ValueError("for now only 'tanh' chi-function is supported")
