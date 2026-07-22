import numpy as np


def make_h_trig_condition(
    h_trig: float,
    L_1: float,
    L_2: float,
    alpha_1: float,
    alpha_2: float
):
    def h_trig_condition(t, y) -> float:
        h_1 = y[2] + L_2 * np.sin(alpha_2 + y[0])
        h_2 = y[2] + L_1 * np.sin(np.pi - alpha_1 + y[0])

        return h_trig - min(h_1, h_2)

    h_trig_condition.terminal = True
    h_trig_condition.direction = -1

    return h_trig_condition

def make_mu_condition(epsilon: float = 1e-4):
    def mu_condition(t, y) -> float:

        return (y[0] - 0) - epsilon

    mu_condition.terminal = True
    mu_condition.direction = -1

    return mu_condition
