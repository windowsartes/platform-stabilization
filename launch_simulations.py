import argparse
import json
import logging
import os
import random
from copy import deepcopy

import numpy as np
import yaml
from joblib import delayed, Parallel
from tqdm import tqdm

from platform_stabilization.chi_functions import ChiFunction, ChiFunctionFactory
from platform_stabilization.constants import g
# from platform_stabilization.parameters_fluctuation import flactuate
from platform_stabilization.systems import DefaultSystem


random.seed(1234)


def setup_worker_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PID %(process)d - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def simulate(
    system: DefaultSystem,
    control_params: dict[str, float],
    initial_state: tuple[float, float, float, float],
    iteration_index: int,
    output_dir: str,
    chi_function: ChiFunction,
    t_span: tuple[float, float],
    rtol: float = 1e-4,
    atol: float = 1e-4,
    method: float = "RK23",
) -> None:
    setup_worker_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"system #{iteration_index}: solve_ivp was called")

    t, y = system.simulate(
        F_0=control_params["F_0"],
        F_theta=control_params["F_theta"],
        mu=control_params["mu"],
        chi_function=chi_function,
        y_0=initial_state,
        t_span=t_span,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    output_dir = os.path.join(output_dir, str(iteration_index))

    np.save(
        os.path.join(output_dir, "t.npy"),
        t,
    )
    np.save(
        os.path.join(output_dir, "y.npy"),
        y,
    )

    logger.info(f"system #{iteration_index}: results were stored at the {output_dir}")



def main(
    config_path: str,
    storage_dir: str,
):
    if os.path.isdir(storage_dir):
        n_subdirs = len(os.listdir(storage_dir))
        storage_dir = os.path.join(storage_dir, str(n_subdirs))
    else:
        storage_dir = os.path.join(storage_dir, "0")

    os.makedirs(
        storage_dir,
        exist_ok=False,
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["estimates"]["alpha_max"] = eval(config["estimates"]["alpha_max"])
    config["estimates"]["theta_max"] = eval(config["estimates"]["theta_max"])

    chi_function = ChiFunctionFactory.construct(config["chi_function"])

    param_to_vary = "L2"

    param_values = config["params"][param_to_vary]

    with Parallel(n_jobs=-1) as parallel:
        for value in param_values:
            config["params"][param_to_vary] = value

            if os.path.isdir(storage_dir):
                n_subdirs = len(os.listdir(storage_dir))
                current_storage_dir = os.path.join(storage_dir, str(n_subdirs))
            else:
                current_storage_dir = os.path.join(storage_dir, "0")

            logging.info(f"current storage dir is {current_storage_dir}")

            os.makedirs(
                current_storage_dir,
                exist_ok=False,
            )

            for iteration_index in range(config["n_trials"]):
                os.makedirs(
                    os.path.join(current_storage_dir, str(iteration_index)),
                    exist_ok=False,
                )

            systems = []
            for iteration_index in range(config["n_trials"]):
                config_copy = deepcopy(config)

                systems.append(
                    DefaultSystem(
                        **config_copy["params"]
                    )
                )

                with open(os.path.join(current_storage_dir, str(iteration_index), "used_config.json"), "w") as f:
                    json.dump(
                        config_copy,
                        f,
                    )

            logging.info("systems were created and stored")

            estimates = config["estimates"]

            for param in ["alpha_max", "theta_max"]:
                estimates[param] = eval(estimates[param]) if isinstance(estimates[param], str) else estimates[param]

            theta_big = np.cos(estimates["theta_max"] + estimates["alpha_max"]) / \
                np.cos(estimates["theta_max"] - estimates["alpha_max"])
            cos_therm = np.cos(estimates["theta_max"] + estimates["alpha_max"])

            control_params_lst = []
            for iteration_index in range(config["n_trials"]):
                F_0_lower = estimates["M_upper"] * g / 2
                F_0_upper = estimates["F_upper"] * (1 + estimates["eta"] * theta_big) / 2
                F_0 = random.uniform(F_0_lower, F_0_upper)

                F_theta_lower = F_0 * ((1 - estimates["eta"] * theta_big) / (1 + estimates["eta"] * theta_big))
                F_theta_upper = min(estimates["F_upper"] - F_0, F_0)
                F_theta = random.uniform(F_theta_lower, F_theta_upper)

                mu_upper = min(
                    estimates["omega_upper"] / chi_function.chi_upper,
                    np.sqrt(
                        (F_theta - F_0 * ((1 - estimates["eta"] * theta_big) / (1 + estimates["eta"] * theta_big))) * \
                        ((estimates["L_lower"]  * cos_therm) / (estimates["J_upper"]  * chi_function.chi_upper * chi_function.chi_dot_upper))
                    ),
                )
                mu_lower = mu_upper * 0.9
                mu = random.uniform(mu_lower, mu_upper)

                current_params = {
                    "F_0": F_0,
                    "F_theta": F_theta,
                    "mu": mu,
                }

                control_params_lst.append(current_params)

                with open(os.path.join(current_storage_dir, str(iteration_index), "control_params.json"), "w") as f:
                    json.dump(
                        current_params,
                        f,
                    )

            logging.info("controllers were created and stored")

            initial_states = []
            for iteration_index in range(config["n_trials"]):
                current_initial_state = (
                    random.uniform(0, estimates["theta_max"] * 0.99),
                    0,
                    1,
                    0,
                )

                initial_states.append(current_initial_state)

                with open(os.path.join(current_storage_dir, str(iteration_index), "initial_state.json"), "w") as f:
                    json.dump(
                        current_initial_state,
                        f,
                    )

            logging.info("initial states were created and stored")

            t_span = (0, config["t_max"])

            parallel(
                delayed(simulate)(
                    system,
                    control_params,
                    initial_state,
                    iteration_index,
                    current_storage_dir,
                    chi_function,
                    t_span,
                    method="RK45",
                    atol=1e-4,
                    rtol=1e-4,
                ) for iteration_index, (system, control_params, initial_state) in tqdm(
                    enumerate(zip(systems, control_params_lst, initial_states)),
                    total=config["n_trials"],
                )
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    main(
        args.config,
        args.output_dir,
    )
