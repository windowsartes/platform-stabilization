import json
import os
import random
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob

random.seed(42)

import numpy as np
import yaml
from pydantic import BaseModel, Field

from platform_stabilization.chi_functions import ChiFunctionFactory
from platform_stabilization.constants import g


class Coefficients_(BaseModel):
    M: float | list[float]
    J: float | list[float]
    L_underline: float | list[float]
    eta: float | list[float]
    alpha_max: float | list[float]
    theta_overline: float | list[float]
    iterate_over: str = Field(
        init=False,
        default="",
    )

    def model_post_init(
        self,
        __context,
    ) -> None:
        if isinstance(self.M, float):
            self.M = [self.M]
        else:
            self.iterate_over = "M"

            return

        if isinstance(self.J, float):
            self.J = [self.J]
        else:
            self.iterate_over = "J"

            return

        if isinstance(self.L_underline, float):
            self.L_underline = [self.L_underline]
        else:
            self.iterate_over = "L_underline"

            return

        if isinstance(self.eta, float):
            self.eta = [self.eta]
        else:
            self.iterate_over = "eta"

            return

        if isinstance(self.alpha_max, float):
            self.alpha_max = [self.alpha_max]
        else:
            self.iterate_over = "alpha_max"

            return

        if isinstance(self.theta_overline, float):
            self.theta_overline = [self.theta_overline]
        else:
            self.iterate_over = "theta_overline"

            return

class Absolute_(BaseModel):
    omega_overline: float
    chi_function: str
    F_overline: float

class Estimates(BaseModel):
    coefficients: Coefficients_
    absolute: Absolute_

class ConfigModel(BaseModel):
    objects_dir: str
    controllers_dir: str
    estimates: Estimates


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def main(config_path: str) -> None:
    config = load_config_from_yaml(config_path)

    controller_params = config.estimates

    saved_objects_dir = config.objects_dir
    saved_objects = glob(os.path.join(saved_objects_dir, "*.json"))

    iterable_parameter = controller_params.coefficients.iterate_over
    iterable_parameter_values = getattr(controller_params.coefficients, iterable_parameter)

    noniterable_parameter_dict = controller_params.coefficients.__dict__

    del noniterable_parameter_dict[iterable_parameter]
    del noniterable_parameter_dict["iterate_over"]

    chi_function = ChiFunctionFactory.construct(controller_params.absolute.chi_function)

    for current_iterable_parameter_value in iterable_parameter_values:
        current_setup_parameters_dict = deepcopy(noniterable_parameter_dict)
        current_setup_parameters_dict[iterable_parameter] = current_iterable_parameter_value

        label = f"{iterable_parameter}_{current_iterable_parameter_value}"

        current_saved_estimates_dir = os.path.join(
            config.controllers_dir,
            label,
            "estimates",
        )
        os.makedirs(
            current_saved_estimates_dir,
            exist_ok=False,
        )

        current_saved_parameters_dir = os.path.join(
            config.controllers_dir,
            label,
            "parameters",
        )
        os.makedirs(
            current_saved_parameters_dir,
            exist_ok=False,
        )

        for object_config_file_path in saved_objects:
            with open(
                object_config_file_path,
                "r",
            ) as f:
                object_config = json.load(f)

            object_id = os.path.splitext(object_config_file_path.split("/")[-1])[0]

            object_parameters = object_config["parameters"]
            object_initial_state = object_config["initial_conditions"]

            current_controller_estimates = {
                "M_overline": current_setup_parameters_dict["M"] * \
                    object_parameters["M"],
                "J_overline": current_setup_parameters_dict["J"] * \
                    object_parameters["J"],
                "L_underline": current_setup_parameters_dict["L_underline"] * \
                    (object_parameters["L_1"] + object_parameters["L_2"]),
                "eta": current_setup_parameters_dict["eta"] * \
                    (object_parameters["L_1"] / object_parameters["L_2"]),
                "alpha_max": current_setup_parameters_dict["alpha_max"] * \
                    max(object_parameters["alpha_1"], object_parameters["alpha_2"]),
                "theta_overline": current_setup_parameters_dict["theta_overline"] * \
                    object_initial_state["theta"],
            }

            with open(
                os.path.join(
                    current_saved_estimates_dir,
                    f"{object_id}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    current_controller_estimates,
                    f,
                )

            cos_therm = np.cos(current_controller_estimates["theta_overline"] + current_controller_estimates["alpha_max"])
            theta_big = cos_therm / \
                np.cos(current_controller_estimates["theta_overline"] - current_controller_estimates["alpha_max"])

            F_0_lower = current_controller_estimates["M_overline"] * g / 2
            F_0_upper = controller_params.absolute.F_overline * \
                (1 + current_controller_estimates["eta"] * theta_big) / 2
            F_0 = random.uniform(F_0_lower, F_0_upper)

            F_theta_lower = F_0 * ((1 - current_controller_estimates["eta"] * theta_big) / \
                (1 + current_controller_estimates["eta"] * theta_big))
            F_theta_upper = min(controller_params.absolute.F_overline - F_0, F_0)
            F_theta = random.uniform(F_theta_lower, F_theta_upper)

            first_multiplier = F_theta - F_0 * ((1 - current_controller_estimates["eta"] * theta_big) / \
                (1 + current_controller_estimates["eta"] * theta_big))
            second_multiplier = (current_controller_estimates["L_underline"]  * cos_therm) / \
                (current_controller_estimates["J_overline"] * chi_function.chi_upper * chi_function.chi_dot_upper)
            
            mu_upper = min(
                controller_params.absolute.omega_overline / chi_function.chi_upper,
                np.sqrt(first_multiplier * second_multiplier),
            )
            mu_lower = mu_upper * 0.9
            mu = random.uniform(mu_lower, mu_upper)

            current_controller_parameters = {
                "F_0": F_0,
                "F_theta": F_theta,
                "mu": mu,
            }

            with open(
                os.path.join(
                    current_saved_parameters_dir,
                    f"{object_id}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    current_controller_parameters,
                    f,
                )


if __name__ == "__main__":
    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
