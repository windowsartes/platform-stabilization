import json
import os
from argparse import ArgumentParser

import numpy as np
import yaml
from joblib import delayed, Parallel
from pydantic import BaseModel
from tqdm import tqdm

from platform_stabilization.chi_functions import ChiFunctionFactory
from platform_stabilization.controllers import FirstStageController
from platform_stabilization.early_stopping_criterions import make_h_trig_condition
from platform_stabilization.systems import System


class ConfigModel(BaseModel):
    saved_objects_dir: str
    saved_controllers_dir: str
    results_dump_dir: str
    chosen_controllers_subset: list[str] | set[str]
    t_max: int

    def model_post_init(
        self,
        __context,
    ) -> None:
        self.chosen_controllers_subset = set(self.chosen_controllers_subset)


def controller_label(subdir: str) -> str:
    return "_".join(subdir.split("_")[:-1])


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def process_single_item(
    controller_parameters_path: str,
    object_path: str,
    t_max: int,
    results_dump_dir: str,
) -> None:
    controller_label = controller_parameters_path.split("/")[-3]
    controller_label = controller_label.split("_")

    parameter_label = "_".join(controller_label[:-1])
    parameter_value = controller_label[-1]

    controller_id = controller_parameters_path.split("/")[-1]
    controller_id = os.path.splitext(controller_id)[0]

    with open(controller_parameters_path, "r") as f:
        controller_parameters = json.load(f)

    with open(object_path, "r") as f:
        object = json.load(f)

    system = System(**object["parameters"])

    y_0 = np.array(
        [
            object["initial_conditions"]["theta"],
            object["initial_conditions"]["d_theta"],
            object["initial_conditions"]["h"],
            object["initial_conditions"]["d_h"],
        ]
    )

    chi = ChiFunctionFactory.construct("tanh").chi

    controller_1 = FirstStageController(
        F_theta=controller_parameters["F_theta"],
        F_0=controller_parameters["F_0"],
        theta_0=object["initial_conditions"]["theta"],
        i=1,
        chi=chi,
    )

    controller_2 = FirstStageController(
        F_theta=controller_parameters["F_theta"],
        F_0=controller_parameters["F_0"],
        theta_0=object["initial_conditions"]["theta"],
        i=2,
        chi=chi,
    )

    t_span = [0, t_max]

    h_trig_criterion = make_h_trig_condition( 
        h_trig=controller_parameters["h_trig"],
        L_1=object["parameters"]["L_1"],
        L_2=object["parameters"]["L_2"],
        alpha_1=object["parameters"]["alpha_1"],
        alpha_2=object["parameters"]["alpha_2"],
    )

    t, y = system.simulate(
        controller_1,
        controller_2,
        y_0,
        t_span,
        h_trig_criterion,
        method="RK45w",
    )

    output_dir = os.path.join(
        results_dump_dir,
        parameter_label,
        parameter_value,
        controller_id,
    )
    os.makedirs(
        output_dir,
        exist_ok=False,
    )

    np.save(
        os.path.join(output_dir, "t.npy"),
        t,
    )
    np.save(
        os.path.join(output_dir, "y.npy"),
        y,
    )


def main(config_path: str):
    config = load_config_from_yaml(config_path)

    controller_subdirs = os.listdir(config.saved_controllers_dir)
    controller_subdirs = [
        subdir for subdir in controller_subdirs if \
        controller_label(subdir) in config.chosen_controllers_subset
    ]

    controller_object_pairs = []

    for controller_subdir in controller_subdirs:
        current_controller_dir = os.path.join(
            config.saved_controllers_dir,
            controller_subdir,
            "parameters",
        )

        for controller_json_file_path in os.listdir(current_controller_dir):
            controller_full_path = os.path.join(
                current_controller_dir,
                controller_json_file_path,
            )

            object_full_path = os.path.join(
                config.saved_objects_dir,
                controller_json_file_path,
            )
            assert os.path.isfile(object_full_path), \
                f"object '{object_full_path}' was not found"
            
            controller_object_pairs.append(
                (
                   controller_full_path,
                   object_full_path,
                )
            )

    with Parallel(n_jobs=-1) as parallel:
        parallel(
            delayed(process_single_item)(
                controller_path,
                object_path,
                config.t_max,
                config.results_dump_dir,
            ) for (controller_path, object_path) in tqdm(controller_object_pairs)
        )


if __name__ == "__main__":
    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
