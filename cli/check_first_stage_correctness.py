import json
import logging
import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import yaml
from pydantic import BaseModel
from tqdm import tqdm


class ConfigModel(BaseModel):
    results_dir: str
    controllers_dir: str
    objects_dir: str
    theta_epsilon_absolute: float


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def main(config_path: str) -> None:
    config = load_config_from_yaml(config_path)

    obtained_results = glob(
        os.path.join(
            config.results_dir,
            "**",
            "**",
            "**",
            "y.npy",
        )
    )

    logging.info(f"{len(obtained_results)} results were found")

    for results_path in tqdm(obtained_results):
        path_parts = results_path.split("/")

        label = "_".join(path_parts[-4:-2])
        id = path_parts[-2]

        assotiated_controller_path = os.path.join(
            config.controllers_dir,
            label,
            "parameters",
            f"{id}.json",
        )

        with open(
            assotiated_controller_path,
            "r",
        ) as f:
            target_h_trig = json.load(f)["h_trig"]

        y_values = np.load(results_path)

        h = y_values[2]
        thetas = y_values[0]

        assotiated_object_path = os.path.join(
            config.objects_dir,
            f"{id}.json",
        )

        with open(
            assotiated_object_path,
            "r",
        ) as f:
            object = json.load(f)

        object_parameters = object["parameters"]
        theta_0 = object["initial_conditions"]["theta"]

        L_1 = object_parameters["L_1"]
        L_2 = object_parameters["L_2"]
        alpha_1 = object_parameters["alpha_1"]
        alpha_2 = object_parameters["alpha_2"]

        h_1 = h[-1] + L_2 * np.sin(alpha_2 + thetas[-1])
        h_2 = h[-1] + L_1 * np.sin(np.pi - alpha_1 + thetas[-1])

        if min(h_1, h_2) + 1e-8 < target_h_trig:
            logging.warning(f"h test failed for the {results_path}")

        thetas = y_values[0]

        theta_diff_absolute = max(thetas.max() - theta_0, theta_0 - thetas.min())

        if theta_diff_absolute > config.theta_epsilon_absolute:
            logging.warning(
                f"theta test failed for the {results_path}: " + \
                f"{theta_diff_absolute} < {config.theta_epsilon_absolute}"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="first_stage_test.log",
        filemode="w"
    )

    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
