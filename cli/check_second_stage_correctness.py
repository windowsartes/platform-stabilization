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
    theta_epsilon_absolute: float
    w_overline: float


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
        y_values = np.load(results_path)
        t_values = np.load(
            os.path.join(
                "/".join(results_path.split("/")[:-1]),
                "t.npy",
            )
        )

        thetas = y_values[0]

        if np.abs(np.abs(thetas[-1]) - config.theta_epsilon_absolute) > 1e-8:
            logging.warning(
                f"theta test failed for the {results_path}: " + \
                f"{np.abs(np.abs(thetas[-1]) - config.theta_epsilon_absolute)} < 1e-8"
            )

        d_thetas = (thetas[1:] - thetas[:-1]) / (t_values[1:] - t_values[:-1])

        if not (np.abs(d_thetas) < config.w_overline).all():
            logging.warning(f"|d_theta| < w failed for the {results_path}")

        if not (d_thetas < 0).all():
            logging.warning(f"d_theta < 0 failed for the {results_path}")

        h = y_values[2]
        d_h = (h[1:] - h[:-1]) / (t_values[1:] - t_values[:-1])

        if not (d_h > 0).all():
            logging.warning(f"d_h > 0 failed for the {results_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="second_stage_test.log",
        filemode="w"
    )

    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
