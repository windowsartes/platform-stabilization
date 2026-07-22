import json
import logging
import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import yaml
from Ball import bcov_test
from pydantic import BaseModel
from tqdm import tqdm


class ConfigModel(BaseModel):
    controllers_dir: str
    results_dir: str
    parameter_estimate_pairs: list[tuple[str, str]]
    num_permutations: int


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def main(config_path: str) -> None:
    config = load_config_from_yaml(config_path)

    for chosen_parameter, assotiated_estimate in config.parameter_estimate_pairs:
        logging.info(f"working with {chosen_parameter}-{assotiated_estimate} pair")

        convergence_times = []
        estimate_values = []

        for coef in os.listdir(
            os.path.join(
                config.results_dir,
                chosen_parameter,
            )
        ):
            for episode_id_path in glob(
                os.path.join(
                    config.results_dir,
                    chosen_parameter,
                    coef,
                    "**",
                )
            ):
                convergence_times.append(np.load(os.path.join(episode_id_path, "t.npy"))[-1])

                episode_id = episode_id_path.split("/")[-1]

                assotiated_controller_estimates_path = os.path.join(
                    config.controllers_dir,
                    f"{chosen_parameter}_{coef}",
                    "estimates",
                    f"{episode_id}.json",
                )

                with open(
                    assotiated_controller_estimates_path,
                    "r",
                ) as f:
                    assotiated_controller_estimate = json.load(f)[assotiated_estimate]

                estimate_values.append(assotiated_controller_estimate)

        current_pair_result = bcov_test(
            estimate_values, 
            convergence_times,
            num_permutations=config.num_permutations,
        )

        logging.info(
            f"for the {chosen_parameter}: statistics={current_pair_result.statistic}, " + \
            f"p-value={current_pair_result.pvalue}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="ball_correlation.log",
        filemode="w"
    )

    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
