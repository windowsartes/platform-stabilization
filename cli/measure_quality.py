import logging
import os
import random
from argparse import ArgumentParser
from glob import glob

import numpy as np
import yaml
from pydantic import BaseModel
from tqdm import tqdm

seed = 42
random.seed(42)
np.random.seed(42)

from platform_stabilization.quality_measurement import get_mean_interval, get_wilcoxon_test_p_value


class ConfigModel(BaseModel):
    results_dump_dir: str
    log_file_path: str


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def main(config_path: str):
    config = load_config_from_yaml(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{config.log_file_path}",
        filemode="w"
    )

    for target_estimate_label in tqdm(os.listdir(config.results_dump_dir)):
        logging.info(f"processing results of {target_estimate_label}")

        current_estimate_values = os.listdir(os.path.join(config.results_dump_dir, target_estimate_label))
        current_estimate_values = [float(value) for value in current_estimate_values]
        current_estimate_values = sorted(
            current_estimate_values,
            key=lambda x: x if x > 1 else -x,
        )

        base_model_value = current_estimate_values[0]

        logging.info(f"version with {target_estimate_label}={base_model_value} was chosen as base model")

        base_model_convergence_time = []

        t_pathes = glob(
            os.path.join(
                config.results_dump_dir,
                target_estimate_label,
                str(base_model_value),
                "**",
                "t.npy",
            )
        )
        t_pathes = sorted(t_pathes, key=lambda x: int(x.split("/")[-2]))

        for t_path in t_pathes:
            base_model_convergence_time.append(np.load(t_path)[-1])

        base_model_convergence_time = np.array(base_model_convergence_time)

        current_estimate_values = current_estimate_values[1:]
        for concurrent_model_value in current_estimate_values:
            logging.info(f"now working with concurrent model with {target_estimate_label}={concurrent_model_value}")

            concurrent_model_convergence_time = []

            t_pathes = glob(
                os.path.join(
                    config.results_dump_dir,
                    target_estimate_label,
                    str(concurrent_model_value),
                    "**",
                    "t.npy",
                )
            )
            t_pathes = sorted(t_pathes, key=lambda x: int(x.split("/")[-2]))

            for t_path in t_pathes:
                concurrent_model_convergence_time.append(np.load(t_path)[-1])

            concurrent_model_convergence_time = np.array(concurrent_model_convergence_time)

            effects = base_model_convergence_time - concurrent_model_convergence_time
            current_lower_effect_bound, current_upper_effect_bound = get_mean_interval(effects)

            logging.info(f"mean effect bound: ({current_lower_effect_bound:.5f}; {current_upper_effect_bound:.5f})")

            p_value = get_wilcoxon_test_p_value(effects)

            logging.info(f"Wilcoxon test p-value: {p_value:.5f}")


if __name__ == "__main__":
    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
