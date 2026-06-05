import json
import os
import random
from argparse import ArgumentParser

random.seed(42)

import numpy as np
import yaml
from pydantic import BaseModel


class EstimatesPair(BaseModel):
    max: float | str
    min: float

    def model_post_init(
        self,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(self.max, str):
            self.max = eval(self.max)

class Parameters_(BaseModel):
    M: EstimatesPair
    J: EstimatesPair
    L_1: EstimatesPair
    L_2: EstimatesPair
    alpha_1: EstimatesPair
    alpha_2: EstimatesPair

class InitialConditions_(BaseModel):
    theta: EstimatesPair
    h: EstimatesPair

class ConfigModel(BaseModel):
    n_objects: int
    output_dir: str
    parameters: Parameters_
    initial_conditions: InitialConditions_


def load_config_from_yaml(file_path: str) -> ConfigModel:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return ConfigModel(**data)


def main(config_path: str):
    config = load_config_from_yaml(config_path)
    config_as_dict = config.model_dump()

    dump_dir = config.output_dir
    os.makedirs(
        config.output_dir,
        exist_ok=False,
    )

    del config_as_dict["n_objects"]
    del config_as_dict["output_dir"]

    for object_index in range(config.n_objects):
        object_as_dict = {}

        for outer_key in config_as_dict:
            object_as_dict[outer_key] = {}

            for inner_key, max_min_pairs in config_as_dict[outer_key].items():
                object_as_dict[outer_key][inner_key] = random.uniform(
                    max_min_pairs["min"],
                    max_min_pairs["max"],
                )

        object_as_dict["parameters"]["L_1"], object_as_dict["parameters"]["L_2"] = \
            sorted([object_as_dict["parameters"]["L_1"], object_as_dict["parameters"]["L_2"]])

        object_as_dict["initial_conditions"]["d_theta"] = 0
        object_as_dict["initial_conditions"]["d_h"] = 0

        with open(
            os.path.join(dump_dir, f"{object_index}.json"),
            "w",
        ) as f:
            json.dump(
                object_as_dict,
                f,
            )


if __name__ == "__main__":
    argument_parser = ArgumentParser()

    argument_parser.add_argument("--config")

    args = argument_parser.parse_args()

    main(args.config)
