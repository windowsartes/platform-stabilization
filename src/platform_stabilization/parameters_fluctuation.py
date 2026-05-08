import random


def flactuate(
    config,
    params_to_fluctuate: list[str],
    value: float,
):
    for param in params_to_fluctuate:
        config["params"][param] = random.uniform(
            config["params"][param] - value,
            config["params"][param] + value,
        )

    return config
