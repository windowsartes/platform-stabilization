import json
import os
from glob import glob

import numpy as np
from tqdm.auto import tqdm


def check_conditions_1(
    theta_values,
    threshold: float,
) -> bool:
    return theta_values[-1] < threshold

def check_conditions_2(
    theta_values,
    theta_upper: float,
) -> bool:
    return np.max(np.abs(theta_values)) < theta_upper

def check_conditions_3(
    theta_dot_values,
    omega_upper: float,
) -> bool:
    return np.max(np.abs(theta_dot_values)) < omega_upper

def check_conditions_4(
    c_dot_values,
) -> bool:
    return np.min(c_dot_values[1:]) > 0


def construct_bootstrap_interval(
    values,
    n_bootstrap_samples: int,
    condidence_level: float,
):
    estimators = []

    for _ in range(n_bootstrap_samples):
        current_bootstrap_sample = np.random.choice(
            values,
            size=len(values),
            replace=True,
        )

        estimators.append(np.mean(current_bootstrap_sample))

    left_border, right_border = np.quantile(
        estimators,
        q=[(1 - condidence_level) / 2, (1 + condidence_level) / 2]
    )
    
    return (
        round(float(left_border), 2),
        round(float(right_border), 2)
    )


if __name__ == "__main__":
    data_dir = "/home/jovyan/people/makarov/platform-stabilization/experiments/5"

    threshold = 0.05
    confidence_level = 0.95

    for setup_dir in glob(os.path.join(data_dir, "**")):
        print(setup_dir)

        with open(os.path.join(setup_dir, "0", "used_config.json")) as f:
            config = json.load(f)
            
        print(config["params"])

        omage_upper = config["estimates"]["omega_upper"]
        theta_upper = config["estimates"]["theta_max"]

        cond_1 = cond_2 = cond_3 = cond_4 = 0
        terminal_timings = []
        
        n_trials = 0

        for experiment_dir in tqdm(glob(os.path.join(setup_dir, "**"))):
            n_trials += 1

            y = np.load(os.path.join(experiment_dir, "y.npy"))
            t = np.load(os.path.join(experiment_dir, "t.npy"))
            
            cond_1_flag = check_conditions_1(
                y[0],
                threshold,
            )
            cond_1 += cond_1_flag

            cond_2 += check_conditions_2(
                y[0],
                theta_upper,
            )

            cond_3 += check_conditions_3(
                y[1],
                omage_upper,
            )

            cond_4 += check_conditions_4(
                y[3]
            )

            if cond_1_flag:
                terminal_index = np.argmax(y[0, :] < threshold)
                terminal_timings.append(t[terminal_index])

        print(round(cond_1/n_trials, 3))
        print(round(cond_2/n_trials, 3))
        print(round(cond_3/n_trials, 3))
        print(round(cond_4/n_trials, 3))
        print(
            construct_bootstrap_interval(
                terminal_timings, 
                n_trials, 
                confidence_level,
            )
        )

        print("-----------------------")