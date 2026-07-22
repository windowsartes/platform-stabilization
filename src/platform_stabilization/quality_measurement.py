import scipy.stats as sts
import numpy as np


def get_mean_interval(
    array: np.ndarray[tuple[int], np.dtype[np.float64]],
    confidence_level: float = 0.95,
    bootstrap_sample_size: int | None = None,
) -> tuple[float, float]:
    array_size = array.shape[0]

    if bootstrap_sample_size is None:
        bootstrap_sample_size = array_size

    bootstrapped_samples = [
        np.random.choice(
            array,
            size=array_size,
            replace=True,
        ) for _ in range(100)
    ]
    bootstrapped_samples = np.array(bootstrapped_samples)
    bootstrapped_means = np.mean(
        bootstrapped_samples,
        axis=1,
    )

    return tuple(np.quantile(
        bootstrapped_means,
        q=[
            confidence_level / 2,
            (1 + confidence_level) / 2,
        ]
    ))


def get_wilcoxon_test_p_value(effects: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    return sts.wilcoxon(
        effects,
        zero_method="pratt",
        alternative="less",
        method="asymptotic",
        correction=False,
    ).pvalue
