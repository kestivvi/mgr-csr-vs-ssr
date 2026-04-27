from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def calculate_confidence_interval(data: pd.Series) -> Tuple[float, float]:
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(data))
    std_err = stats.sem(data)
    if std_err == 0 or np.isnan(std_err):
        return (mean, mean)
    interval = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std_err)
    return (max(0.0, float(interval[0])), float(interval[1]))


def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    n1, n2 = len(group1), len(group2)
    s1, s2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))
    if (n1 + n2 - 2) == 0:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)
