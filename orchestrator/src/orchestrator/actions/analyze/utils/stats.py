from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class GroupStats:
    n: int
    median: float
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p75: float
    iqr: float


def compute_group_stats(values: Sequence[float]) -> GroupStats:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("compute_group_stats requires at least one value")
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return GroupStats(
        n=n,
        median=float(np.median(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if n >= 2 else float("nan"),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p25=p25,
        p75=p75,
        iqr=p75 - p25,
    )


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
