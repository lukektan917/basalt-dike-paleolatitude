"""Synthetic transects, so the pipeline runs without the field data.

The raw survey CSVs are not redistributed with this repository. Everything in
:mod:`dike` therefore has to be exercisable against data generated here: a known
dike, a known magnetisation, correlated noise, and a different unknown baseline
offset on every run — the same three nuisances the real analysis has to survive.

Because the true parameters are known, this module doubles as the correctness
check for the whole pipeline: run the analysis on synthetic data and the fitted
inclination should come back close to the value that went in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import DEFAULT_TRANSECT_LENGTH
from .model import bz_dike

__all__ = ["TRUE_PARAMETERS", "synthetic_transect", "synthetic_transects"]

#: Plausible stand-in values: a ~34 cm dike, a phone 3 cm above it, steep
#: magnetisation. Not the fitted values from the field survey.
TRUE_PARAMETERS = {
    "offset": 0.40,
    "depth": 0.03,
    "half_width": 0.17,
    "inclination": 68.0,
    "intensity": 10.0,
}


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One AR(1) sequence, started from its stationary distribution."""
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - rho**2))
    innovations = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def synthetic_transect(
    n_samples: int = 1000,
    sample_rate: float = 100.0,
    total_length: float = DEFAULT_TRANSECT_LENGTH,
    rho: float = 0.99,
    sigma: float = 0.35,
    baseline: float = 0.0,
    rng: np.random.Generator | None = None,
    **parameters: float,
) -> pd.DataFrame:
    """One synthetic transect, in the same shape as a real magnetometer CSV.

    The returned frame has a ``time`` column, not a ``distance`` column, so it
    goes through :func:`dike.io.calibrate_distance` exactly as real data does.
    """
    rng = np.random.default_rng() if rng is None else rng
    values = {**TRUE_PARAMETERS, **parameters}

    time = np.arange(n_samples) / sample_rate
    distance = time / time[-1] * total_length

    signal = bz_dike(
        distance,
        values["offset"],
        values["depth"],
        values["half_width"],
        values["inclination"],
        values["intensity"],
        baseline,
    )
    field = signal + _ar1(n_samples, rho, sigma, rng)

    return pd.DataFrame({"time": time, "Bz": field})


def synthetic_transects(
    n_transects: int = 10,
    baseline_spread: float = 15.0,
    seed: int | None = 0,
    **kwargs: float,
) -> list[pd.DataFrame]:
    """A survey's worth of synthetic transects with differing DC offsets.

    ``baseline_spread`` is the standard deviation of the per-run offset, which
    stands in for the ambient field and hard-iron contribution that the
    alignment step exists to remove.
    """
    rng = np.random.default_rng(seed)
    return [
        synthetic_transect(baseline=rng.normal(0.0, baseline_spread), rng=rng, **kwargs)
        for _ in range(n_transects)
    ]
