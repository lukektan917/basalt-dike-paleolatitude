"""Uncertainty on the fitted inclination, accounting for correlated residuals.

The naive standard errors that come out of ``scipy.optimize.curve_fit`` assume
independent, identically distributed errors. Along a magnetometer transect that
assumption is badly wrong: consecutive samples are a hundredth of a second and a
few millimetres apart, and the residual from the smooth dike model is dominated
by real, spatially coherent structure — small-scale heterogeneity in the
outcrop, the operator's hand wobbling, the phone's own drift. Fitting an AR(1)
model to the pooled residuals of this survey returns an autoregressive
coefficient near 0.99.

With correlation that strong the *effective* sample size is a tiny fraction of
the nominal one, and i.i.d. standard errors are optimistic by roughly an order
of magnitude. The honest route is a parametric bootstrap: model the residual
process, simulate new profiles from the fitted dike plus simulated residuals,
refit each one, and read the spread of the refitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .model import bz_dike, fit_dike, paleolatitude_from_inclination

__all__ = ["fit_ar1", "BootstrapResult", "bootstrap_inclination"]


def _simulate_seed_keyword(ar_model) -> str:
    """statsmodels renamed ``random_state`` to ``rng`` in 0.15; support both."""
    import inspect

    parameters = inspect.signature(ar_model.simulate).parameters
    return "rng" if "rng" in parameters else "random_state"


def fit_ar1(residuals: np.ndarray):
    """Fit a mean-centred AR(1) model to pooled residuals.

    Returns the fitted ``statsmodels`` results object, whose ``simulate``
    method generates new residual sequences with the same correlation
    structure.
    """
    residuals = np.asarray(residuals, dtype=float)
    return ARIMA(residuals - residuals.mean(), order=(1, 0, 0)).fit()


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution of the fitted parameters."""

    #: One row per successful replicate, columns ordered as ``bz_dike`` takes them.
    parameters: np.ndarray
    #: Parameters of the fit to the real data.
    observed: np.ndarray
    #: Replicates attempted, including those that failed to converge.
    attempted: int

    @property
    def inclination(self) -> np.ndarray:
        return self.parameters[:, 3]

    @property
    def paleolatitude(self) -> np.ndarray:
        return paleolatitude_from_inclination(self.inclination)

    @property
    def converged(self) -> int:
        return self.parameters.shape[0]

    def interval(self, values: np.ndarray, level: float = 95.0) -> tuple[float, float]:
        """Percentile interval at the given confidence level."""
        tail = (100.0 - level) / 2.0
        lo, hi = np.percentile(values, [tail, 100.0 - tail])
        return float(lo), float(hi)

    def summary(self, level: float = 95.0) -> str:  # pragma: no cover - presentation
        inc_lo, inc_hi = self.interval(self.inclination, level)
        lat_lo, lat_hi = self.interval(self.paleolatitude, level)
        observed_lat = paleolatitude_from_inclination(float(self.observed[3]))
        return (
            f"replicates      : {self.converged}/{self.attempted} converged\n"
            f"inclination     : {self.observed[3]:.2f} deg "
            f"({level:g}% CI {inc_lo:.2f} to {inc_hi:.2f})\n"
            f"paleolatitude   : {observed_lat:.2f} deg "
            f"({level:g}% CI {lat_lo:.2f} to {lat_hi:.2f})"
        )


def bootstrap_inclination(
    x_grid: np.ndarray,
    observed_params: np.ndarray,
    ar_model,
    n_boot: int = 2000,
    seed: int | None = 0,
    fit_baseline: bool = False,
) -> BootstrapResult:
    """Parametric bootstrap of the dike fit under an AR(1) residual process.

    Each replicate evaluates the fitted model on ``x_grid``, adds a simulated
    AR(1) residual sequence, and refits. Replicates that fail to converge are
    dropped and counted rather than retried, so a fit that is unstable shows up
    as a low convergence rate instead of being quietly hidden.

    ``x_grid`` should have about as many points as a single real transect. Using
    a far denser grid would understate the uncertainty, because it would imply
    more independent information than the survey actually collected.

    ``fit_baseline`` must match the setting used for the observed fit, or the
    replicates are being fitted with a different model than the estimate they
    are meant to describe.

    Each replicate is fitted from the observed parameters without multi-start.
    That is safe here only because the bounds exclude the mirror solutions; a
    replicate cannot quietly land in a flipped-sign basin and skew the interval.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    observed_params = np.asarray(observed_params, dtype=float)
    rng = np.random.default_rng(seed)

    baseline = bz_dike(x_grid, *observed_params)
    replicates = []
    seed_keyword = _simulate_seed_keyword(ar_model)

    for _ in range(n_boot):
        simulated = ar_model.simulate(
            nsimulations=len(x_grid),
            **{seed_keyword: int(rng.integers(0, 2**31 - 1))},
        )
        try:
            fit = fit_dike(
                x_grid,
                baseline + np.asarray(simulated),
                p0=tuple(observed_params),
                fit_baseline=fit_baseline,
                restarts=False,
            )
        except RuntimeError:
            continue
        replicates.append(fit.params)

    if not replicates:
        raise RuntimeError("no bootstrap replicate converged")

    return BootstrapResult(
        parameters=np.vstack(replicates),
        observed=observed_params,
        attempted=n_boot,
    )
