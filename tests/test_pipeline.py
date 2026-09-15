"""End-to-end check: can the pipeline recover a known answer?

These are the tests that matter. Everything else checks that a function does
what its docstring says; these check that the analysis as a whole returns the
truth when the truth is known.
"""

import numpy as np
import pytest

from dike.align import align_transects, pool
from dike.bootstrap import bootstrap_inclination, fit_ar1
from dike.io import calibrate_distance
from dike.model import fit_dike
from dike.synthetic import TRUE_PARAMETERS, synthetic_transects


@pytest.fixture(scope="module")
def pipeline():
    raw = synthetic_transects(n_transects=10, seed=7)
    transects = [calibrate_distance(df) for df in raw]
    aligned, _ = align_transects(transects, 0.15, 0.70, at="middle", strategy="reference")
    x, y = pool(aligned)
    return x, y, fit_dike(x, y)


def test_pipeline_recovers_the_true_inclination(pipeline):
    _, _, fit = pipeline
    assert fit.inclination == pytest.approx(TRUE_PARAMETERS["inclination"], abs=8.0)


def test_pipeline_recovers_the_true_half_width(pipeline):
    _, _, fit = pipeline
    assert fit.half_width == pytest.approx(TRUE_PARAMETERS["half_width"], abs=0.06)


def test_residuals_are_strongly_autocorrelated(pipeline):
    """The premise of the whole uncertainty argument."""
    x, y, fit = pipeline
    ar_model = fit_ar1(y - fit.predict(x))
    assert ar_model.params[1] > 0.9


def test_bootstrap_interval_brackets_the_truth(pipeline):
    x, _, fit = pipeline
    ar_model = fit_ar1(_bootstrap_residuals(pipeline))
    x_grid = np.linspace(x.min(), x.max(), 400)
    result = bootstrap_inclination(x_grid, fit.params, ar_model, n_boot=60, seed=3)

    lo, hi = result.interval(result.inclination)
    assert lo < TRUE_PARAMETERS["inclination"] < hi
    assert result.converged > 0


def test_bootstrap_interval_is_wider_than_naive_errors(pipeline):
    """Correlated residuals must inflate the interval, or the exercise is pointless."""
    x, _, fit = pipeline
    naive_sd = np.sqrt(np.diag(fit.covariance))[3]

    ar_model = fit_ar1(_bootstrap_residuals(pipeline))
    x_grid = np.linspace(x.min(), x.max(), 400)
    result = bootstrap_inclination(x_grid, fit.params, ar_model, n_boot=60, seed=3)

    assert result.inclination.std(ddof=1) > naive_sd


def _bootstrap_residuals(pipeline):
    x, y, fit = pipeline
    return y - fit.predict(x)
