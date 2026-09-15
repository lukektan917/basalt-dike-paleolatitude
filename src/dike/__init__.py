"""Paleolatitude of a basalt dike from a phone-magnetometer survey.

Ten transects were walked across a basalt dike in the Middlesex Fells, north of
Boston, with a phone magnetometer. This package turns those transects into an
estimate of the magnetic inclination recorded by the dike, and from that an
estimate of the latitude at which it cooled — with an uncertainty that takes the
strong autocorrelation of the residuals seriously.

Typical use::

    from dike import io, align, model, bootstrap

    transects = io.load_transects("data/high_50cm")
    aligned, baseline = align.align_transects(transects, start=0.15, end=0.70)
    x, y = align.pool(aligned)
    fit = model.fit_dike(x, y)
    print(fit.paleolatitude)
"""

from . import align, bootstrap, io, model, plots, synthetic
from .model import DikeFit, bz_dike, fit_dike, paleolatitude_from_inclination

__version__ = "0.1.0"

__all__ = [
    "align",
    "bootstrap",
    "io",
    "model",
    "plots",
    "synthetic",
    "DikeFit",
    "bz_dike",
    "fit_dike",
    "paleolatitude_from_inclination",
]
