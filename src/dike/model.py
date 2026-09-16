"""Forward model for the magnetic anomaly over a two-dimensional dike.

An infinitely long, vertically-sided tabular body of half-width ``half_width``
at depth ``depth`` below the sensor, uniformly magnetised at ``inclination``
degrees from horizontal. Closed form as in Telford, Geldart & Sheriff, *Applied
Geophysics*, 2nd ed., ch. 3: a symmetric pair of arctangent terms scaled by
``sin(inclination)``, and an antisymmetric logarithmic term scaled by
``cos(inclination)``. Their ratio is what fixes the inclination.

The parameterisation is degenerate. Negating ``depth``, ``half_width`` or
``intensity`` can each be absorbed by moving ``inclination``, so an
unconstrained fit can return a sensor height below ground and a sign-flipped
inclination while fitting just as well. :func:`fit_dike` bounds the geometry to
stay physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "bz_dike",
    "initial_guess",
    "default_bounds",
    "RESTART_INCLINATIONS",
    "DikeFit",
    "fit_dike",
    "r_squared",
    "paleolatitude_from_inclination",
    "inclination_from_paleolatitude",
]


def bz_dike(
    x: np.ndarray,
    offset: float,
    depth: float,
    half_width: float,
    inclination: float,
    intensity: float,
    baseline: float = 0.0,
) -> np.ndarray:
    """Vertical field anomaly over a 2-D dike, plus a constant baseline.

    Parameters
    ----------
    x
        Position along the profile, in metres.
    offset
        Position of the dike centre along the profile, in metres. Fitted rather
        than assumed, because the operator does not know where the dike is when
        the transect starts.
    depth
        Sensor height above the top of the dike, in metres.
    half_width
        Half-width of the dike, in metres.
    inclination
        Angle of the magnetisation vector from horizontal, in **degrees**.
    intensity
        Magnetisation intensity, in the units of the input field (here uT).
    baseline
        Constant background level, in the units of the input field. Off by
        default in :func:`fit_dike`; see ``fit_baseline`` there.

    Returns
    -------
    numpy.ndarray
        Modelled vertical field anomaly at each ``x``.

    Notes
    -----
    ``depth`` is constrained to be non-zero by the geometry; the logarithm and
    the arctangents are both singular for a sensor sitting exactly on an
    infinitely thin source. Fits are started away from zero for that reason.
    """
    x = np.asarray(x, dtype=float)
    xs = x - offset
    phi = np.deg2rad(inclination)

    symmetric = np.cos(phi) * np.log(
        ((xs - half_width) ** 2 + depth**2) / ((xs + half_width) ** 2 + depth**2)
    )
    antisymmetric = np.sin(phi) * (
        np.arctan((xs + half_width) / depth) - np.arctan((xs - half_width) / depth)
    )

    return 2.0 * intensity * (symmetric + antisymmetric) + baseline


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination for a fit.

    Reported for orientation only. The observations along a transect are
    strongly autocorrelated (see :mod:`dike.bootstrap`), so R-squared here
    should not be read as evidence about parameter uncertainty.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class DikeFit:
    """Result of fitting :func:`bz_dike` to a profile."""

    offset: float
    depth: float
    half_width: float
    inclination: float
    intensity: float
    baseline: float
    covariance: np.ndarray
    r2: float

    @property
    def params(self) -> np.ndarray:
        """Parameters as the positional array ``bz_dike`` expects."""
        return np.array(
            [
                self.offset,
                self.depth,
                self.half_width,
                self.inclination,
                self.intensity,
                self.baseline,
            ]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return bz_dike(x, *self.params)

    @property
    def paleolatitude(self) -> float:
        """Paleolatitude implied by the fitted inclination, in degrees."""
        return paleolatitude_from_inclination(self.inclination)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"offset      = {self.offset:+.4f} m\n"
            f"depth       = {self.depth:.4f} m\n"
            f"half_width  = {self.half_width:.4f} m\n"
            f"inclination = {self.inclination:.2f} deg\n"
            f"intensity   = {self.intensity:.4f}\n"
            f"baseline    = {self.baseline:+.4f}\n"
            f"R^2         = {self.r2:.3f}\n"
            f"paleolat.   = {self.paleolatitude:.2f} deg"
        )


def initial_guess(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """A starting guess read off the data rather than hard-coded.

    The dike centre is taken to be where the profile is most extreme relative to
    its flanks; width and depth start at values typical of a hand-held survey
    over a metre-scale outcrop; the magnetisation starts steep, which is the
    expectation for a mid-latitude intrusion.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    flank = 0.5 * (ys[: max(1, len(ys) // 10)].mean() + ys[-max(1, len(ys) // 10) :].mean())
    centre = float(xs[np.argmax(np.abs(ys - flank))])

    span = float(xs[-1] - xs[0])
    return (
        centre,
        0.03,
        max(0.05 * span, 0.05),
        75.0,
        max(np.ptp(ys) / 4.0, 1e-3),
        float(flank),
    )


def default_bounds(x: np.ndarray) -> tuple[list[float], list[float]]:
    """Bounds that keep the fitted geometry physical and the model identifiable.

    Three constraints, each with a reason:

    * **Positive depth and half-width, positive intensity, inclination in
      [-90, 90].** These break the sign degeneracy described in the module
      docstring. Without them the fit can return a negative sensor height — the
      phone underground — which is the same curve as the physical solution with
      the inclination's sign flipped, and so silently reverses the hemisphere.
    * **The dike centre lies within the surveyed window.** A fit that places the
      source outside the transect is describing a body the survey never crossed,
      using the far flank of a broad anomaly to imitate a local trend. Such fits
      score well and mean nothing.
    * **The half-width is at most the length of the transect.** A dike wider
      than the ground you walked is not constrained by your data.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    return (
        [lo, 1e-4, 1e-3, -90.0, 0.0, -np.inf],
        [hi, 1.0, span, 90.0, np.inf, np.inf],
    )


#: Starting inclinations tried by the multi-start fit. The objective surface has
#: more than one local minimum, and which one a single start falls into depends
#: on the initial inclination more than on anything else.
#:
#: Both signs must be present. A steeply negative inclination with the sensor
#: above the dike produces exactly the same profile as the equally steep
#: positive inclination with the sensor *below* it, so a restart set that only
#: looks at positive angles will converge to the +90 boundary rather than find
#: the physical solution on the negative side.
RESTART_INCLINATIONS = (-88.0, -70.0, -45.0, -15.0, 15.0, 45.0, 70.0, 88.0)


def fit_dike(
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple[float, ...] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    max_nfev: int = 20000,
    restarts: bool = True,
    fit_baseline: bool = False,
) -> DikeFit:
    """Bounded, multi-start least-squares fit of :func:`bz_dike` to a profile.

    ``p0`` defaults to :func:`initial_guess` and ``bounds`` to
    :func:`default_bounds`.

    With ``restarts`` the fit is repeated from several starting inclinations and
    the best-fitting result is kept. This is not defensive padding: the
    objective has multiple local minima, and a single start from a hand-chosen
    guess — the original notebook's approach — can settle in one where the dike
    is a metre wide and the inclination is meaningless, while still reporting a
    respectable R-squared.

    ``fit_baseline`` decides whether the constant background level is a free
    parameter. **It defaults to False, and the choice depends on the transect.**

    * If the traverse extends far enough past the dike that the profile flattens
      on *both* sides, the background is pinned down by those flanks and fitting
      it removes a real bias — on synthetic transects, leaving it out costs
      several degrees of inclination.
    * If the traverse does not, the baseline is not identified: it trades off
      against the anomaly's own amplitude and width, and the fit slides the dike
      centre to the edge of the window. On the Middlesex Fells transects that is
      exactly what happens, so the baseline is pinned to zero there and the
      alignment step is relied on to set the level.

    Turning it on when the flanks do not support it does not produce a worse
    R-squared — it produces a *better* one, attached to a geometry that is not
    real. That is the trap this flag exists to make explicit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if bounds is None:
        bounds = default_bounds(x)
        if not fit_baseline:
            bounds[0][5], bounds[1][5] = 0.0, 1e-9
    lower, upper = bounds

    default_start = initial_guess(x, y)
    candidates: list[tuple[float, ...]] = []
    if p0 is not None:
        candidates.append(tuple(p0))
    candidates.append(default_start)
    if restarts:
        # Vary the starting inclination over both signs, and start the baseline
        # both at the observed flank level and at zero. A free baseline is a
        # strictly richer model than a pinned one, so if the search only ever
        # starts from one baseline it can score *worse* than the constrained
        # fit — which is a search failure, not a modelling result.
        baselines = (default_start[5], 0.0) if fit_baseline else (0.0,)
        for baseline in baselines:
            for inclination in RESTART_INCLINATIONS:
                start = list(default_start)
                start[3] = inclination
                start[5] = baseline
                candidates.append(tuple(start))

    best: tuple[np.ndarray, np.ndarray] | None = None
    best_sse = np.inf
    failures: list[Exception] = []

    for candidate in candidates:
        start = np.clip(np.asarray(candidate, dtype=float), lower, upper)
        try:
            params, cov = curve_fit(
                bz_dike, x, y, p0=start, bounds=bounds, max_nfev=max_nfev
            )
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - rare
            failures.append(exc)
            continue
        sse = float(np.sum((y - bz_dike(x, *params)) ** 2))
        if sse < best_sse:
            best_sse, best = sse, (params, cov)

    if best is None:
        raise RuntimeError(f"every start failed to converge; last error: {failures[-1]}")

    params, cov = best
    fitted = bz_dike(x, *params)

    return DikeFit(
        offset=float(params[0]),
        depth=float(params[1]),
        half_width=float(params[2]),
        inclination=float(params[3]),
        intensity=float(params[4]),
        baseline=float(params[5]),
        covariance=np.asarray(cov),
        r2=r_squared(y, fitted),
    )


def paleolatitude_from_inclination(inclination: float | np.ndarray) -> float | np.ndarray:
    """Paleolatitude from magnetic inclination, in degrees.

    Uses the geocentric axial dipole (GAD) relation ``tan(I) = 2 tan(lambda)``,
    the standard assumption of paleomagnetism: averaged over enough time, the
    geomagnetic field looks like a dipole aligned with the spin axis, so the
    inclination recorded by a rock fixes the latitude at which it cooled.

    Caveats worth stating plainly, because they bound what this repository can
    claim:

    * A single dike records an instant, not a time average. Secular variation
      is not averaged out, so the GAD assumption is weaker here than it would
      be for a sequence of flows.
    * The relation is insensitive to the sign convention of the hemisphere; a
      magnitude is what is recovered.
    """
    inclination = np.asarray(inclination, dtype=float)
    result = np.rad2deg(np.arctan(np.tan(np.deg2rad(inclination)) / 2.0))
    return float(result) if result.ndim == 0 else result


def inclination_from_paleolatitude(latitude: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`paleolatitude_from_inclination`."""
    latitude = np.asarray(latitude, dtype=float)
    result = np.rad2deg(np.arctan(2.0 * np.tan(np.deg2rad(latitude))))
    return float(result) if result.ndim == 0 else result
