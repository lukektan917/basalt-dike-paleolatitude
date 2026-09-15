import numpy as np
import pytest

from dike.model import (
    bz_dike,
    default_bounds,
    fit_dike,
    inclination_from_paleolatitude,
    paleolatitude_from_inclination,
    r_squared,
)


def test_horizontal_magnetisation_is_antisymmetric():
    """At zero inclination only the logarithmic term survives, and it is odd:
    a horizontally magnetised dike gives a profile that rises on one flank and
    falls on the other."""
    x = np.linspace(-1.0, 1.0, 201)
    y = bz_dike(x, offset=0.0, depth=0.05, half_width=0.17, inclination=0.0, intensity=10.0)
    assert np.allclose(y, -y[::-1], atol=1e-9)


def test_vertical_magnetisation_is_symmetric():
    """At 90 degrees only the arctangent terms survive, and they are even:
    a vertically magnetised dike gives a symmetric peak over its centre."""
    x = np.linspace(-1.0, 1.0, 201)
    y = bz_dike(x, offset=0.0, depth=0.05, half_width=0.17, inclination=90.0, intensity=10.0)
    assert np.allclose(y, y[::-1], atol=1e-9)
    assert np.argmax(np.abs(y)) == pytest.approx(len(y) // 2, abs=1)


def test_offset_translates_the_profile():
    x = np.linspace(-1.0, 1.0, 201)
    base = bz_dike(x, 0.0, 0.05, 0.17, 60.0, 10.0)
    shifted = bz_dike(x + 0.3, 0.3, 0.05, 0.17, 60.0, 10.0)
    assert np.allclose(base, shifted)


def test_intensity_scales_linearly():
    x = np.linspace(-1.0, 1.0, 101)
    single = bz_dike(x, 0.0, 0.05, 0.17, 60.0, 1.0)
    triple = bz_dike(x, 0.0, 0.05, 0.17, 60.0, 3.0)
    assert np.allclose(3.0 * single, triple)


def test_anomaly_decays_away_from_the_dike():
    """Far from a 2-D source the anomaly dies off; the logarithmic term makes
    that decay slow, so the test is relative rather than absolute."""
    near = abs(bz_dike(np.array([0.0]), 0.0, 0.05, 0.17, 60.0, 10.0)[0])
    mid = abs(bz_dike(np.array([5.0]), 0.0, 0.05, 0.17, 60.0, 10.0)[0])
    far = abs(bz_dike(np.array([50.0]), 0.0, 0.05, 0.17, 60.0, 10.0)[0])
    assert far < mid < near
    assert far < 0.01 * near


def test_fit_recovers_known_parameters_from_clean_data():
    truth = dict(offset=0.40, depth=0.03, half_width=0.17, inclination=68.0, intensity=10.0)
    x = np.linspace(0.15, 0.70, 800)
    y = bz_dike(x, **truth)

    fit = fit_dike(x, y)

    assert fit.inclination == pytest.approx(truth["inclination"], abs=0.5)
    assert fit.half_width == pytest.approx(truth["half_width"], abs=0.01)
    assert fit.r2 > 0.999


def test_r_squared_is_one_for_a_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_is_nan_for_constant_observations():
    assert np.isnan(r_squared(np.ones(5), np.ones(5)))


@pytest.mark.parametrize("latitude", [0.0, 15.0, 45.0, 60.0, -30.0])
def test_paleolatitude_round_trips(latitude):
    inclination = inclination_from_paleolatitude(latitude)
    assert paleolatitude_from_inclination(inclination) == pytest.approx(latitude)


def test_dipole_relation_at_the_equator_and_pole():
    assert paleolatitude_from_inclination(0.0) == pytest.approx(0.0)
    assert paleolatitude_from_inclination(90.0) == pytest.approx(90.0)
    # tan(I) = 2 tan(lambda): 45 degrees of latitude gives a steeper inclination.
    assert paleolatitude_from_inclination(63.4349) == pytest.approx(45.0, abs=1e-3)


def test_fit_stays_in_the_physical_basin_from_a_poor_start():
    """The bounds exist so a bad starting guess cannot produce a mirror
    solution with negative width and a meaningless inclination."""
    truth = dict(offset=0.40, depth=0.03, half_width=0.17, inclination=68.0, intensity=10.0)
    x = np.linspace(0.15, 0.70, 800)
    y = bz_dike(x, **truth)

    fit = fit_dike(x, y, p0=(-0.17, 0.03, 0.17, 5.0, 10.0, 0.0))

    assert fit.half_width > 0
    assert fit.depth > 0
    assert -90.0 <= fit.inclination <= 90.0
    assert fit.inclination == pytest.approx(truth["inclination"], abs=1.0)


def test_restarts_never_return_a_worse_fit():
    """Multi-start keeps the best of several starts, so by construction it can
    only match or beat any single start. Checked on noisy data, where the
    objective surface is not as forgiving as it is on a clean curve."""
    truth = dict(offset=0.40, depth=0.03, half_width=0.17, inclination=68.0, intensity=10.0)
    rng = np.random.default_rng(4)
    x = np.linspace(0.15, 0.70, 800)
    y = bz_dike(x, **truth) + rng.normal(0.0, 3.0, size=x.size)

    single = fit_dike(x, y, p0=(-0.17, 0.03, 0.17, 5.0, 10.0, 0.0), restarts=False)
    multi = fit_dike(x, y, p0=(-0.17, 0.03, 0.17, 5.0, 10.0, 0.0), restarts=True)

    assert multi.r2 >= single.r2 - 1e-9


def test_baseline_is_recovered():
    truth = dict(offset=0.40, depth=0.03, half_width=0.17, inclination=68.0, intensity=10.0)
    x = np.linspace(0.15, 0.70, 800)
    y = bz_dike(x, **truth, baseline=12.5)

    fit = fit_dike(x, y, fit_baseline=True)

    assert fit.baseline == pytest.approx(12.5, abs=0.5)
    assert fit.inclination == pytest.approx(truth["inclination"], abs=1.0)


def test_baseline_defaults_to_zero_in_the_forward_model():
    x = np.linspace(-0.5, 0.5, 51)
    assert np.allclose(
        bz_dike(x, 0.0, 0.05, 0.17, 60.0, 10.0),
        bz_dike(x, 0.0, 0.05, 0.17, 60.0, 10.0, 0.0),
    )


def test_a_wrong_dc_level_biases_the_inclination_without_a_baseline_term():
    """The reason ``baseline`` is fitted rather than assumed away.

    Offsetting the data by a constant and then refusing to fit that constant
    does not leave the inclination alone; it moves it by degrees.
    """
    truth = dict(offset=0.40, depth=0.03, half_width=0.17, inclination=68.0, intensity=10.0)
    x = np.linspace(0.15, 0.70, 800)
    y = bz_dike(x, **truth) + 8.0

    without = fit_dike(x, y, fit_baseline=False)
    with_baseline = fit_dike(x, y, fit_baseline=True)

    assert abs(without.inclination - truth["inclination"]) > 2.0
    assert with_baseline.inclination == pytest.approx(truth["inclination"], abs=1.0)


def test_negative_depth_is_the_mirror_of_a_flipped_inclination():
    """The degeneracy that reverses a hemisphere.

    A sensor at height +z over a dike inclined -phi records exactly the profile
    that a sensor at -z over a dike inclined +phi would. The data cannot tell
    them apart; only the fact that the instrument was above ground can. This is
    why :func:`default_bounds` forbids negative depth.
    """
    x = np.linspace(0.15, 0.70, 400)
    physical = bz_dike(x, 0.285, +0.0257, 0.229, -77.9, 12.3)
    mirrored = bz_dike(x, 0.285, -0.0257, 0.229, +77.9, 12.3)

    assert np.allclose(physical, mirrored, atol=1e-12)
    assert paleolatitude_from_inclination(-77.9) == pytest.approx(
        -paleolatitude_from_inclination(77.9)
    )


def test_bounds_keep_the_source_inside_the_surveyed_window():
    x = np.linspace(0.15, 0.70, 100)
    lower, upper = default_bounds(x)
    assert lower[0] == pytest.approx(0.15)
    assert upper[0] == pytest.approx(0.70)
    assert lower[1] > 0.0          # depth strictly positive
    assert upper[2] <= 0.70 - 0.15 + 1e-9   # half-width no wider than the transect
