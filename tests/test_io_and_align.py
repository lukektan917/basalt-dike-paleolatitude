import numpy as np
import pandas as pd
import pytest

from dike.align import align_transects, anchor_value, crop, pool
from dike.io import calibrate_distance, load_transects


def make_frame(n=101, rate=100.0, offset=0.0):
    time = np.arange(n) / rate
    return pd.DataFrame({"time": time, "Bz": np.linspace(0.0, 10.0, n) + offset})


def test_calibration_spans_the_transect_length():
    out = calibrate_distance(make_frame(), total_length=0.8)
    assert out["distance"].iloc[0] == pytest.approx(0.0)
    assert out["distance"].iloc[-1] == pytest.approx(0.8)
    assert "time" not in out.columns


def test_calibration_is_independent_of_sample_rate():
    """The old nominal-10Hz formula was not; this one is."""
    slow = calibrate_distance(make_frame(n=101, rate=10.0))
    fast = calibrate_distance(make_frame(n=101, rate=1000.0))
    assert np.allclose(slow["distance"], fast["distance"])


def test_calibration_leaves_the_input_untouched():
    df = make_frame()
    calibrate_distance(df)
    assert "time" in df.columns


def test_calibration_rejects_degenerate_input():
    with pytest.raises(ValueError):
        calibrate_distance(pd.DataFrame({"time": [1.0], "Bz": [0.0]}))
    with pytest.raises(KeyError):
        calibrate_distance(pd.DataFrame({"t": [1.0, 2.0], "Bz": [0.0, 1.0]}))


def test_load_transects_errors_clearly_on_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transects(tmp_path)


def test_crop_restricts_the_distance_range():
    frames = [calibrate_distance(make_frame())]
    cropped = crop(frames, 0.2, 0.6)
    assert cropped[0]["distance"].min() >= 0.2
    assert cropped[0]["distance"].max() <= 0.6


def test_crop_rejects_an_empty_window():
    frames = [calibrate_distance(make_frame())]
    with pytest.raises(ValueError):
        crop(frames, 5.0, 6.0)


def test_anchor_middle_interpolates():
    df = calibrate_distance(make_frame())
    mid = anchor_value(df, "middle", 0.0, 0.8)
    assert mid == pytest.approx(5.0, abs=0.1)


def test_alignment_removes_per_run_offsets():
    """Three identical runs with different DC offsets should land on top."""
    frames = [calibrate_distance(make_frame(offset=o)) for o in (0.0, 12.0, -7.5)]
    aligned, baseline = align_transects(frames, 0.15, 0.70, at="middle", strategy="reference")

    reference = aligned[0]["Bz"].to_numpy()
    for df in aligned[1:]:
        assert np.allclose(df["Bz"].to_numpy(), reference, atol=1e-9)
    assert baseline == pytest.approx(reference[len(reference) // 2], abs=0.2)


def test_median_strategy_centres_on_the_median_run():
    """The baseline should be the anchor of the middle run, not of an outlier."""
    frames = [calibrate_distance(make_frame(offset=o)) for o in (0.0, 5.0, 100.0)]
    _, baseline = align_transects(frames, 0.15, 0.70, at="middle", strategy="median")

    expected = anchor_value(crop(frames, 0.15, 0.70)[1], "middle", 0.15, 0.70)
    assert baseline == pytest.approx(expected)


def test_zero_strategy_puts_the_anchor_at_zero():
    frames = [calibrate_distance(make_frame(offset=o)) for o in (0.0, 5.0)]
    aligned, baseline = align_transects(frames, 0.15, 0.70, at="end", strategy="zero")
    assert baseline == 0.0
    for df in aligned:
        assert df["Bz"].iloc[-1] == pytest.approx(0.0)


def test_unknown_strategy_is_rejected():
    frames = [calibrate_distance(make_frame())]
    with pytest.raises(ValueError):
        align_transects(frames, strategy="nonsense")


def test_pool_concatenates_every_sample():
    frames = [calibrate_distance(make_frame()) for _ in range(3)]
    x, y = pool(frames)
    assert len(x) == len(y) == 3 * len(frames[0])
