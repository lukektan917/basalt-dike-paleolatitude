"""Loading magnetometer transects and converting sample time into distance.

The field data are phone-magnetometer CSVs, one file per walked transect, with a
``time`` column in seconds and the three field components ``Bx``, ``By``, ``Bz``.
The phone records time, not position, so the first job is to turn one into the
other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["DEFAULT_TRANSECT_LENGTH", "calibrate_distance", "load_transect", "load_transects"]

#: Length of a survey transect in metres. Every transect was walked between the
#: same two marked pegs, paced to a metronome.
DEFAULT_TRANSECT_LENGTH = 0.8


def calibrate_distance(
    df: pd.DataFrame,
    total_length: float = DEFAULT_TRANSECT_LENGTH,
    time_column: str = "time",
    distance_column: str = "distance",
) -> pd.DataFrame:
    """Replace the time axis of a transect with a distance axis.

    Assumes constant walking speed between the endpoints, which is what the
    metronome in the field procedure was for. Elapsed time is mapped linearly
    onto ``[0, total_length]``.

    Returns a new frame; the input is not modified.

    Notes
    -----
    The original exploratory notebook normalised by ``n_samples * 0.1`` — the
    nominal duration at an assumed 10 Hz sample rate — and then scaled by 8.
    That happens to give the right answer for these files, which were recorded
    at 100 Hz, but it silently couples the calibration to the sample rate. This
    version normalises by the *observed* elapsed time instead, so it is correct
    for any sample rate and states its assumption in one line of arithmetic.
    """
    if time_column not in df.columns:
        raise KeyError(f"expected a {time_column!r} column, found {list(df.columns)}")

    out = df.copy()
    t = out[time_column].to_numpy(dtype=float)

    if len(t) < 2:
        raise ValueError("a transect needs at least two samples to calibrate")

    elapsed = t[-1] - t[0]
    if elapsed <= 0:
        raise ValueError("transect timestamps are not increasing")

    out[distance_column] = (t - t[0]) / elapsed * total_length
    out = out.drop(columns=[time_column])

    return out


def load_transect(path: str | Path, total_length: float = DEFAULT_TRANSECT_LENGTH) -> pd.DataFrame:
    """Load one magnetometer CSV and calibrate its distance axis."""
    return calibrate_distance(pd.read_csv(path), total_length=total_length)


def load_transects(
    directory: str | Path,
    pattern: str = "magnetometer_*.csv",
    total_length: float = DEFAULT_TRANSECT_LENGTH,
) -> list[pd.DataFrame]:
    """Load every transect in ``directory``, sorted by filename.

    Filenames carry a timestamp, so sorting by name puts the transects in the
    order they were walked.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")
    return [load_transect(p, total_length=total_length) for p in paths]
