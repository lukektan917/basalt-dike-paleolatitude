"""Cropping and vertically aligning repeated transects.

Repeated runs over the same dike cannot simply be pooled. Each carries an
unknown DC offset — ambient field plus whatever the phone contributes in the
orientation it was held — and the first and last samples of each run are
contaminated by starting and stopping. So: crop the ends, then shift each run
onto a common level.

Which level is a real analytical choice, not a detail; see
:func:`align_transects`.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

__all__ = ["Anchor", "crop", "anchor_value", "align_transects"]

Anchor = Literal["begin", "middle", "end"]


def crop(
    frames: Sequence[pd.DataFrame],
    start: float = 0.15,
    end: float = 0.70,
    distance_column: str = "distance",
) -> list[pd.DataFrame]:
    """Restrict every transect to ``[start, end]`` metres."""
    cropped = []
    for df in frames:
        d = df[(df[distance_column] >= start) & (df[distance_column] <= end)].copy()
        if d.empty:
            raise ValueError(f"cropping to [{start}, {end}] left a transect empty")
        cropped.append(d.reset_index(drop=True))
    return cropped


def anchor_value(
    df: pd.DataFrame,
    at: Anchor,
    start: float | None = None,
    end: float | None = None,
    distance_column: str = "distance",
    field_column: str = "Bz",
) -> float:
    """Value of ``field_column`` at the chosen anchor point of one transect.

    ``middle`` interpolates rather than taking a nearest sample, so the anchor
    does not jitter with sampling.
    """
    d = df[distance_column].to_numpy(dtype=float)
    b = df[field_column].to_numpy(dtype=float)

    if at == "begin":
        return float(b[0])
    if at == "end":
        return float(b[-1])
    if at == "middle":
        lo = d[0] if start is None else start
        hi = d[-1] if end is None else end
        order = np.argsort(d)
        return float(np.interp(0.5 * (lo + hi), d[order], b[order]))

    raise ValueError("at must be 'begin', 'middle' or 'end'")


def align_transects(
    frames: Sequence[pd.DataFrame],
    start: float = 0.15,
    end: float = 0.70,
    at: Anchor = "middle",
    strategy: Literal["median", "reference", "zero"] = "reference",
    reference_index: int = 0,
    distance_column: str = "distance",
    field_column: str = "Bz",
) -> tuple[list[pd.DataFrame], float]:
    """Crop and vertically align a set of transects onto a common baseline.

    Parameters
    ----------
    strategy
        How the common baseline is chosen:

        ``"reference"``
            Shift every transect so its anchor matches that of transect
            ``reference_index``. Keeps the data on the physical scale of one
            real measurement. **This is the default**, and the one used for the
            published fit.
        ``"median"``
            Shift every transect onto the median anchor value across runs.
            Robust, and sensible when no single run is privileged, but the
            resulting baseline is a number no instrument actually recorded.
        ``"zero"``
            Force the anchor to zero on every run. Superficially the tidiest
            option, and the one to avoid here: the field either side of the dike
            is not zero, so zeroing at an endpoint drags the flanks of the
            profile to the wrong sign and the model fit degrades badly. It is
            kept because reproducing that failure is part of the record.

    at
        Which point of the cropped profile to use as the anchor. ``"middle"``
        is preferred: the field procedure put a metronome beat at the midpoint,
        so it is the position best known across runs, and unlike the endpoints
        it sits where the signal is strong.

    Returns
    -------
    (aligned, baseline)
        The shifted transects, and the baseline value they now share — useful
        for drawing the reference line on a plot.
    """
    cropped = crop(frames, start=start, end=end, distance_column=distance_column)

    values = [
        anchor_value(df, at, start, end, distance_column, field_column) for df in cropped
    ]

    if strategy == "zero":
        baseline = 0.0
        shifts = values
    elif strategy == "median":
        baseline = float(np.median(values))
        shifts = [v - baseline for v in values]
    elif strategy == "reference":
        if not 0 <= reference_index < len(cropped):
            raise IndexError(f"reference_index {reference_index} out of range")
        baseline = float(values[reference_index])
        shifts = [v - baseline for v in values]
    else:
        raise ValueError("strategy must be 'median', 'reference' or 'zero'")

    aligned = []
    for df, shift in zip(cropped, shifts):
        out = df.copy()
        out[field_column] = df[field_column] - shift
        aligned.append(out)

    return aligned, baseline


def pool(
    frames: Sequence[pd.DataFrame],
    distance_column: str = "distance",
    field_column: str = "Bz",
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate aligned transects into a single (x, y) sample for fitting."""
    x = np.concatenate([df[distance_column].to_numpy(dtype=float) for df in frames])
    y = np.concatenate([df[field_column].to_numpy(dtype=float) for df in frames])
    return x, y
