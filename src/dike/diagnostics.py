"""Checks that have to pass before a fitted inclination means anything.

The dike model is fitted to ``Bz``, and it interprets ``Bz`` as the *vertical*
component of the field. That interpretation rests on two assumptions about how
the data was collected, neither of which the fit itself can verify:

1. **The sensor frame is the Earth frame.** A phone reports its field components
   in the phone's own frame. If the phone is not held in a fixed orientation,
   ``Bz`` is some varying projection of the field rather than its vertical part,
   and the symmetric/antisymmetric balance that the inclination is read from is
   corrupted by the rotation.
2. **The sensor is measuring the ambient field.** The magnitude ``|B|`` away
   from the dike should be near the local geomagnetic field strength, about
   50 uT in New England. A background magnitude far from that means a large
   instrument bias or a nearby magnetic object, and a bias *vector* does not
   subtract out of ``|B|`` the way it subtracts out of a single component.

A fit can return R-squared above 0.99 while both assumptions are violated, so
these checks are not optional. :func:`diagnose` runs them and
:func:`frame_consistency_figure` draws the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import ACCENT, DATA_GREY, MODEL, _titles

__all__ = ["EARTH_FIELD_UT", "RunDiagnostics", "diagnose", "frame_consistency_figure"]

#: Approximate total geomagnetic field strength in eastern Massachusetts (uT).
EARTH_FIELD_UT = 52.0


@dataclass(frozen=True)
class RunDiagnostics:
    """Frame and calibration diagnostics for one transect."""

    index: int
    background_magnitude: float
    peak_magnitude: float
    direction_change_deg: float
    bz_swing: float

    @property
    def magnitude_ratio(self) -> float:
        """Background |B| relative to the expected geomagnetic field."""
        return self.background_magnitude / EARTH_FIELD_UT

    @property
    def plausible_magnitude(self) -> bool:
        return 0.5 <= self.magnitude_ratio <= 2.0

    @property
    def stable_orientation(self) -> bool:
        return self.direction_change_deg <= 20.0

    @property
    def usable(self) -> bool:
        return self.plausible_magnitude and self.stable_orientation


def diagnose(
    frames: list[pd.DataFrame],
    background: float = 0.08,
    tail: float = 0.72,
) -> list[RunDiagnostics]:
    """Frame and calibration checks for every transect.

    ``background`` and ``tail`` delimit the stretches at each end of the
    transect that are taken to be off the dike. The direction of the field in
    those two stretches should agree: both sample the same ambient field, so a
    large angle between them means the sensor moved, not that the rock changed.
    """
    out = []
    for i, df in enumerate(frames):
        head = df[df["distance"] < background][["Bx", "By", "Bz"]].mean().to_numpy(float)
        rear = df[df["distance"] > tail][["Bx", "By", "Bz"]].mean().to_numpy(float)

        unit_head = head / np.linalg.norm(head)
        unit_rear = rear / np.linalg.norm(rear)
        angle = float(np.degrees(np.arccos(np.clip(unit_head @ unit_rear, -1.0, 1.0))))

        out.append(
            RunDiagnostics(
                index=i + 1,
                background_magnitude=float(np.linalg.norm(head)),
                peak_magnitude=float(df["Btotal"].max()),
                direction_change_deg=angle,
                bz_swing=float(df["Bz"].max() - df["Bz"].min()),
            )
        )
    return out


def report(diagnostics: list[RunDiagnostics]) -> str:
    """A plain-text table of the checks, for the CLI."""
    lines = [
        f"{'run':>4} {'|B| bg':>8} {'vs Earth':>9} {'dir drift':>10} {'Bz swing':>9}  verdict",
        "-" * 62,
    ]
    for d in diagnostics:
        verdict = "ok" if d.usable else ", ".join(
            filter(
                None,
                [
                    None if d.plausible_magnitude else "magnitude",
                    None if d.stable_orientation else "orientation",
                ],
            )
        )
        lines.append(
            f"{d.index:>4} {d.background_magnitude:8.1f} {d.magnitude_ratio:8.1f}x "
            f"{d.direction_change_deg:9.1f}° {d.bz_swing:9.1f}  {verdict}"
        )
    usable = sum(d.usable for d in diagnostics)
    lines.append("-" * 62)
    lines.append(f"{usable} of {len(diagnostics)} runs pass both checks.")
    return "\n".join(lines)


def frame_consistency_figure(diagnostics: list[RunDiagnostics], axes=None):
    """Both checks side by side, with the acceptable region marked."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))

    runs = [d.index for d in diagnostics]
    magnitudes = [d.background_magnitude for d in diagnostics]
    drifts = [d.direction_change_deg for d in diagnostics]

    ax = axes[0]
    ax.axhspan(0.5 * EARTH_FIELD_UT, 2.0 * EARTH_FIELD_UT, color=ACCENT, alpha=0.12,
               label="Plausible ambient field")
    ax.axhline(EARTH_FIELD_UT, color=ACCENT, lw=1.4, ls="--")
    colors = [MODEL if d.plausible_magnitude else DATA_GREY for d in diagnostics]
    ax.bar(runs, magnitudes, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(runs)
    ax.set_xlabel("Run")
    ax.set_ylabel("Background |B| (µT, log scale)")
    ax.legend(loc="upper left")
    _titles(ax, "Is the sensor measuring the Earth?",
            "Background magnitude away from the dike, against the local field.")

    ax = axes[1]
    ax.axhspan(0, 20, color=ACCENT, alpha=0.12, label="Stable orientation")
    colors = [MODEL if d.stable_orientation else DATA_GREY for d in diagnostics]
    ax.bar(runs, drifts, color=colors, width=0.62)
    ax.set_xticks(runs)
    ax.set_xlabel("Run")
    ax.set_ylabel("Field direction change, start to end (°)")
    ax.legend(loc="upper right")
    _titles(ax, "Did the phone stay put?",
            "Both ends sample the same ambient field, so this should be near zero.")

    return axes
