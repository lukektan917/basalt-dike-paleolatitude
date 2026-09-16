"""Figures for the dike analysis.

The ten transects are repeated measurements of one thing, not ten categories,
so they are drawn as one grey family with the fitted curve over them rather
than ten colours and a ten-entry legend. :func:`transect_grid` gives small
multiples where a per-run comparison is wanted.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import bz_dike, paleolatitude_from_inclination

__all__ = [
    "DATA_GREY",
    "MODEL",
    "ACCENT",
    "use_house_style",
    "transect_overlay",
    "transect_grid",
    "fit_figure",
    "residual_figure",
    "bootstrap_figure",
]

#: Recessive grey for replicate data.
DATA_GREY = "#8A8A85"
#: The one colour that carries meaning: the fitted model.
MODEL = "#1A1A18"
#: Secondary accent, used only for intervals and reference marks.
ACCENT = "#2F6FED"

_INK = "#1A1A18"
_MUTED = "#6B6B66"
_GRID = "#E3E3DE"


def use_house_style() -> None:
    """Apply a quiet, consistent style: recessive axes, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": _GRID,
            "axes.labelcolor": _MUTED,
            "axes.titlecolor": _INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.6,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 130,
        }
    )


def _titles(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional subtitle beneath it.

    The title is padded clear of the axes so the subtitle has room; writing
    both at the same offset is how they end up printed on top of each other.
    """
    ax.set_title(title, loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9, color=_MUTED, va="bottom"
        )


def _axis_labels(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("B$_z$ (µT)")
    _titles(ax, title, subtitle)


def transect_overlay(
    frames: Sequence[pd.DataFrame],
    title: str = "Ten transects over the dike",
    subtitle: str | None = None,
    baseline: float | None = None,
    ax=None,
):
    """All transects as one grey family. No legend: they are replicates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df in frames:
        ax.plot(df["distance"], df["Bz"], color=DATA_GREY, lw=1.0, alpha=0.75)

    if baseline is not None:
        ax.axhline(baseline, color=ACCENT, lw=1.5, ls="--", label="Common baseline")
        ax.legend(loc="best")

    _axis_labels(ax, title, subtitle)
    return ax


def transect_grid(
    frames: Sequence[pd.DataFrame],
    ncols: int = 5,
    title: str = "Transects individually",
):
    """Small multiples — the readable way to compare ten runs."""
    n = len(frames)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for i, df in enumerate(frames):
        ax = axes[i]
        ax.plot(df["distance"], df["Bz"], color=MODEL, lw=1.0)
        ax.set_title(f"Run {i + 1}", loc="left", fontsize=9)
        ax.grid(True)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="semibold")
    fig.supxlabel("Distance along transect (m)", fontsize=9, color=_MUTED)
    fig.supylabel("B$_z$ (µT)", fontsize=9, color=_MUTED)
    fig.tight_layout()
    return fig


def fit_figure(frames: Sequence[pd.DataFrame], fit, ax=None):
    """Aligned data in grey with the fitted dike model through it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df in frames:
        ax.plot(df["distance"], df["Bz"], color=DATA_GREY, lw=1.0, alpha=0.7)

    x = np.linspace(
        min(df["distance"].min() for df in frames),
        max(df["distance"].max() for df in frames),
        600,
    )
    ax.plot(x, fit.predict(x), color=MODEL, lw=2.2, label="Fitted dike model")

    ax.annotate(
        f"inclination {fit.inclination:.1f}°\nhalf-width {fit.half_width:.3f} m\nR² {fit.r2:.3f}",
        xy=(0.015, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color=_MUTED,
    )
    ax.legend(loc="lower left")
    _axis_labels(
        ax,
        "Pooled transects and the best-fit dike model",
        "Grey: ten aligned runs. Black: the two-dimensional dike model.",
    )
    return ax


def residual_figure(frames: Sequence[pd.DataFrame], fit, ax=None):
    """Residuals from the fit — the input to the AR(1) noise model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6))

    for df in frames:
        x = df["distance"].to_numpy(dtype=float)
        ax.plot(x, df["Bz"].to_numpy(dtype=float) - fit.predict(x), color=DATA_GREY, lw=0.9, alpha=0.7)

    ax.axhline(0, color=MODEL, lw=1.2, ls="--")
    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("Residual (µT)")
    _titles(
        ax,
        "Residuals are smooth, not noise",
        "Long excursions from zero are why i.i.d. standard errors would mislead.",
    )
    return ax


def bootstrap_figure(result, level: float = 95.0, ax=None):
    """Bootstrap distribution of paleolatitude with its percentile interval."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))

    values = result.paleolatitude
    lo, hi = result.interval(values, level)
    observed = paleolatitude_from_inclination(float(result.observed[3]))

    ax.hist(values, bins=40, color=DATA_GREY, edgecolor="white", linewidth=0.6)
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.12, label=f"{level:g}% interval")
    ax.axvline(observed, color=MODEL, lw=2.0, label="Estimate")

    ax.margins(y=0.18)
    ax.annotate(
        f"{observed:.1f}°   {level:g}% CI {lo:.1f}–{hi:.1f}°",
        xy=(0.015, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        color=_INK,
    )
    ax.legend(loc="upper right")

    ax.set_xlabel("Paleolatitude (degrees)")
    ax.set_ylabel("Bootstrap replicates")
    _titles(
        ax,
        "Where the dike cooled",
        f"{result.converged} parametric bootstrap replicates under an AR(1) residual model.",
    )
    return ax
