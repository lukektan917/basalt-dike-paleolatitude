"""Command-line entry point: run the whole analysis and write every figure.

    python -m dike.run --data data/high_50cm --out figures
    python -m dike.run --synthetic --out figures      # no field data needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from . import align, io, plots  # noqa: E402
from .bootstrap import bootstrap_inclination, fit_ar1  # noqa: E402
from .model import fit_dike  # noqa: E402
from .synthetic import TRUE_PARAMETERS, synthetic_transects  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/high_50cm"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate stand-in transects instead of reading the survey CSVs",
    )
    parser.add_argument("--crop-start", type=float, default=0.15)
    parser.add_argument("--crop-end", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plots.use_house_style()

    if args.synthetic:
        print("Using synthetic transects (no field data required).")
        print(f"  true inclination = {TRUE_PARAMETERS['inclination']:.1f} deg")
        raw = synthetic_transects(seed=args.seed)
        transects = [io.calibrate_distance(df) for df in raw]
    else:
        transects = io.load_transects(args.data)
        print(f"Loaded {len(transects)} transects from {args.data}")

    def save(fig_or_ax, name: str) -> None:
        fig = getattr(fig_or_ax, "figure", fig_or_ax)
        fig.tight_layout()
        fig.savefig(args.out / name, bbox_inches="tight")
        plt.close(fig)

    save(plots.transect_overlay(transects, subtitle="Raw, before alignment."), "01-raw.png")
    save(plots.transect_grid(transects), "02-runs.png")

    aligned, baseline = align.align_transects(
        transects, start=args.crop_start, end=args.crop_end, at="middle", strategy="reference"
    )
    save(
        plots.transect_overlay(
            aligned,
            title="After cropping and alignment",
            subtitle="Each run shifted onto the midpoint of the first.",
            baseline=baseline,
        ),
        "03-aligned.png",
    )

    x, y = align.pool(aligned)
    fit = fit_dike(x, y)
    print("\nFit to pooled transects:")
    print(fit)

    save(plots.fit_figure(aligned, fit), "04-fit.png")
    save(plots.residual_figure(aligned, fit), "05-residuals.png")

    residuals = y - fit.predict(x)
    ar_model = fit_ar1(residuals)
    print(f"\nAR(1) coefficient on pooled residuals: {ar_model.params[1]:.4f}")

    n_grid = int(np.median([len(df) for df in aligned]))
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    result = bootstrap_inclination(
        x_grid, fit.params, ar_model, n_boot=args.n_boot, seed=args.seed
    )
    print("\n" + result.summary())

    save(plots.bootstrap_figure(result), "06-bootstrap.png")
    print(f"\nFigures written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
