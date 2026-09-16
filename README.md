# Paleolatitude of a basalt dike from a phone magnetometer

Middlesex Fells, Massachusetts. Ten transects walked across a basalt dike with a
phone magnetometer, fitted to a two-dimensional dike model to recover the
magnetisation inclination, with uncertainty from a parametric bootstrap over
AR(1) residuals.

Final project for an Earth science course, December 2025. Covers the upper dike
at the 50 cm transect, where ten repeat runs were recorded — my part of a group
survey that also covered a lower dike and further transects along strike.

![Pooled transects and the fitted dike model](figures/04-fit.png)

## Result

| | |
|---|---|
| Magnetic inclination | 77.9° (95% CI 69.8 – 85.2°) |
| Implied paleolatitude | 66.8° (95% CI 53.6 – 80.4°) |
| Dike half-width | 0.229 m |
| Sensor height above the dike | 0.026 m |
| R² | 0.935 |

Magnitudes, not signed values — see the notes below.

## Method

1. Map elapsed time onto the 0.8 m traverse (each run walked to a metronome at
   120 bpm).
2. Crop to 0.15–0.70 m and shift each run onto a common level, anchored at the
   midpoint of the first run. Runs carry large and differing DC offsets from
   sensor drift.
3. Fit the 2-D dike model to the pooled `Bz`. The balance of the symmetric
   (`sin φ`) and antisymmetric (`cos φ`) parts of the profile fixes the
   inclination.
4. Fit AR(1) to the residuals — they come out autocorrelated at 0.993.
5. Bootstrap: 2000 synthetic profiles from the fitted curve plus simulated AR(1)
   residuals, refit each, take the spread. Then `tan(I) = 2 tan(λ)`.

## Running it

```bash
pip install -r requirements.txt
python -m dike.run --data data/high_50cm
python -m dike.run --synthetic     # simulated data with a known answer
pytest
```

## Contents

- `notebooks/` — the original Colab notebooks, as written
- `src/dike/` — the same analysis as a package, with tests
- `data/high_50cm/` — the ten transect CSVs
- `figures/` — output of `dike.run`

## Notes

**Uncertainty.** `curve_fit` reports ±0.21° on the inclination. The residuals
are autocorrelated at 0.993, so that figure assumes far more independent
observations than the survey collected; the bootstrap puts it nearer ±3°.

**Sign.** A sensor at height *+z* over a dike inclined *−φ* gives exactly the
same profile as one at *−z* over a dike inclined *+φ*. An unconstrained fit
returns a negative sensor height, which reverses the sign of the inclination
while fitting just as well. The fit here bounds the geometry to stay physical,
and the sign is left open: it also depends on the phone's z-axis convention,
which the CSVs do not record.

**Background level.** The dike formula has no constant term, so the fitted
result depends on where the alignment step puts the data. On a traverse long
enough to flatten either side of the dike the background can be fitted instead
(`--fit-baseline`); these transects are not long enough for that, so it is off
by default.

**Remanent vs induced.** An in-situ survey measures the ancient remanence and
the magnetisation induced by today's field together. Separating them needs
oriented samples and lab demagnetisation. The fitted 77.9° is closer to the
present-day field inclination here (~66°) than to published paleolatitudes for
eastern Massachusetts, so the induced part likely dominates.

**Data quality.** `dike.diagnostics` checks whether the background field
magnitude is near the local geomagnetic field and whether the field direction
agrees between the two ends of a traverse. Runs 1–3 read a plausible ambient
field but rotate 90–118° across the traverse; runs 4–10 hold orientation but
read four to nine times the Earth's field. Dropping the glitched run 4 moves the
inclination by 0.3°.

![Data quality checks](figures/00-diagnostics.png)

**Validation.** No independent measurement of this dike exists to check against,
so `--synthetic` generates transects with a known inclination: the pipeline
recovers 67.8° against a true 68.0°.

**Authorship.** The transect-alignment helpers and parts of the bootstrap loop
were drafted with AI assistance and then tested against hand-worked cases, as
noted in the original notebooks. The experimental design, physical model and
interpretation are my own.
