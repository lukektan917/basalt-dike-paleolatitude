# Paleolatitude of a basalt dike from a phone magnetometer

**Where on Earth was Massachusetts when this rock cooled?**

A basalt dike in the Middlesex Fells, north of Boston, was intruded as molten
rock and then froze. As it cooled through the Curie temperature its magnetic
minerals locked in the direction of the Earth's field at that moment. That
direction is still there, and the steepness of it — the *inclination* — is a
record of latitude: the field is horizontal at the equator and vertical at the
poles.

This repository recovers that inclination from ten transects walked across the
dike with a phone magnetometer, and converts it into a paleolatitude with an
uncertainty that takes the structure of the measurement error seriously.

![Pooled transects and the fitted dike model](figures/04-fit.png)

## The measurement

Ten traverses, each 0.8 m long, perpendicular to the strike of the dike, walked
to a metronome so that time maps linearly onto distance. The phone records the
three components of the field at 100 Hz; only the vertical component `Bz` is
used here.

Ten runs rather than one because a single pass over a rock with a hand-held
sensor is not a measurement — it is an anecdote. The repeats are what make it
possible to say anything about uncertainty at all.

## The method

**1. Time to distance.** Elapsed time is mapped linearly onto the 0.8 m
transect. (`dike.io`)

**2. Crop and align.** The ends of each run are contaminated by starting and
stopping, so they are trimmed. Each run also carries its own unknown DC offset —
the ambient regional field plus the phone's own hard-iron contribution — so the
runs are shifted onto a common level, anchored at the midpoint, where the
metronome beat makes position best known across runs. (`dike.align`)

**3. Fit the dike model.** The vertical anomaly over a two-dimensional dike has
a closed form with two parts: a symmetric pair of arctangent terms scaled by
`sin(inclination)`, and an antisymmetric logarithmic term scaled by
`cos(inclination)`. The balance between the symmetric and antisymmetric shape of
the observed profile is what fixes the inclination. (`dike.model`)

**4. Model the residuals.** The residuals from the fit are not noise — they are
smooth, wandering excursions, because consecutive samples are a hundredth of a
second and a few millimetres apart. An AR(1) model fitted to the pooled
residuals returns a coefficient of about 0.99. (`dike.bootstrap`)

![Residuals](figures/05-residuals.png)

**5. Bootstrap the uncertainty.** With correlation that strong, the effective
sample size is a small fraction of the nominal one. `curve_fit` reports a
standard error of 0.14° on the inclination; the bootstrap below puts it nearer
3°, so the naive number is optimistic by a factor of roughly forty. Instead,
2000 synthetic profiles are generated from the fitted model plus simulated AR(1)
residuals, each is refitted, and the spread of the refitted inclinations is the
uncertainty. That interval is then carried through the dipole equation
`tan(I) = 2 tan(λ)` to a paleolatitude.

![Bootstrap distribution](figures/06-bootstrap.png)

## Results

> **Note.** The survey CSVs are not committed to this repository yet. The
> figures and numbers above come from `--synthetic`: simulated transects with a
> known answer, used to validate the pipeline. On that data the analysis
> recovers an inclination of **67.8°** against a true value of **68.0°**, and a
> half-width of **0.170 m** against a true **0.170 m**, with a 95% interval of
> 61.6–72.7° that brackets the truth. Re-run with `--data` once the CSVs are in
> place to replace this section with the field result.

## Reproducing it

```bash
git clone https://github.com/lukektan917/basalt-dike-paleolatitude
cd basalt-dike-paleolatitude
pip install -r requirements.txt

python -m dike.run --synthetic --out figures     # no field data required
python -m dike.run --data data/high_50cm         # with the survey CSVs
pytest                                           # 37 tests
```

The notebook in `notebooks/` walks through the same analysis with commentary.

## What this repository does differently from the original notebook

This started as a course project written in Colab. Three things changed in the
move, and they changed the answer, not just the tidiness:

**A baseline term is now fitted.** The dike formula has no constant term, so
handing it data at the wrong DC level does not produce a fit that is uniformly
offset — it produces a fit that has tilted the balance between the symmetric and
antisymmetric terms, which is to say it has biased the inclination. On synthetic
data with a known answer, fitting without a baseline recovers 61.7° where the
truth is 68.0°. Fitting with one recovers 67.8°. That is a six-degree error in
the one number the project exists to measure, and it was invisible without a
ground truth to check against.

**The fit is bounded and multi-start.** Negating the half-width, the depth, or
the intensity can each be absorbed by moving the inclination 180°, so the
parameterisation is degenerate and an unconstrained fit can settle in a mirror
solution that reports a metre-wide dike and a meaningless angle while still
showing a respectable R². The original avoided this with a hand-tuned starting
guess. Bounds that keep the geometry physical, plus several starting
inclinations, make it robust instead of lucky.

**The inclination is converted to a paleolatitude.** The original stopped at the
fitted angle. The angle is not the answer to the question the project asks; the
latitude is.

The distance calibration was also rewritten to normalise by observed elapsed
time rather than by a nominal sample count, which gives the same answer for
these files and the right answer for any others.

## Layout

```
src/dike/
  io.py          loading and time → distance calibration
  align.py       cropping and vertical alignment of repeated runs
  model.py       the forward model, the bounded fit, the dipole equation
  bootstrap.py   AR(1) residual model and the parametric bootstrap
  plots.py       figures
  synthetic.py   simulated transects with a known answer
  run.py         CLI: regenerate every figure
tests/           37 tests, including end-to-end parameter recovery
notebooks/       narrative walkthrough
```

## Caveats

* **One dike records an instant, not an average.** The dipole equation assumes
  the field, averaged over enough time, is that of a geocentric axial dipole.
  A single intrusion does not average out secular variation, so the latitude
  here is a point estimate from a single sample of the field, not a
  paleomagnetic pole.
* **The bootstrap covers fit uncertainty, not every error.** It propagates the
  residual process through the fit. It does not capture error in the distance
  calibration, in the assumption that the dike is two-dimensional and
  vertically sided, or in the alignment step.
* **A phone magnetometer is not a survey instrument.** Its absolute calibration
  is unknown, which is why intensity is reported in raw units and only the
  *direction* is interpreted.

## Notes on authorship

The transect-alignment helpers and parts of the bootstrap loop were drafted with
AI assistance and then checked against hand-worked cases. The experimental
design, the physical model, the diagnosis of the baseline bias, and the
interpretation are my own.

Field data collected September 2025. Analysis originally submitted as a final
project for an Earth science course at Harvard, December 2025.
