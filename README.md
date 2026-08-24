# EBP_sweep

Measure eclipse timing variations (ETVs) and apsidal precession in
eclipsing binaries from TESS photometry, with optional refinement from
ground-based follow-up observations.

Given a TIC ID, the pipeline:

1. Downloads and cleans the TESS light curve (QLP, via `lightkurve`).
2. Finds the orbital period with a BLS + TLS search.
3. Flattens the light curve sector by sector and separates primary and
   secondary eclipses.
4. Measures each eclipse's mid-time with up to five independent methods:
   half-depth crossing, folding symmetry, cross-correlation,
   ingress/egress modelling, and full `batman` transit-model fitting.
5. Builds an O-C diagram, refines the primary/secondary periods, and picks
   the lowest-scatter method per eclipse type.
6. Saves diagnostic plots and a one-page PDF summary per target.

Ground-based follow-up photometry (e.g. a single partially-covered eclipse)
can reuse the TESS-derived eclipse shape to recover a precise mid-eclipse
time from sparse data — see `EBP_sweep.followup` and
`examples/ground_based_followup.py`.

This package was extracted from an exploratory analysis notebook; the
underlying algorithms are unchanged, they're just organised into a proper,
importable, testable package.

## Installation

Some dependencies (`batman-package`, `transitleastsquares`) build C
extensions, so a working C compiler is required. A conda environment is
the easiest way to get everything working:

```bash
conda create -n ebp_sweep python=3.10
conda activate ebp_sweep
pip install -e ".[dev]"
```

Or with plain `pip` in an existing environment:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from EBP_sweep import run_target

target = run_target("TIC 343127696", quality_bitmask="default")
print(target.oc["P_primary_new"], target.oc["P_secondary_new"])
```

For several targets:

```python
from EBP_sweep import run_many

results = run_many(["TIC 343127696", "TIC 81741369"])
```

Or from the command line:

```bash
ebp-sweep run "TIC 343127696" "TIC 81741369" --quality-bitmask default
```

## Tutorial

New to the package (or to eclipse timing in general)? Start with
[`notebooks/tutorial_TIC343127696.ipynb`](notebooks/tutorial_TIC343127696.ipynb) — a guided,
pedagogical walkthrough of the whole pipeline on a real target, with background on apsidal
precession and O&ndash;C diagrams, explanations of each of the five eclipse-timing methods, and
exercises to check your understanding.

For a step-by-step walkthrough with access to every intermediate product
(the cleaned light curve, the measured period, the separated eclipses,
...), see [`examples/single_target.py`](examples/single_target.py) or use
`EclipsingBinaryTarget` directly:

```python
from EBP_sweep import EclipsingBinaryTarget

target = EclipsingBinaryTarget("TIC 343127696")
target.download()
target.find_period()
target.flatten()
target.separate_eclipses()
target.compute_timing()
target.compute_oc()
target.plot_diagnostics()
target.save_summary()
```

See [`examples/`](examples/) for batch runs, a check against literature
periods, and the ground-based follow-up workflow.

## Output

Each run writes diagnostic plots under `./Figures/` and CSV/PDF outputs
under `./Data/`, relative to the current working directory. Targets that
fail at some stage (too few eclipses detected, ambiguous period, no data
available, ...) are logged to a CSV file under `./Data/` instead of raising,
so batch runs can continue past individual failures. Override
`EBP_sweep.config.FIGURES_DIR` / `EBP_sweep.config.DATA_DIR` before running
a pipeline to write elsewhere.

## Package layout

```
src/EBP_sweep/
    io.py          — downloading/cleaning TESS light curves; magnitude-to-flux conversion
    periods.py     — BLS/TLS period search; primary/secondary eclipse separation
    timing.py      — four eclipse-timing methods (half-depth, fold, cross-correlation,
                      ingress/egress) and O-C computation
    batman_fit.py  — eclipse timing via batman transit-model fitting
    followup.py    — fitting ground-based photometry against a TESS-derived eclipse shape
    plotting.py    — diagnostic and summary plots
    reporting.py   — PDF data-validation summaries
    pipeline.py    — EclipsingBinaryTarget, run_target, run_many
    utils.py       — generic numeric helpers (robust statistics, outlier clipping, ...)
    config.py      — output directory/log-file locations
    cli.py         — `ebp-sweep` command-line entry point
notebooks/         — pedagogical tutorial notebook (see "Tutorial" above)
examples/          — runnable scripts covering the workflows above
tests/             — unit tests for the dependency-light, deterministic functions
```

## Testing

```bash
pytest
```

The test suite covers the pure numeric helpers, the `batman` model
wrappers, and magnitude-to-flux conversion — it does not hit the network,
so it runs without downloading any TESS data.

## License

MIT — see [LICENSE](LICENSE).
