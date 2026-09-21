"""EBP_sweep: eclipse timing variations and apsidal precession from TESS photometry.

Public API
----------
Pipeline (recommended entry point):
    :class:`EclipsingBinaryTarget`, :func:`run_target`, :func:`run_many`

Individual stages, for step-by-step / notebook-style workflows:
    :mod:`EBP_sweep.io`, :mod:`EBP_sweep.periods`, :mod:`EBP_sweep.timing`,
    :mod:`EBP_sweep.batman_fit`, :mod:`EBP_sweep.plotting`,
    :mod:`EBP_sweep.reporting`

See the ``examples/`` directory in the source repository for full workflows,
and ``notebooks/Follow_up_fitting.ipynb`` for reusing a TESS-derived eclipse
shape to fit ground-based follow-up photometry.
"""

import warnings

# lightkurve and transitleastsquares raise non-actionable UserWarnings on nearly every
# import and light-curve search which clutter pipeline output. Undo
# with `warnings.resetwarnings()` if you'd rather see everything.
warnings.filterwarnings("ignore") 

from .pipeline import EclipsingBinaryTarget, run_target, run_many, load
from .io import load_and_clean_lc, mag_to_flux, read_global_eclipse_params
from .periods import find_orbital_period, prepare_flat_lc, separate_eclipses
from .timing import compute_eclipse_times, compute_oc_and_best_period
from .reporting import create_DV_summary_pdf

__version__ = "0.1.3"

__all__ = [
    "EclipsingBinaryTarget",
    "run_target",
    "run_many",
    "load",
    "load_and_clean_lc",
    "mag_to_flux",
    "read_global_eclipse_params",
    "find_orbital_period",
    "prepare_flat_lc",
    "separate_eclipses",
    "compute_eclipse_times",
    "compute_oc_and_best_period",
    "create_DV_summary_pdf",
]
