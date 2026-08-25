"""EBP_sweep: eclipse timing variations and apsidal precession from TESS photometry.

Public API
----------
Pipeline (recommended entry point):
    :class:`EclipsingBinaryTarget`, :func:`run_target`, :func:`run_many`

Individual stages, for step-by-step / notebook-style workflows:
    :mod:`EBP_sweep.io`, :mod:`EBP_sweep.periods`, :mod:`EBP_sweep.timing`,
    :mod:`EBP_sweep.batman_fit`, :mod:`EBP_sweep.plotting`,
    :mod:`EBP_sweep.reporting`, :mod:`EBP_sweep.followup`

See the ``examples/`` directory in the source repository for full workflows.
"""

from .pipeline import EclipsingBinaryTarget, run_target, run_many
from .io import load_and_clean_lc, mag_to_flux, read_global_eclipse_params
from .periods import find_orbital_period, prepare_flat_lc, separate_eclipses
from .timing import compute_eclipse_times, compute_oc_and_best_period
from .reporting import create_DV_summary_pdf

__version__ = "0.1.0"

__all__ = [
    "EclipsingBinaryTarget",
    "run_target",
    "run_many",
    "load_and_clean_lc",
    "mag_to_flux",
    "read_global_eclipse_params",
    "find_orbital_period",
    "prepare_flat_lc",
    "separate_eclipses",
    "compute_eclipse_times",
    "compute_oc_and_best_period",
    "load_ground_based_photometry",
    "run_followup_fit",
    "create_DV_summary_pdf",
]
