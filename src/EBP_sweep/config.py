"""Output locations shared across the package.

Every pipeline stage writes diagnostic figures and log files to disk as it
runs (this mirrors the original notebook, which wrote to the same relative
paths). The paths are collected here so they can be inspected or overridden
in one place instead of being repeated as string literals in every module.

Override the constants *before* running a pipeline stage if you want output
written somewhere other than ``./Figures`` and ``./Data`` relative to the
current working directory, e.g.::

    from EBP_sweep import config
    config.FIGURES_DIR = "/path/to/output/Figures"
    config.DATA_DIR = "/path/to/output/Data"
"""

import csv
import os

FIGURES_DIR = "Figures"
DATA_DIR = "Data"

# Subdirectories under FIGURES_DIR used by individual plotting routines.
FIG_4PANEL_DIR = "4panels"
FIG_OC_DIR = "OC_plots"
FIG_ALLSECTORS_DIR = "AllSectors"
FIG_EPOCHTIMES_DIR = "EpochTimes"
FIG_VARIATION_DIR = "Variation"
FIG_ECLIPSEFIT_DIR = "EclipseFit"
FIG_DVSUMMARY_DIR = "DV_Summaries"

# Log files that individual pipeline stages append a TIC ID to when a
# target needs manual follow-up (e.g. too few eclipses, ambiguous period,
# no data available).
MANUAL_TICIDS_LOG = "manual_ticids.csv"
ELEANOR_TICIDS_LOG = "eleanor_ticids.csv"
IMPOSSIBLE_TICIDS_LOG = "impossible_ticids.csv"
SHORTPERIOD_TICIDS_LOG = "shortperiod_ticids.csv"
GOOD_PERIODS_LOG = "good_periods.csv"
OC_TXT_DIR = "OC_txtfiles"


def data_path(*parts):
    """Build a path under :data:`DATA_DIR`, e.g. ``data_path(OC_TXT_DIR, "file.txt")``."""
    return os.path.join(DATA_DIR, *parts)


def fig_path(*parts):
    """Build a path under :data:`FIGURES_DIR`, e.g. ``fig_path(FIG_OC_DIR, "file.png")``."""
    return os.path.join(FIGURES_DIR, *parts)


def flag_ticid(tic_id, log_name):
    """Append ``tic_id`` as a new row to a log file under :data:`DATA_DIR`.

    Creates :data:`DATA_DIR` if it does not already exist. Used throughout
    the pipeline to record targets that failed a given stage and need
    manual inspection, e.g. ``flag_ticid(tic_id, MANUAL_TICIDS_LOG)``.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(data_path(log_name), "a", newline="") as f:
        csv.writer(f).writerow([tic_id])
