"""Output locations shared across the package.

Every pipeline stage writes diagnostic figures, log files, and saved
pickles to disk as it runs. All of it nests under a single
:data:`OUTPUT_DIR`, so one run of the pipeline over many targets can be
isolated from another by setting just that one path.

Override :data:`OUTPUT_DIR` *before* running a pipeline stage to send every
output (figures, data logs, saved ``.pkl`` files) to its own folder, e.g.::

    from EBP_sweep import config
    config.OUTPUT_DIR = "/path/to/output/2026-09-04_run"

``FIGURES_DIR``, ``DATA_DIR`` and ``RESULTS_DIR`` are the subfolder *names*
created under ``OUTPUT_DIR`` (``Figures``, ``Data``, ``saved_pkl``) —
override one of those instead if you only want to rename or relocate a
single category of output.
"""

import csv
import os
from datetime import datetime, timezone

OUTPUT_DIR = "Result"

FIGURES_DIR = "Figures"
DATA_DIR = "Data"
RESULTS_DIR = "saved_pkl"

# Subdirectories under FIGURES_DIR used by individual plotting routines.
FIG_4PANEL_DIR = "4panels"
FIG_OC_DIR = "OC_plots"
FIG_ALLSECTORS_DIR = "AllSectors"
FIG_EPOCHTIMES_DIR = "EpochTimes"
FIG_VARIATION_DIR = "Variation"
FIG_ECLIPSEFIT_DIR = "EclipseFit"
FIG_DVSUMMARY_DIR = "DV_Summaries"
FIG_METHODCOMP_DIR = "MethodComparison"

# Log files that individual pipeline stages append a TIC ID to when a
# target needs manual follow-up (e.g. too few eclipses, ambiguous period,
# no data available).
MANUAL_TICIDS_LOG = "manual_ticids.csv"
ELEANOR_TICIDS_LOG = "eleanor_ticids.csv"
IMPOSSIBLE_TICIDS_LOG = "impossible_ticids.csv"
SHORTPERIOD_TICIDS_LOG = "shortperiod_ticids.csv"
GOOD_PERIODS_LOG = "good_periods.csv"
OC_TXT_DIR = "OC_txtfiles"

# Log file that batch runs (e.g. run_many) append to whenever a target
# fails, so it can be revisited later without re-running the whole batch.
ERRORS_LOG = "errors.csv"


def output_path(*parts):
    """Build a path under :data:`OUTPUT_DIR`."""
    return os.path.join(OUTPUT_DIR, *parts)


def data_path(*parts):
    """Build a path under ``OUTPUT_DIR/DATA_DIR``, e.g. ``data_path(OC_TXT_DIR, "file.txt")``."""
    return os.path.join(OUTPUT_DIR, DATA_DIR, *parts)


def fig_path(*parts):
    """Build a path under ``OUTPUT_DIR/FIGURES_DIR``, e.g. ``fig_path(FIG_OC_DIR, "file.png")``."""
    return os.path.join(OUTPUT_DIR, FIGURES_DIR, *parts)


def results_path(*parts):
    """Build a path under ``OUTPUT_DIR/RESULTS_DIR``, e.g. ``results_path(f"{tic_id}.pkl")``."""
    return os.path.join(OUTPUT_DIR, RESULTS_DIR, *parts)


def flag_ticid(tic_id, log_name):
    """Append ``tic_id`` as a new row to a log file under ``OUTPUT_DIR/DATA_DIR``.

    Creates the directory if it does not already exist. Used throughout
    the pipeline to record targets that failed a given stage and need
    manual inspection, e.g. ``flag_ticid(tic_id, MANUAL_TICIDS_LOG)``.
    """
    os.makedirs(data_path(), exist_ok=True)
    with open(data_path(log_name), "a", newline="") as f:
        csv.writer(f).writerow([tic_id])


def log_error(tic_id, message, log_name=ERRORS_LOG):
    """Append a ``(tic_id, message, timestamp)`` row to an error log under ``OUTPUT_DIR/DATA_DIR``.

    Writes a header row the first time the file is created. Used by batch
    runs (e.g. :func:`EBP_sweep.pipeline.run_many`) to keep a durable record
    of which targets failed and why, so they can be revisited later without
    re-running the whole batch.
    """
    os.makedirs(data_path(), exist_ok=True)
    path = data_path(log_name)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["tic_id", "error", "timestamp"])
        writer.writerow([tic_id, message, datetime.now(timezone.utc).isoformat(timespec="seconds")])
