"""High-level orchestration of the per-target eclipse-timing pipeline.

:class:`EclipsingBinaryTarget` walks a single TIC ID through every pipeline
stage (download, period-finding, flattening, eclipse separation, timing,
O-C) while keeping the intermediate products around as attributes, so the
same object supports both a step-by-step, notebook-style workflow and a
one-line :meth:`EclipsingBinaryTarget.run`. :func:`run_target` and
:func:`run_many` are thin convenience wrappers around it for single-target
and batch runs.
"""

import numpy as np

from . import io, periods, plotting, reporting, timing


class EclipsingBinaryTarget:
    """Stateful pipeline for measuring eclipse timing variations of one TESS target.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target, e.g. ``'TIC 343127696'``.
    quality_bitmask : str, optional
        Quality bitmask passed to `lightkurve`; see
        :func:`EBP_sweep.io.load_and_clean_lc`. Default ``'default'``.
    mask_outliers : bool, optional
        Whether to mask outlier points during download-time cleaning.
        Default False.
    methods : sequence of str, optional
        Eclipse-timing methods to run, in canonical order; see
        :func:`EBP_sweep.timing.compute_eclipse_times`.
        Default ``('hd', 'fold', 'cc', 'gress', 'batman')``.
    epoch_width : float, optional
        Width (in units of P) of the per-epoch window used by the batman
        fitting method. Default 0.2.

    Each pipeline stage is available as its own method (`download`,
    `find_period`, `flatten`, `separate_eclipses`, `compute_timing`,
    `compute_oc`) and stores its output on `self` for inspection; `run()`
    calls them all in order and stops early if any stage fails to find a
    usable signal (the target is logged to one of the CSV files under
    `EBP_sweep.config.DATA_DIR` for manual follow-up).
    """

    def __init__(self, tic_id, quality_bitmask='default', mask_outliers=False,
                 methods=('hd', 'fold', 'cc', 'gress', 'batman'), epoch_width=0.2):
        self.tic_id = tic_id
        self.quality_bitmask = quality_bitmask
        self.mask_outliers = mask_outliers
        self.methods = list(methods)
        self.epoch_width = epoch_width

        self.good_lc = None
        self.P_TLS = None
        self.results_TLS = None
        self.flat_lc = None
        self.t0 = None
        self.phased = None
        self.o_factor = None
        self.eclipses = None
        self.methods_used = None
        self.obs_pri = self.obs_sec = self.err_pri = self.err_sec = None
        self.oc = None

    def download(self):
        """Download and clean the light curve. Returns True on success."""
        self.good_lc = io.load_and_clean_lc(self.tic_id, self.quality_bitmask, self.mask_outliers)
        return self.good_lc is not None

    def find_period(self):
        """Find the orbital period via BLS + TLS. Returns True on success."""
        self.P_TLS, self.results_TLS = periods.find_orbital_period(self.tic_id, self.good_lc)
        return self.P_TLS is not None

    def flatten(self):
        """Flatten the light curve sector-by-sector and phase-fold it on `P_TLS`."""
        self.flat_lc, self.t0, self.phased, self.o_factor = periods.prepare_flat_lc(
            self.tic_id, self.good_lc, self.P_TLS, self.results_TLS)

    def separate_eclipses(self):
        """Separate primary and secondary eclipses. Returns True on success."""
        self.eclipses = periods.separate_eclipses(
            self.tic_id, self.flat_lc, self.phased, self.P_TLS, self.t0, self.o_factor)
        return self.eclipses is not None

    def compute_timing(self):
        """Compute eclipse times with all configured methods. Returns True on success."""
        result = timing.compute_eclipse_times(
            self.tic_id, self.eclipses, epoch_width=self.epoch_width, methods=self.methods)
        if result is None:
            return False
        (self.obs_pri, self.obs_sec, self.err_pri, self.err_sec), self.methods_used = result
        return True

    def compute_oc(self):
        """Compute O-C arrays and refined periods from the eclipse timing results."""
        self.oc = timing.compute_oc_and_best_period(
            self.tic_id, self.obs_pri, self.obs_sec, self.err_pri, self.err_sec,
            self.eclipses['P_primary'], self.eclipses['P_secondary'], self.methods_used)
        return self.oc

    @property
    def best_primary_method(self):
        return self.methods_used[self.oc['best_index_primary']]

    @property
    def best_secondary_method(self):
        return self.methods_used[self.oc['best_index_secondary']]

    def plot_diagnostics(self):
        """Save the 4-panel eclipse and O-C diagnostic plots to disk."""
        e = self.eclipses
        plotting.plot_subfigures(
            self.tic_id, e['phased_primary'], e['phased_secondary'],
            e['phased_secondary_on_primary'], e['phased_primary_on_secondary'],
            e['primary_xlims'], e['secondary_xlims'])

        bi_pri, bi_sec = self.oc['best_index_primary'], self.oc['best_index_secondary']
        plotting.plot_best_oc(
            self.tic_id, self.obs_pri[bi_pri], self.obs_sec[bi_sec],
            self.oc['OC_pri_corr'], self.oc['OC_sec_corr'],
            np.array(self.err_pri[bi_pri]) * 24 * 60,
            np.array(self.err_sec[bi_sec]) * 24 * 60)

    def save_summary(self):
        """Write the PDF data-validation summary and append the periods to the CSV log."""
        bi_pri, bi_sec = self.oc['best_index_primary'], self.oc['best_index_secondary']
        reporting.create_DV_summary_pdf(
            self.tic_id, self.oc['P_primary_new'], self.oc['pri_uncerts'][bi_pri],
            self.oc['P_secondary_new'], self.oc['sec_uncerts'][bi_sec],
            self.flat_lc, self.best_primary_method, self.best_secondary_method,
            self.oc['pri_stds'][bi_pri], self.oc['sec_stds'][bi_sec])

    def run(self, make_plots=True, make_summary=True):
        """Run the full pipeline end-to-end.

        Parameters
        ----------
        make_plots : bool, optional
            Save the diagnostic plots (4-panel eclipse plot, O-C plot). Default True.
        make_summary : bool, optional
            Save the PDF data-validation summary. Default True.

        Returns
        -------
        bool
            True if every stage completed and periods were measured, False
            if the target was flagged for manual follow-up at some stage.
        """
        if not self.download():
            return False
        if not self.find_period():
            return False
        self.flatten()
        if not self.separate_eclipses():
            return False
        if not self.compute_timing():
            return False
        self.compute_oc()

        if make_plots:
            self.plot_diagnostics()
        if make_summary:
            self.save_summary()

        print(self.oc['P_primary_new'], self.oc['P_secondary_new'])
        return True


def run_target(tic_id, quality_bitmask='default', **kwargs):
    """Run the full eclipse-timing pipeline for a single TIC ID.

    Convenience wrapper around :class:`EclipsingBinaryTarget`. Additional
    keyword arguments (`mask_outliers`, `methods`, `epoch_width`,
    `make_plots`, `make_summary`) are forwarded appropriately.

    Returns
    -------
    EclipsingBinaryTarget
        The target object with all intermediate products populated,
        regardless of whether the run completed successfully — inspect
        `target.oc` (or the boolean returned by `target.run`) to see how
        far it got.
    """
    run_kwargs = {k: kwargs.pop(k) for k in ('make_plots', 'make_summary') if k in kwargs}
    target = EclipsingBinaryTarget(tic_id, quality_bitmask=quality_bitmask, **kwargs)
    target.run(**run_kwargs)
    return target


def run_many(tic_ids, quality_bitmask='default', **kwargs):
    """Run the full pipeline over several TIC IDs, continuing past individual failures.

    Parameters
    ----------
    tic_ids : list of str
        TIC IDs to process.
    quality_bitmask, **kwargs
        Forwarded to :func:`run_target` for every target.

    Returns
    -------
    dict
        Mapping of TIC ID to its :class:`EclipsingBinaryTarget` (or None if
        an unhandled exception was raised while processing it — the
        traceback is printed to stdout and the run continues with the next
        target).
    """
    results = {}
    for tic_id in tic_ids:
        try:
            results[tic_id] = run_target(tic_id, quality_bitmask=quality_bitmask, **kwargs)
        except Exception as e:
            print(f"{tic_id}: unhandled error — {e}")
            results[tic_id] = None
    return results
