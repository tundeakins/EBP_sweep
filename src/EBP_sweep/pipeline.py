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
import os
import pickle
import traceback

from astropy.table import Column
from astropy.utils.masked import Masked

from . import io, periods, plotting, reporting, timing, config


def _strip_unpicklable_masks(lc):
    """Return a copy of a Lightkurve/astropy Table with any `Masked` columns
    (e.g. `MaskedQuantity`) replaced by plain, NaN-filled columns.

    Astropy's `Masked` mixin classes (`astropy.utils.masked.Masked`, which
    backs columns like `MaskedQuantity`) are generated dynamically and can't
    be resolved by reference during pickling — `pickle.dumps` raises
    ``PicklingError: Can't pickle <class '...MaskedQuantity'>`` and `dill`
    fails too, on a separate bug in `MaskedQuantityInfo.__getstate__`.
    Ordinary `astropy.table.MaskedColumn` (numpy.ma-backed) is unaffected
    and is left untouched, keeping its mask exactly.
    """
    if lc is None:
        return None
    lc = lc.copy()
    for name in lc.colnames:
        col = lc[name]
        if isinstance(col, Masked):
            filled = col.filled(np.nan)
            lc[name] = (Column(np.asarray(filled.value), unit=filled.unit)
                        if hasattr(filled, "unit") else Column(np.asarray(filled)))
    return lc


class EclipsingBinaryTarget:
    """Stateful pipeline for measuring eclipse timing variations of one TESS target.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target, e.g. ``'TIC 343127696'``.
    quality_bitmask : str, optional
        Quality bitmask passed to `lightkurve`; see
        :func:`EBP_sweep.io.load_and_clean_lc`. Default ``'hard'``.
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
    verbose : bool, optional
        Whether to print progress messages. Default True. 

    Each pipeline stage is available as its own method (`download`,
    `find_period`, `flatten`, `separate_eclipses`, `compute_timing`,
    `compute_oc`) and stores its output on `self` for inspection; `run()`
    calls them all in order and stops early if any stage fails to find a
    usable signal (the target is logged to one of the CSV files under
    `EBP_sweep.config.DATA_DIR` for manual follow-up).

    `run()` is resumable: on failure it saves the target with every stage
    completed so far to `EBP_sweep.config.RESULTS_DIR`. Reloading it with
    `load(tic_id)` and calling `run()` again skips the already-completed
    stages and continues from the one that failed.
    """

    # Light-curve-valued attributes (on self and inside self.eclipses) that
    # need their Masked columns stripped before pickling; see
    # _strip_unpicklable_masks. TLS result objects (results_TLS and the
    # results_primary/results_secondary inside self.eclipses) are plain
    # numpy-backed namespaces and pickle fine as-is.
    _PICKLE_LC_ATTRS = ('good_lc', 'flat_lc', 'phased')
    _PICKLE_LC_ECLIPSE_KEYS = {
        'phased',
        'phased_primary',
        'phased_secondary',
        'lc_nonsecondary',
        'lc_nonprimary',
        'phased_primary_on_secondary',
        'phased_secondary_on_primary',
    }

    def __init__(self, tic_id, quality_bitmask='hardest', mask_outliers=False, bkg_clip_sigma=2,
                 sector_bounds=(None, None), methods=['hd', 'fold', 'cc', 'gress', 'batman'], 
                 epoch_width=0.2, get_single_best_method=True, verbose=True):
        
        self.tic_id = tic_id
        self.quality_bitmask = quality_bitmask
        self.mask_outliers = mask_outliers
        self.methods = [methods] if isinstance(methods, str) else methods
        for m in self.methods:
            assert m in ['hd', 'fold', 'cc', 'gress', 'batman'], \
                f"Method {m} is not recognized. Must be one of ['hd', 'fold', 'cc', 'gress', 'batman']."
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
        self.get_single_best_method = get_single_best_method
        self.verbose=verbose
        self.bkg_clip_sigma = bkg_clip_sigma
        self.sector_bounds = sector_bounds  # Default sector bounds, can be modified as needed
        self.error_message = None

    def __getstate__(self):
        """Return a pickle-safe state with every attribute preserved.

        Lightkurve/astropy light-curve objects are copied with their
        unpicklable Masked columns stripped (see _strip_unpicklable_masks);
        everything else, including the TLS result objects, is kept as-is.
        """
        state = self.__dict__.copy()
        for attr in self._PICKLE_LC_ATTRS:
            if state.get(attr) is not None:
                state[attr] = _strip_unpicklable_masks(state[attr])

        if state.get('eclipses') is not None:
            eclipses = dict(state['eclipses'])
            for key in self._PICKLE_LC_ECLIPSE_KEYS:
                if key in eclipses and eclipses[key] is not None:
                    eclipses[key] = _strip_unpicklable_masks(eclipses[key])
            state['eclipses'] = eclipses
        return state

    def __setstate__(self, state):
        """Restore a target from a pickled state."""
        self.__dict__.update(state)

    def download(self):
        """Download and clean the light curve. Returns True on success."""
        self.good_lc = io.load_and_clean_lc(self.tic_id, self.quality_bitmask, self.mask_outliers, self.bkg_clip_sigma, self.sector_bounds)
        return self.good_lc is not None

    def find_period(self):
        """Find the orbital period via BLS + TLS. Returns True on success."""
        self.P_TLS, self.results_TLS = periods.find_orbital_period(self.tic_id, self.good_lc, verbose=self.verbose)
        return self.P_TLS is not None

    def flatten(self):
        """Flatten the light curve sector-by-sector and phase-fold it on `P_TLS`."""
        self.flat_lc, self.t0, self.phased, self.o_factor = periods.prepare_flat_lc(
            self.tic_id, self.good_lc, self.P_TLS, self.results_TLS, verbose=self.verbose)
        return self.flat_lc != [] 

    def separate_eclipses(self):
        """Separate primary and secondary eclipses. Returns True on success."""
        self.eclipses = periods.separate_eclipses(
            self.tic_id, self.flat_lc, self.phased, self.P_TLS, self.t0, self.o_factor, verbose=self.verbose)
        return self.eclipses is not None

    def compute_timing(self):
        """Compute eclipse times with all configured methods. Returns True on success."""
        result = timing.compute_eclipse_times(
            self.tic_id, self.eclipses, epoch_width=self.epoch_width, methods=self.methods,verbose=self.verbose)
        if result is None:
            return False
        (self.obs_pri, self.obs_sec, self.err_pri, self.err_sec), self.methods_used = result
        return True

    def compute_oc(self):
        """Compute O-C arrays and refined periods from the eclipse timing results."""
        try:
            self.oc = timing.compute_oc_and_best_period(
                self.tic_id, self.obs_pri, self.obs_sec, self.err_pri, self.err_sec,
                self.eclipses['P_primary'], self.eclipses['P_secondary'], self.methods_used, 
                self.get_single_best_method, verbose=self.verbose)
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error computing O-C: {e}")
            return False

    @property
    def best_primary_method(self):
        return self.methods_used[self.oc['best_index_primary']]

    @property
    def best_secondary_method(self):
        return self.methods_used[self.oc['best_index_secondary']]

    def plot_diagnostics(self):
        """Save the 4-panel eclipse and O-C diagnostic plots to disk."""
        e = self.eclipses
        if 'phased_primary' in e and 'phased_secondary' in e:
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
        self.save()

    def save(self):
        """Save derived results to :data:`EBP_sweep.config.RESULTS_DIR` as a pickle file."""
        os.makedirs(config.results_path(), exist_ok=True)
        with open(config.results_path(f"{self.tic_id}.pkl"), "wb") as f:
            pickle.dump(self, f)

    def _run_stage(self, method_name, fail_msg, done_attr=None):
        """Run one pipeline stage, or skip it if already completed.

        ``done_attr`` names the attribute this stage populates (e.g.
        ``'good_lc'`` for `download`). If it's already truthy -- e.g.
        because this target was `load()`-ed from a save made after an
        earlier failure -- the stage is skipped, so a re-run resumes from
        the first incomplete stage instead of repeating slow steps like
        the download or period search.

        Records either the stage's own failure message or the actual
        exception (with traceback) if it raised one, and saves the target
        immediately on failure so every stage that *did* complete is kept
        even though this run didn't finish.

        Returns True on success (or skip), False on failure.
        """
        if done_attr is not None and bool(getattr(self, done_attr)):
            if self.verbose:
                print(f"Skipping {method_name} for TIC ID {self.tic_id} (already computed)")
            return True
        try:
            ok = getattr(self, method_name)()
        except Exception:
            tb = traceback.format_exc()
            self.error_message = f"{fail_msg}: {tb}"
            print(f"{fail_msg} for TIC ID {self.tic_id}\n{tb}")
            self.save()
            return False
        if not ok:
            self.error_message = fail_msg
            print(f"{fail_msg} for TIC ID {self.tic_id}")
            self.save()
            return False
        return True

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

        Notes
        -----
        Resumable: if a run fails partway through, the target is saved with
        every stage completed so far. Reload it with `load(tic_id)` and call
        `run()` again to pick up from the first incomplete stage instead of
        repeating the (often slow) earlier ones.
        """
        if not self._run_stage("download", "Failed to download data", done_attr="good_lc"):
            return False
        if not self._run_stage("find_period", "Failed to find period", done_attr="P_TLS"):
            return False
        if not self._run_stage("flatten", "Failed to flatten light curve", done_attr="flat_lc"):
            return False
        if not self._run_stage("separate_eclipses", "Failed to separate eclipses", done_attr="eclipses"):
            return False
        if not self._run_stage("compute_timing", "Failed to compute timing", done_attr="obs_pri"):
            return False
        if not self._run_stage("compute_oc", "Failed to compute O-C", done_attr="oc"):
            return False

        if make_plots:
            self.plot_diagnostics()
        if make_summary:
            self.save_summary()

        print(f"P_primary: {self.oc['P_primary_new']}\nP_secondary: {self.oc['P_secondary_new']}")
        self.save()
        return True


def run_target(tic_id, quality_bitmask='hard',**kwargs):
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


def run_many(tic_ids, quality_bitmask='hard', **kwargs):
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

    Notes
    -----
    Every target that fails (whether via a handled stage failure or an
    unhandled exception) gets an ``(id, error, timestamp)`` row appended to
    ``errors.csv`` under :data:`EBP_sweep.config.DATA_DIR`, so the batch can
    be revisited later without re-running everything.
    """
    results = {}
    for tic_id in tic_ids:
        try:
            target = run_target(tic_id, quality_bitmask=quality_bitmask, **kwargs)
            results[tic_id] = target
            if target.oc is None:
                config.log_error(tic_id, target.error_message or "Failed (no error message set)")
        except Exception:
            tb = traceback.format_exc()
            print(f"{tic_id}: unhandled error —\n{tb}")
            config.log_error(tic_id, f"unhandled error: {tb}")
            results[tic_id] = None
    return results


def load(tic_id):
    """Load a previously saved :class:`EclipsingBinaryTarget` from disk.
    Parameters
    ----------
    tic_id : str or list of str
        TIC ID(s) of the target(s) to load.

    Returns
    -------
    EclipsingBinaryTarget or dict
        The loaded target object if a single TIC ID is provided, or a dictionary
        mapping TIC IDs to their corresponding target objects if multiple TIC IDs
        are provided. Returns None for any TIC ID that could not be found on disk.
    """

    results = {}
    if isinstance(tic_id, str):
        tic_id = [tic_id]
        
    for tid in tic_id:
        file_path = config.results_path(f"{tid}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                results[tid] = pickle.load(f)
        else:
            results[tid] = None
    if len(results) == 1:
        return results[tic_id[0]]
    
    return results
