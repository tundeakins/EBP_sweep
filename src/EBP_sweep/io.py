"""Downloading and cleaning input light curves.

Currently supports TESS light curves (via `lightkurve`/QLP) as the primary
input, plus a helper for converting ground-based follow-up photometry
(magnitudes) into relative flux.
"""

import os

import numpy as np
import lightkurve as lk
import pandas as pd
from scipy.stats import mode
from uncertainties import ufloat

from . import config
from .plotting import plot_all_sectors
from .utils import outlier_clipping


def load_and_clean_lc(tic_id, quality_bitmask='hard', mask_outliers=False, bkg_clip_sigma=2, sector_bounds=(None,None)):
    """Download, background-filter, and outlier-clean the light curve.

    Returns good_lc (stitched, cleaned TessLightCurve) or None on failure.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target to download and clean.

    quality_bitmask : str or int, optional
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored
            * "default": cadences with severe quality issues will be ignored
            * "hard": more conservative choice of flags to ignore
                This is known to remove good data.
            * "hardest": removes all data that has been flagged
                This mask is not recommended.
    mask_outliers : bool, optional
        If True, outlier points will be masked from the light curve.
        See All sector plot to confirm that the outlier masking is not removing real eclipses.
    bkg_clip_sigma : float, optional
        Number of standard deviations to use when clipping the background flux.
        Default is 2.
    sector_bounds : tuple, optional
        Tuple specifying the lower and upper bounds of sectors to include.
        Default is (None, None), which includes all sectors.

    Returns
    -------
    lc_final : TessLightCurve or None
        The cleaned and stitched light curve, or None if the download or cleaning failed.

    Notes
    -----
    The function will automatically flag TIC IDs that fail to download or have impossible light curves.
    It also allows filtering of sectors based on the specified sector_bounds.
    

    """
    for attempt in range(3):
        try:
            search_result = lk.search_lightcurve(tic_id, mission='TESS', author="QLP")
            break
        except Exception as e:
            print(f"Data Download:Attempt {attempt+1} failed with error: {e}")
            if attempt == 2:
                config.flag_ticid(tic_id, config.ELEANOR_TICIDS_LOG)
                return None

    if len(search_result.table) == 0:
        config.flag_ticid(tic_id, config.ELEANOR_TICIDS_LOG)
        return None

    print(f"Downloading light curve for {tic_id}...")
    lc_collection = search_result.download_all(quality_bitmask=quality_bitmask)

    # filter sectors based on sector_bounds
    lower_bound, upper_bound = sector_bounds
    lc_collection = lk.LightCurveCollection([
        lc for lc in lc_collection
        if (lower_bound is None or lc.sector >= lower_bound)
        and (upper_bound is None or lc.sector <= upper_bound)
    ])

    # remove outlier points even below median flux
    for i, lc in enumerate(lc_collection):
        lc_collection[i] = lc_collection[i][np.isfinite(lc_collection[i].flux.value)]
        if mask_outliers:
            exp_time = mode(np.diff(lc.time.value)).mode * 24 * 60
            width = 0 if exp_time > 20 else 7 if exp_time > 5 else 15
            _, _, idx = outlier_clipping(lc_collection[i].time.value, lc_collection[i].flux.value,
                                          clip=6, width=width, return_clipped_indices=True, verbose=False)
            print(f"\tMasked {sum(idx)} outliers in sector {lc_collection[i].sector}")
            lc_collection[i].masked_points = idx
        else:
            lc_collection[i].masked_points = np.zeros_like(lc_collection[i].flux.value, dtype=bool)

    plot_all_sectors(tic_id, lc_collection)

    for i in range(len(lc_collection)):
        lc_collection[i] = lc_collection[i][~lc_collection[i].masked_points]
        lc_collection[i]['sector'] = lc_collection[i].sector

    lc = lc_collection.stitch()
    bkg = lc.sap_bkg.value
    if isinstance(bkg, np.ma.MaskedArray):
        bkg = bkg.data

    upper = np.nanmedian(bkg) + bkg_clip_sigma * np.nanstd(bkg)
    lower = np.nanmedian(bkg) - bkg_clip_sigma * np.nanstd(bkg)
    k = np.isfinite(bkg) & (bkg < upper) & (bkg > lower)
    lc_final = lc[k]

    if np.all(np.isnan(lc_final.flux.value)):
        config.flag_ticid(tic_id, config.IMPOSSIBLE_TICIDS_LOG)
        return None

    return lc_final.remove_outliers(sigma_lower=np.inf, sigma_upper=1.5, cenfunc=np.nanmean).remove_nans()


def mag_to_flux(m, merr, nsim=1000):
    """
    Convert magnitude to relative fluxes via Monte Carlo sampling

    Parameters
    ----------
    m : list or np.array
        List or array of magnitudes
    merr : list or np.array
        List or array of errors on the magnitudes (same dimension as `m`).
    nsim : int, optional
        Number of Monte Carlo samples drawn per point. Default is 1000.

    Returns
    -------
    fluxes : np.array
        Relative fluxes
    fluxes_err : np.array
        Error on relative fluxes.
    """

    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))

    for i in range(len(m)):

        dist = 10 ** (-np.random.normal(m[i], merr[i], nsim) / 2.5)

        fluxes[i] = np.mean(dist)

        fluxes_err[i] = np.sqrt(np.var(dist))

    return fluxes, fluxes_err


def read_global_eclipse_params(tic_id):
    """Read back the pooled batman eclipse-shape parameters for a target.

    `EBP_sweep.batman_fit.get_batman_eclipse_times` saves the global
    (pooled-eclipse) fit parameters for each eclipse type to
    ``<FIGURES_DIR>/<FIG_ECLIPSEFIT_DIR>/TIC<id>/TIC<id>_GlobalParams.csv``,
    one row per parameter named ``'{param}_{ecl_type}'`` (e.g. ``'t0_pri'``,
    ``'rp_sec'``). This function reads that file back and reorganises it by
    eclipse type, with the ``'_pri'`` / ``'_sec'`` suffix stripped from each
    parameter name.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target, e.g. ``'TIC 343127696'``.

    Returns
    -------
    tuple of dict
        Two dictionaries, one for the primary eclipse and one for the
        secondary eclipse, each in the format ``{param_name: ufloat}``.
        Parameters that were held fixed in the fit come back with a
        zero uncertainty (``ufloat(value, 0)``), since they weren't varied.

    Raises
    ------
    FileNotFoundError
        If no ``GlobalParams`` CSV has been saved for this target yet — run
        the ``batman`` eclipse-timing method first, e.g. via
        ``EBP_sweep.timing.compute_eclipse_times(..., methods='batman')``.
    """
    csv_path = config.fig_path(config.FIG_ECLIPSEFIT_DIR, f'TIC{tic_id[4:]}', f'TIC{tic_id[4:]}_GlobalParams.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No global eclipse-shape parameters found for {tic_id} at '{csv_path}'. "
            "Run the batman eclipse-timing method first, e.g. via "
            "EBP_sweep.timing.compute_eclipse_times(..., methods='batman')."
        )

    print(f"Reading global eclipse parameters from '{csv_path}'")
    df = pd.read_csv(csv_path, index_col=0)

    params = {}
    for row_label, row in df.iterrows():
        param_name, _, ecl_type = str(row_label).rpartition('_')
        value = row['value'] if pd.notna(row['value']) else None
        # a parameter held fixed (vary=False) in the fit has no stderr; treat its
        # uncertainty as exactly 0 rather than None, which ufloat can't accept
        stderr = row['stderr'] if pd.notna(row['stderr']) else 0
        params.setdefault(ecl_type, {})[param_name] = ufloat(value, stderr)

    return params['pri'], params['sec']
