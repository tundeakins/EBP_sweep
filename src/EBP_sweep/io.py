"""Downloading and cleaning input light curves.

Supports archived TESS light curves with an optional eleanor FFI fallback,
plus a helper for converting ground-based follow-up photometry
(magnitudes) into relative flux.
"""

import os

import numpy as np
import lightkurve as lk
import pandas as pd
import eleanor
from scipy.stats import mode
from uncertainties import ufloat

from . import config
from .plotting import plot_all_sectors
from .utils import outlier_clipping



def _load_eleanor_lcs(tic_id, sector_bounds):
    """Extract corrected FFI photometry, retaining only eleanor quality == 0."""
    import eleanor  # Optional: archived light curves do not need this dependency.

    tic = int(str(tic_id).upper().removeprefix("TIC").strip())
    lower, upper = sector_bounds
    sources = eleanor.multi_sectors(tic=tic, sectors='all')
    curves = []
    for source in sources:
        if ((lower is not None and source.sector < lower)
                or (upper is not None and source.sector > upper)):
            continue
        try:
            data = eleanor.TargetData(source, do_pca=False, do_psf=False)
            # eleanor's flags include its own diagnostics, so use its recommended
            # zero-quality selection rather than interpreting them as SPOC bits.
            good = ((np.asarray(data.quality) == 0)
                    & np.isfinite(data.time) & np.isfinite(data.corr_flux)
                    & np.isfinite(data.flux_err))
            if not np.any(good):
                raise ValueError("No usable eleanor cadences")
            lc = lk.TessLightCurve(
                time=np.asarray(data.time)[good], time_format="btjd", time_scale="tdb",
                flux=np.asarray(data.corr_flux)[good],
                flux_err=np.asarray(data.flux_err)[good],
                meta={"SECTOR": int(source.sector), "TARGETID": tic,
                      "LABEL": str(tic_id), "AUTHOR": "eleanor"},
            )
            lc["quality"] = np.asarray(data.quality)[good]
            if getattr(data, "flux_bkg", None) is not None:
                lc["sap_bkg"] = np.asarray(data.flux_bkg)[good]
            curves.append(lc)
        except Exception as exc:
            config.log_error(tic_id, f"eleanor sector {source.sector}: {exc}")
            print(f"eleanor sector {source.sector} failed: {exc}")
    return lk.LightCurveCollection(curves)


def load_and_clean_lc(tic_id, quality_bitmask='hard', mask_outliers=False, bkg_clip_sigma=2, sector_bounds=(None,None), eleanor_fallback=True):
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

    eleanor_fallback : bool, optional
        Extract FFI photometry with the optional ``eleanor`` package when no
        QLP or SPOC curves exist within the requested sectors. Default True.
        Eleanor always uses quality == 0; quality_bitmask applies to archive data.

    Returns
    -------
    lc_final : TessLightCurve or None
        The cleaned and stitched light curve, or None if the download or cleaning failed.

    Notes
    -----
    The function will automatically flag TIC IDs that fail to download or have impossible light curves.
    It also allows filtering of sectors based on the specified sector_bounds.
    

    """
    lower_bound, upper_bound = sector_bounds
    lc_collection = lk.LightCurveCollection([])
    for author in ("QLP", "SPOC"):
        for attempt in range(3):
            try:
                search_result = lk.search_lightcurve(tic_id, mission="TESS", author=author)
                break
            except Exception as exc:
                print(f"{author} search: attempt {attempt + 1} failed: {exc}")
                if attempt == 2:
                    # A failed search is not evidence that archive data are absent.
                    config.log_error(tic_id, f"{author} search failed: {exc}")
                    return None
        if len(search_result.table) == 0:
            continue
        sectors = np.asarray(search_result.table["sequence_number"])
        keep = np.ones(len(sectors), dtype=bool)
        if lower_bound is not None:
            keep &= sectors >= lower_bound
        if upper_bound is not None:
            keep &= sectors <= upper_bound
        if not np.any(keep):
            continue
        print(f"Downloading {author} light curve for {tic_id}...")
        try:
            downloaded = search_result[keep].download_all(quality_bitmask=quality_bitmask)
        except Exception as exc:
            config.log_error(tic_id, f"{author} download failed: {exc}")
            print(f"{author} download failed for {tic_id}: {exc}")
            return None
        if downloaded is not None:
            lc_collection = lk.LightCurveCollection([
                lc for lc in downloaded if lc is not None and len(lc) > 0
            ])
        # Failed downloads should be retried, not treated as missing products.
        if not len(lc_collection):
            config.log_error(tic_id, f"{author} download returned no usable light curves")
            return None
        break

    if not len(lc_collection):
        if eleanor_fallback:
            print(f"No QLP/SPOC data in requested sectors for {tic_id}; trying eleanor...")
            try:
                lc_collection = _load_eleanor_lcs(tic_id, sector_bounds)
            except ImportError as exc:
                message = f"eleanor unavailable: {exc}. Install from the EBP_sweep clone with pip install -e '.[eleanor]'."
                print(message)
                config.log_error(tic_id, message)
            except Exception as exc:
                print(f"eleanor extraction failed for {tic_id}: {exc}")
                config.log_error(tic_id, f"eleanor extraction failed: {exc}")
        if not len(lc_collection):
            config.flag_ticid(tic_id, config.ELEANOR_TICIDS_LOG)
            return None

    lc_collection = lk.LightCurveCollection([
        lc[np.isfinite(lc.flux.value)] for lc in lc_collection
        if np.any(np.isfinite(lc.flux.value))
    ])
    if not len(lc_collection):
        config.flag_ticid(tic_id, config.IMPOSSIBLE_TICIDS_LOG)
        return None

    # remove outlier points even below median flux
    for i, lc in enumerate(lc_collection):
        if mask_outliers and len(lc_collection[i]) > 1:
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
    lc_final = lc
    if "sap_bkg" in lc.colnames:
        bkg = np.asarray(np.ma.filled(lc.sap_bkg.value, np.nan))
        finite = np.isfinite(bkg)
        if np.any(finite):
            center, scatter = np.nanmedian(bkg), np.nanstd(bkg)
            # A constant background is valid; strict bounds used to discard it.
            keep = finite if scatter == 0 else finite & (np.abs(bkg - center) < bkg_clip_sigma * scatter)
            lc_final = lc[keep]

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
