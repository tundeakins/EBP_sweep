"""Orbital period determination and primary/secondary eclipse separation."""

import csv
import os

import scipy.signal
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.stats import mode
from transitleastsquares import transitleastsquares
from uncertainties import ufloat

from . import config
from .plotting import plot_all_sectors, save_all_sectors_multipage_pdf
from .utils import find_nearest_window, get_oversampling_factor


def correct_bls_period(periodogram):
    """
    Correct the period of a BLS periodogram if necessary.

    Parameters
    -----------
    periodogram: object
        The BLS periodogram object.

    Returns
    -------
    float
        The corrected period.

    Logic
    -----
    Three statistics from compute_stats() drive the decision:

    depth_half  : depth of a model at P/2.
                  If depth_half ~= depth, the P/2 model fits just as well, meaning
                  BLS found 2x the true period -> halve.

    depth_odd   : depth of a model at 2P, fitting only the "odd" eclipses
                  (at phase 0 of the 2P model).
    depth_even  : depth of a model at 2P, fitting only the "even" eclipses
                  (offset by one fiducial period P).
                  When BLS found P/2 of the true period, odd eclipses = primary
                  and even eclipses = secondary, so they have different depths.
                  If depth_odd != depth_even -> BLS found 1/2 the true period -> double.
                  Note: equal-depth EBs will not trigger this (handled downstream
                  by the phase-fold eclipse count check).
    """

    max_power_period = periodogram.period_at_max_power
    max_power_duration = periodogram.duration_at_max_power
    max_power_transit_time = periodogram.transit_time_at_max_power

    # Compute statistics from the periodogram
    stats = periodogram.compute_stats(max_power_period, max_power_duration, max_power_transit_time)

    depth = stats['depth'][0]
    depth_half = stats['depth_half'][0]
    depth_odd = stats['depth_odd'][0]
    depth_even = stats['depth_even'][0]
    diff_odd_even = abs(ufloat(*stats['depth_odd']) - ufloat(*stats['depth_even']))
    diff_odd_even_significant = diff_odd_even.n / diff_odd_even.s > 3

    # Avoid division by zero for very shallow signals
    depth_scale = max(depth, 0.002)

    # Halve: P/2 model is equally deep -> BLS found 2x the true period
    if abs(depth_half - depth) / depth_scale < 0.01:
        corrected_period = max_power_period / 2

    # Double: alternating eclipses have different depths -> BLS found 1/2 the true period
    # depth_odd = primary depth, depth_even = secondary depth (at 2xP_BLS)
    elif abs(depth_odd - depth_even) / depth_scale > 0.01 and diff_odd_even_significant:
        corrected_period = max_power_period * 2
    else:
        corrected_period = max_power_period  # No correction needed

    return corrected_period


def flatten_tess_by_sector(lc, P, T0):
    """
    Flatten a TESS light curve by sector, masking eclipses.

    Parameters
    -----------
    lc: LightCurve
        The TESS light curve object.
    P: float
        The orbital period of the binary system.
    T0: float
        The time of primary eclipse.

    Returns
    -------
    eclipse_masks: list of array-like
        The eclipse masks for each sector.
    flat_lc: list of LightCurve
        The flattened light curves for each sector.
    trends: list of array-like
        The trends removed from each sector.
    """

    unique_sectors = np.unique(lc.sector)
    flat_lc = []
    eclipse_masks = []
    trends = []

    for sector in unique_sectors:
        sector_lc = lc[lc.sector == sector]

        if len(sector_lc.flux.value) < 3 or np.all(np.isnan(sector_lc.flux.value)) or ((sector_lc.time.value[-1] - sector_lc.time.value[0]) < 0.5 * P):
            continue

        else:
            bls = sector_lc.to_periodogram(method='bls', minimum_period=0.9 * P, maximum_period=1.1 * P)
            prim_eclipse_mask = bls.get_transit_mask(period=P, transit_time=T0, duration=4.5 * bls.duration_at_max_power)

            bls_sec = sector_lc[~prim_eclipse_mask].to_periodogram(method='bls', minimum_period=0.9 * P, maximum_period=1.1 * P)
            sec_eclipse_mask = bls.get_transit_mask(period=P, transit_time=bls_sec.transit_time_at_max_power, duration=5.5 * bls_sec.duration_at_max_power)

            eclipse_mask = prim_eclipse_mask | sec_eclipse_mask

            flat_sector_lc, trend = sector_lc.flatten(window_length=int(1 / np.median(np.diff(sector_lc.time.value))), sigma=1, mask=eclipse_mask, return_trend=True)
            flat_lc.append(flat_sector_lc)
            trends.append(trend)
            eclipse_masks.append(eclipse_mask)

    return eclipse_masks, flat_lc, trends


def get_secondary_period(lc, phased, P_best, T0, o_factor, mf=2):
    """
    Determine the secondary period of a binary system using the phased light curve.

    Parameters
    -----------
    lc: LightCurve
        The TESS light curve object.
    phased: LightCurve
        The phased light curve.
    P_best: float
        The best-fit orbital period.
    T0: float
        The time of primary eclipse.
    o_factor: float
        The oversampling factor.
    mf: float, optional
        Multiplicative factor for eclipse width. Default is 2.

    Returns
    -------
    float
        The secondary period of the binary system.
    """

    min_index = np.nanargmin(phased.flux)
    mid_eclipse_phase = phased.time[min_index]
    mid_eclipse_flux = phased.flux[min_index]

    if (mid_eclipse_phase < -0.45) or (mid_eclipse_phase > 0.45):
        T0 = T0 + 0.1 * P_best
        phased = lc.fold(P_best, T0, normalize_phase=True)

    noneclipse_flux = np.nanmedian(phased.flux)

    half_depth = (noneclipse_flux - mid_eclipse_flux) / 2 + mid_eclipse_flux

    flux_left = phased.flux[min_index::-1]
    ingress_index = min_index - np.nanargmax(flux_left >= half_depth)

    flux_right = phased.flux[min_index:]
    egress_index = min_index + np.nanargmax(flux_right >= half_depth)
    phase_ingress = phased.time[ingress_index]
    phase_egress = phased.time[egress_index]

    eclipse_width = phase_egress - phase_ingress

    eclipse_min = mid_eclipse_phase - (mf * eclipse_width)
    eclipse_max = mid_eclipse_phase + (mf * eclipse_width)

    noneclipse_times = []
    noneclipse_phases = []
    noneclipse_fluxes = []
    noneclipse_flux_errs = []
    noneclipse_sectors = []

    noneclipse_indexes = []

    for i in range(len(phased.time)):

        if (phased.time[i] > eclipse_max) or (phased.time[i] < eclipse_min):

            noneclipse_times.append(phased.time_original[i].value)
            noneclipse_phases.append(phased.time[i].value)
            noneclipse_fluxes.append(phased.flux[i].value)
            noneclipse_flux_errs.append(phased.flux_err[i].value)
            noneclipse_sectors.append(np.array(phased.sector)[i])
            noneclipse_indexes.append(i)

        else:

            noneclipse_times.append(phased.time_original[i].value)
            noneclipse_fluxes.append(np.nan)
            noneclipse_flux_errs.append(np.nan)
            noneclipse_sectors.append(np.array(phased.sector)[i])

    sorted_nonprimary_times = np.array(noneclipse_times)[np.argsort(noneclipse_times)]
    sorted_nonprimary_fluxes = np.array(noneclipse_fluxes)[np.argsort(noneclipse_times)]
    sorted_nonprimary_flux_errs = np.array(noneclipse_flux_errs)[np.argsort(noneclipse_times)]
    sorted_nonprimary_sectors = np.array(noneclipse_sectors)[np.argsort(noneclipse_times)]

    lc_nonprimary = lk.TessLightCurve(time=sorted_nonprimary_times, flux=sorted_nonprimary_fluxes, flux_err=sorted_nonprimary_flux_errs)
    lc_nonprimary['sector'] = sorted_nonprimary_sectors

    if np.all(np.isnan(lc_nonprimary.flux.value)):
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    lc_nonprimary_nonan = lc_nonprimary[~np.isnan(lc_nonprimary.flux)]

    model_secondary = transitleastsquares(lc_nonprimary_nonan.time.value, lc_nonprimary_nonan.flux.value)
    results_secondary = model_secondary.power(period_min=0.99 * P_best, period_max=1.01 * P_best, oversampling_factor=o_factor, show_progress_bar=False)
    P_secondary = results_secondary.period

    phased_secondary = lc_nonprimary.fold(P_secondary, T0, normalize_phase=True)

    if (phased_secondary.time.value[np.nanargmin(phased_secondary.flux.value)] < -0.47) or (phased_secondary.time.value[np.nanargmin(phased_secondary.flux.value)] > 0.47):
        T0 = T0 + 0.1 * P_best
        phased = lc.fold(P_best, T0, normalize_phase=True)
        phased_secondary = lc_nonprimary.fold(P_secondary, T0, normalize_phase=True)
        eclipse_min = eclipse_min - 0.1
        eclipse_max = eclipse_max - 0.1

    return P_secondary, phased_secondary, lc_nonprimary, noneclipse_fluxes, eclipse_min, eclipse_max, results_secondary, phased, T0


def get_primary_period(lc, phased, phased_secondary, P_best, T0, o_factor, mf=2):
    """
    Determine the primary period of a binary system using the phased light curve.

    Parameters
    -----------
    lc: LightCurve
        The TESS light curve object.
    phased: LightCurve
        The phased light curve.
    phased_secondary: LightCurve
        The phased light curve of the secondary eclipse.
    P_best: float
        The best-fit orbital period.
    T0: float
        The time of primary eclipse.
    o_factor: float
        The oversampling factor.
    mf: float, optional
        Multiplicative factor for eclipse width. Default is 2.

    Returns
    -------
    float
        The primary period of the binary system.
    """

    window_size = int(len(phased_secondary) / 20)  # Number of points per window (every 0.05 phase)

    # Compute the global percentiles of all flux values
    global_percentiles = np.argsort(np.argsort(phased_secondary.flux.value)) / len(phased_secondary.flux.value) * 100  # Rank-based percentile

    # Slide the window and compute the average global percentile in each window
    num_windows = len(phased_secondary.flux.value) - window_size + 1
    avg_global_percentiles = np.zeros(num_windows)

    for i in range(num_windows):
        window_percentiles = global_percentiles[i: i + window_size]  # Get percentiles in window
        avg_global_percentiles[i] = np.nanmean(window_percentiles)  # Compute average global percentile

    # Find the window with the lowest average global percentile
    min_index = np.nanargmin(avg_global_percentiles)
    eclipse_phase_range = phased_secondary.time.value[min_index: min_index + window_size]

    # Find the index of the minimum flux within that window
    eclipse_window_flux = phased_secondary.flux.value[min_index: min_index + window_size]
    min_secondary_index = min_index + np.nanargmin(eclipse_window_flux)  # Convert to original index

    # Get the corresponding phase & flux of the minimum flux point
    mid_secondary_eclipse_phase = phased_secondary.time.value[min_secondary_index]
    mid_secondary_eclipse_flux = phased_secondary.flux.value[min_secondary_index]

    if (mid_secondary_eclipse_phase < -0.45) or (mid_secondary_eclipse_phase > 0.45):
        T0 = T0 + 0.1 * P_best
        phased = lc.fold(P_best, T0, normalize_phase=True)

    noneclipse_flux = np.nanmedian(phased.flux)

    half_secondary_depth = (noneclipse_flux - mid_secondary_eclipse_flux) / 2 + mid_secondary_eclipse_flux

    # Define small sliding window size
    small_window_size = int(len(phased_secondary) / 100)  # 0.025 phase

    # Search for ingress and egress windows
    pre_eclipse_indices = np.where(phased_secondary.time.value < mid_secondary_eclipse_phase)[0]  # Before min flux
    post_eclipse_indices = np.where(phased_secondary.time.value > mid_secondary_eclipse_phase)[0]  # After min flux

    # Find ingress and egress phases

    if (len(pre_eclipse_indices) - small_window_size + 1) < 1 or (len(post_eclipse_indices) - small_window_size + 1) < 1:
        return 0, 0, 0, 0, 0, 0, 0, 0

    phase_ingress_secondary = find_nearest_window(phased_secondary.flux.value, phased_secondary.time.value, pre_eclipse_indices, half_secondary_depth, small_window_size)
    phase_egress_secondary = find_nearest_window(phased_secondary.flux.value, phased_secondary.time.value, post_eclipse_indices, half_secondary_depth, small_window_size)

    secondary_eclipse_width = phase_egress_secondary - phase_ingress_secondary

    secondary_eclipse_min = phased_secondary.time.value[np.nanargmin(phased_secondary.flux.value)] - (mf * secondary_eclipse_width)
    secondary_eclipse_max = phased_secondary.time.value[np.nanargmin(phased_secondary.flux.value)] + (mf * secondary_eclipse_width)

    non_secondary_times = []
    non_secondary_fluxes = []
    non_secondary_flux_errs = []

    non_secondary_eclipse_indexes = []

    non_secondary_eclipse_sectors = []

    for i in range(len(phased.time)):

        if (phased.time[i] > secondary_eclipse_max) or (phased.time[i] < secondary_eclipse_min):

            non_secondary_times.append(phased.time_original[i].value)
            non_secondary_fluxes.append(phased.flux[i].value)
            non_secondary_flux_errs.append(phased.flux_err[i].value)
            non_secondary_eclipse_indexes.append(i)
            non_secondary_eclipse_sectors.append(np.array(phased.sector)[i])

        else:

            non_secondary_times.append(phased.time_original[i].value)
            non_secondary_fluxes.append(np.nan)
            non_secondary_flux_errs.append(np.nan)
            non_secondary_eclipse_sectors.append(np.array(phased.sector)[i])

    sorted_nonsecondary_times = np.array(non_secondary_times)[np.argsort(non_secondary_times)]
    sorted_nonsecondary_fluxes = np.array(non_secondary_fluxes)[np.argsort(non_secondary_times)]
    sorted_nonsecondary_flux_errs = np.array(non_secondary_flux_errs)[np.argsort(non_secondary_times)]
    sorted_nonsecondary_sectors = np.array(non_secondary_eclipse_sectors)[np.argsort(non_secondary_times)]

    lc_nonsecondary = lk.TessLightCurve(time=sorted_nonsecondary_times, flux=sorted_nonsecondary_fluxes, flux_err=sorted_nonsecondary_flux_errs)
    lc_nonsecondary['sector'] = sorted_nonsecondary_sectors

    if np.all(np.isnan(lc_nonsecondary.flux.value)):
        return 0, 0, 0, 0, 0, 0, 0, 0

    lc_nonsecondary_nonan = lc_nonsecondary[~np.isnan(lc_nonsecondary.flux)]

    model_primary = transitleastsquares(lc_nonsecondary_nonan.time.value, lc_nonsecondary_nonan.flux.value)

    results_primary = model_primary.power(period_min=0.99 * P_best, period_max=1.01 * P_best, oversampling_factor=o_factor, show_progress_bar=False)
    P_primary = results_primary.period

    phased_primary = lc_nonsecondary.fold(P_primary, T0, normalize_phase=True)

    if (phased_primary.time.value[np.nanargmin(phased_primary.flux.value)] < -0.45) or (phased_primary.time.value[np.nanargmin(phased_primary.flux.value)] > 0.45):
        T0 = T0 + 0.1 * P_best
        phased = lc.fold(P_best, T0, normalize_phase=True)
        phased_primary = lc_nonsecondary.fold(P_primary, T0, normalize_phase=True)
        secondary_eclipse_min = secondary_eclipse_min - 0.1
        secondary_eclipse_max = secondary_eclipse_max - 0.1

    return P_primary, phased_primary, lc_nonsecondary, secondary_eclipse_min, secondary_eclipse_max, results_primary, phased, T0


def find_orbital_period(tic_id, good_lc, flux_threshold=None):
    """Run BLS + TLS to find and refine the orbital period.

    Returns (P_TLS, results_TLS) or (None, None) on failure.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target.
    good_lc : TessLightCurve
        Cleaned light curve of the target.

    flux_threshold : float, optional
        Flux value below which to identify eclipses. If None, a default value of mean(y)-std(y) is used.
    """
    # Rough count of eclipses to set up a range of periods to test on for initial BLS

    comparison_range = int((4 / 24) / mode(np.diff(good_lc.time.value)).mode)  # 4/24 hours: No detached binaries with periods < ~4.6 hours, this variable represents number of points to check btw detected minima to determine if they are eclipses or not. If the minima are too close together, they are likely just noise and not eclipses.
    minima_idx = scipy.signal.argrelextrema(good_lc.flux.value, np.less, order=comparison_range)[0]
    if flux_threshold is None:
        threshold = np.mean(good_lc.flux.value) - (1.25 * np.std(good_lc.flux.value) + 0.01 * np.mean(good_lc.flux.value))  # Deep (more than 1% deep past noise)
    else:
        threshold = flux_threshold

    eclipse_times = good_lc.time.value[[i for i in minima_idx if good_lc.flux.value[i] < threshold]]
    sectors = np.unique(good_lc['sector'].value)

    if len(eclipse_times) < 3:
        print(f"{tic_id}: only {len(eclipse_times)} eclipse(s) detected — flagged for manual inspection")
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None, None

    if len(eclipse_times) <= len(sectors) or len(eclipse_times) / 1.5 <= len(sectors):
        minP = 27 / 4  # The time length of a sector / 4 because there could be some missed eclipses due to cut data
        maxP = min(300, 4 * np.median(np.diff(eclipse_times)), (good_lc.time.value[-1] - good_lc.time.value[0]) / 2)
    else:  # If, on average, there are MORE THAN two eclipses per sector
        minP = max(2.99, 0.5 * np.median(np.diff(eclipse_times)))
        maxP = min(300, 4 * np.median(np.diff(eclipse_times)))

    if minP > maxP:
        return None, None

    pdgrm = good_lc.to_periodogram(method='bls', minimum_period=minP, maximum_period=maxP,
                                    frequency_factor=len(good_lc) / 20)
    P_correct = correct_bls_period(pdgrm).value

    # verify number of eclipses in phase-fold
    phased_test = good_lc.fold(P_correct, float(pdgrm.transit_time_at_max_power.value) - 0.1 * P_correct, normalize_phase=True)
    phased_bin = phased_test.bin(time_bin_size=0.001, aggregate_func=lambda x: np.nanmedian(np.asarray(x).copy()))
    if all(np.isfinite(phased_bin.flux)):
        pcr = int(0.1 / np.median(np.diff(phased_bin.time.value)))  # length of 0.1 phase in points
        e_idx = scipy.signal.argrelextrema(phased_bin.flux.value, np.less, order=pcr, mode='wrap')[0]  # eclipse indices
        n_ecl = sum(1 for i in e_idx if phased_bin.flux.value[i] < 0.98 * np.nanmean(phased_bin.flux.value))
    else:
        pcr = int(0.1 / np.median(np.diff(phased_test.time.value)))  # length of 0.1 phase in points
        e_idx = scipy.signal.argrelextrema(phased_test.flux.value, np.less, order=pcr, mode='wrap')[0]  # eclipse indices
        n_ecl = sum(1 for i in e_idx if phased_test.flux.value[i] < 0.98 * np.nanmean(phased_test.flux.value))

    # there should be exactly 2 eclipses (primary and secondary) in the phase-folded light curve.
    # If there are more or less, adjust the period accordingly.
    if n_ecl > 2:
        P_correct /= 2
    elif n_ecl == 2:
        P_correct = P_correct
    else:  # n_ecl == 1
        P_correct *= 2

    if P_correct < 3:
        # Flag short-period binaries for manual inspection
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(config.data_path(config.SHORTPERIOD_TICIDS_LOG), 'a', newline='') as f:
            csv.writer(f).writerow([tic_id, P_correct])
        return None, None

    o_factor = get_oversampling_factor(good_lc.time.value)
    tls_model = transitleastsquares(good_lc.time.value, good_lc.flux.value)
    results = tls_model.power(period_min=0.99 * P_correct, period_max=1.01 * P_correct,
                               oversampling_factor=o_factor, show_progress_bar=False)

    if np.isnan(results.period):
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None, None

    return results.period, results


def prepare_flat_lc(tic_id, good_lc, P_TLS, results_TLS):
    """Flatten the light curve sector by sector and return (flat_lc, t0, phased, o_factor).

    Also saves a diagnostic phase-fold plot.
    """
    _, flat_test, _ = flatten_tess_by_sector(good_lc, P_TLS, results_TLS.T0)
    plot_all_sectors(tic_id, flat_test, 'detrended')
    flat_lc = flat_test[0].append(flat_test[i + 1] for i in range(len(flat_test) - 1))
    t0 = results_TLS.T0 + 0.2 * P_TLS
    phased = flat_lc.fold(P_TLS, t0, normalize_phase=True)

    fig, ax = plt.subplots(figsize=(15, 7))
    phased.scatter(ax=ax)
    ax.set_title(f"{P_TLS = :.5f} days")
    out_dir = config.fig_path(config.FIG_ALLSECTORS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_Allphased_TLS.jpg')
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    print(f"\tSaved phase-fold plot to {out_path}")
    plt.close()
    save_all_sectors_multipage_pdf(tic_id)

    o_factor = get_oversampling_factor(good_lc.time.value)

    return flat_lc, t0, phased, o_factor


def separate_eclipses(tic_id, flat_lc, phased, P_TLS, t0, o_factor):
    """Separate primary and secondary eclipses using TLS on masked light curves.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target.
    flat_lc : TessLightCurve
        Flattened light curve of the target.
    phased : TessLightCurve
        Phase-folded light curve of the target.
    P_TLS : float
        Orbital period determined by TLS.
    t0 : float
        Reference time for phase-folding.
    o_factor : float
        Oversampling factor for TLS.

    Returns
    -------
    dict:
        A dictionary with all the phased/period quantities, or None on failure.
        The dictionary keys and meaning are:
        * P_primary: Orbital period of the primary eclipse.
        * P_secondary: Orbital period of the secondary eclipse.
        * t0: Reference time for phase-folding.
        * phased: Phase-folded light curve.
        * phased_primary: Phase-folded light curve of the primary eclipse.
        * phased_secondary: Phase-folded light curve of the secondary eclipse.
        * lc_nonsecondary: Light curve with secondary eclipses removed.
        * lc_nonprimary: Light curve with primary eclipses removed.
        * results_primary: TLS results for the primary eclipse.
        * results_secondary: TLS results for the secondary eclipse.
    """
    P_sec, phased_sec, lc_nonpri, _, ecl_min, ecl_max, res_sec, phased, t0 = \
        get_secondary_period(flat_lc, phased, P_TLS, t0, o_factor, mf=3)
    if P_sec == 0:
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None

    P_pri, phased_pri, lc_nonsec, sec_ecl_min, sec_ecl_max, res_pri, phased, t0 = \
        get_primary_period(flat_lc, phased, phased_sec, P_sec, t0, o_factor, mf=3)
    if P_pri == 0:
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None

    return dict(
        P_primary=P_pri, P_secondary=P_sec, t0=t0,
        phased=phased,
        phased_primary=phased_pri, phased_secondary=phased_sec,
        lc_nonsecondary=lc_nonsec, lc_nonprimary=lc_nonpri,
        results_primary=res_pri, results_secondary=res_sec,
        primary_xlims=(ecl_min, ecl_max),
        secondary_xlims=(sec_ecl_min, sec_ecl_max),
        phased_primary_on_secondary=lc_nonsec.fold(P_sec, t0, normalize_phase=True),
        phased_secondary_on_primary=lc_nonpri.fold(P_pri, t0, normalize_phase=True),
    )
