"""Eclipse mid-time estimation (four classical methods + batman) and O-C analysis.

Four lightweight methods are implemented directly here:

* ``hd``    — half-depth crossing time on either side of the eclipse.
* ``fold``  — best-fit symmetry point when folding ingress onto egress.
* ``cc``    — cross-correlation of each eclipse against a pooled spline model.
* ``gress`` — same idea as ``cc`` but fit separately to ingress and egress.

A fifth, more precise method (``batman``, full transit-model fitting) lives
in :mod:`EBP_sweep.batman_fit` and is invoked from :func:`compute_eclipse_times`
below alongside these four.
"""

import os

import numpy as np
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d, LSQUnivariateSpline
from scipy.stats import mode

from . import config
from .batman_fit import get_batman_eclipse_times
from .plotting import save_epoch_fits_pdf, plot_shape_variation
from .utils import estimate_time_uncertainty, find_nearest, get_lc_noise_level, select_best_index


def get_t0_err_shift(local_times, local_fluxes, local_fluxerrs, T_pred, P, spline, shift_limit, best_shift, ingress_shift, gress=False):
    """
    Estimate the timing error for a given eclipse shift.

    Parameters
    -----------
    local_times: array-like
        The local times of the observations.
    local_fluxes: array-like
        The local fluxes of the observations.
    local_fluxerrs: array-like
        The errors in the local fluxes.
    T_pred: float
        The predicted time of the eclipse.
    P: float
        The orbital period.
    spline: callable
        The spline model of the eclipse.
    shift_limit: float
        The maximum allowed shift.
    best_shift: float
        The best-fit shift.
    ingress_shift: float
        The ingress shift.
    gress: bool, optional
        Whether to consider the egress. Default is False.

    Returns
    -------
    float
        The estimated timing error.
    """

    def shift_residuals(params, time, flux, flux_err, T_pred, P, spline, shift_limit, ingress_shift, gress=False):

        shift = params['shift'].value

        if gress is False:
            shift_time = time - T_pred + (0.1 * P) - shift
        else:
            shift_time = time - T_pred - shift + ingress_shift

        mask = (shift_time > shift_time[0] + shift_limit) & (shift_time < shift_time[-1] - shift_limit)

        shifted_time = shift_time[mask]

        model_flux = spline(shifted_time)

        data_flux = flux[mask]
        data_err = flux_err[mask]

        resid = (data_flux - model_flux) / data_err

        return resid

    params = Parameters()
    params.add('shift', value=best_shift, min=-shift_limit, max=shift_limit)

    minner = Minimizer(shift_residuals, params, fcn_args=(local_times, local_fluxes, local_fluxerrs, T_pred, P, spline, shift_limit, ingress_shift))

    result = minner.minimize(method='leastsq')
    best_t0_err = result.params['shift'].stderr

    return best_t0_err


def fold_residuals(params, time, flux, fluxerr_value):
    """
    Fold the flux around the predicted mid-eclipse time and compute the residuals.

    Parameters
    ----------
    params: lmfit.Parameters
        The parameters containing the predicted mid-eclipse time 't0'.
    time: array-like
        The observation times.
    flux: array-like
        The observed fluxes.
    fluxerr_value: float
        The flux error value.

    Returns
    -------
    array-like
        The folded residuals.
    """

    t0 = params['t0'].value

    delta_t = time - t0

    mask_pos = delta_t >= 0
    mask_neg = delta_t <= 0

    t_pos = delta_t[mask_pos]
    t_neg = -delta_t[mask_neg]
    f_pos = flux[mask_pos]
    f_neg = flux[mask_neg]

    interp_neg = interp1d(t_neg, f_neg, bounds_error=False, fill_value='extrapolate')
    interp_pos = interp1d(t_pos, f_pos, bounds_error=False, fill_value='extrapolate')

    t_grid = np.linspace(0, min(t_neg.max(), t_pos.max()), 100)

    f_neg_interp = interp_neg(t_grid)
    f_pos_interp = interp_pos(t_grid)

    data_err = np.ones_like(f_pos_interp) * fluxerr_value
    resid = (f_pos_interp - f_neg_interp) / data_err

    return resid


def get_eclipse_times(tic_id, phased_lc, eclipse_lc, P, pdgrm_results, ecl_type,
                       methods=['hd', 'fold', 'cc', 'gress']):
    """
    Estimate eclipse times using various methods.

    Parameters
    ----------
    tic_id: int
        The TIC ID of the target.
    phased_lc: LightCurve
        The phased light curve.
    eclipse_lc: LightCurve
        The eclipse light curve.
    P: float
        The orbital period.
    pdgrm_results: object
        The periodogram results containing predicted transit times.
    ecl_type: str
        The type of eclipse ('pri' or 'sec').
    methods: list of str, optional
        The methods to use for estimating eclipse times. Default is ['hd', 'fold', 'cc', 'gress'].

    Returns
    -------
    tuple
        Four lists containing the observed eclipse times using different methods.
    """

    observed_eclipse_times_halfdepth = []
    observed_eclipse_times_folding = []
    observed_eclipse_times_cc = []
    observed_eclipse_times_gress = []

    observed_eclipse_time_errs_hd = []
    observed_eclipse_time_errs_fold = []
    observed_eclipse_time_errs_cc = []
    observed_eclipse_time_errs_gress = []

    local_times_arrays = []
    local_fluxes_arrays = []
    local_fluxerrs_arrays = []

    if np.all(np.isnan(pdgrm_results.transit_times)):
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return 0, 0, 0, 0
    epoch_fits_hd = []
    valid_T_preds = []

    if 'hd' in methods or 'fold' in methods:
        print(f"Getting {ecl_type} eclipse times using half-depth and folding methods ...")
        for T_pred in pdgrm_results.transit_times:
            mask = abs(eclipse_lc.time.value - T_pred) <= 0.1 * P
            local_exptime = mode(np.diff(eclipse_lc.time.value[mask])).mode
            local_expected_npts = 0.2 * P / local_exptime
            mask_left = eclipse_lc.time.value[mask] < T_pred
            mask_right = eclipse_lc.time.value[mask] > T_pred

            ingress_coverage = np.sum(mask_left) / local_expected_npts
            egress_coverage = np.sum(mask_right) / local_expected_npts

            if (np.sum(mask_left) > 1) & (np.sum(mask_right) > 1) & (ingress_coverage >= 0.1) & (egress_coverage >= 0.1):  # Make sure there's enough data to work with
                local_times = eclipse_lc.time.value[mask]
                local_fluxes = eclipse_lc.flux.value[mask]
                local_fluxerrs = eclipse_lc.flux_err.value[mask]

                interp_model = interp1d(local_times, local_fluxes, bounds_error=False, fill_value='extrapolate')
                new_times = np.linspace(local_times[0], local_times[-1], 10000)
                new_fluxes = interp_model(new_times)
                valid_T_preds.append(T_pred)

                # Half Depth Method
                noneclipse_flux = np.nanmedian(phased_lc.flux)
                half_depth = (noneclipse_flux + np.nanmin(new_fluxes)) / 2

                if np.all(np.isnan(new_fluxes[np.where(new_times < T_pred)])) or np.all(np.isnan(new_fluxes[np.where(new_times > T_pred)])):
                    continue

                ingress_fluxes = new_fluxes[np.where(new_times < T_pred)]
                ingress_times = new_times[np.where(new_times < T_pred)]

                half_index_ingress, half_time_ingress, half_flux_ingress = find_nearest(ingress_times, ingress_fluxes, half_depth)

                egress_fluxes = new_fluxes[np.where(new_times > T_pred)]
                egress_times = new_times[np.where(new_times > T_pred)]

                half_index_egress, half_time_egress, half_flux_egress = find_nearest(egress_times, egress_fluxes, half_depth)

                mid_eclipse = (half_time_ingress + half_time_egress) / 2
                observed_eclipse_times_halfdepth.append(mid_eclipse)

                # Get HD Uncert

                try:
                    ingress_flux_err = np.ones_like(ingress_fluxes) * get_lc_noise_level(local_fluxes)
                    t_ingress_err = estimate_time_uncertainty(ingress_times, ingress_fluxes, ingress_flux_err, half_depth)

                    egress_flux_err = np.ones_like(egress_fluxes) * get_lc_noise_level(local_fluxes)
                    t_egress_err = estimate_time_uncertainty(egress_times, egress_fluxes, egress_flux_err, half_depth)

                    mid_eclipse_time_err = 0.5 * np.sqrt(t_ingress_err**2 + t_egress_err**2)
                    observed_eclipse_time_errs_hd.append(mid_eclipse_time_err)

                except Exception:
                    mid_eclipse_time_err = (3 / (24 * 60))  # set to 3minutes
                    observed_eclipse_time_errs_hd.append(mid_eclipse_time_err)

                epoch_fits_hd.append((local_times, local_fluxes, new_times, new_fluxes, mid_eclipse, mid_eclipse_time_err))

                # Folding Method
                shift_limit = 0.05
                t_min, t_max = T_pred - shift_limit, T_pred + shift_limit
                num_trials = 1000
                candidate_midpoints = np.linspace(t_min, t_max, num_trials)

                best_error = np.inf

                for t0 in candidate_midpoints:

                    delta_t = new_times - t0

                    mask_pos = delta_t >= 0
                    mask_neg = delta_t <= 0

                    t_pos = delta_t[mask_pos]
                    t_neg = -delta_t[mask_neg]  # flip to positive
                    f_pos = new_fluxes[mask_pos]
                    f_neg = new_fluxes[mask_neg]

                    # Interpolate the negative side onto positive side
                    if len(t_neg) > 0:
                        interp_model = interp1d(t_neg, f_neg, bounds_error=False, fill_value='extrapolate')
                        f_neg_interp = interp_model(t_pos)
                    else:
                        continue

                    error = np.nanmean((f_pos - f_neg_interp)**2)

                    if error < best_error:
                        best_error = error
                        best_midpoint = t0

                observed_eclipse_times_folding.append(best_midpoint)

                # Get Folding Uncert
                try:

                    params = Parameters()
                    params.add('t0', value=best_midpoint, min=T_pred - shift_limit, max=T_pred + shift_limit)

                    minner = Minimizer(fold_residuals, params, fcn_args=(new_times, new_fluxes, get_lc_noise_level(local_fluxes)))

                    result = minner.minimize(method='leastsq')
                    best_t0_err = result.params['t0'].stderr

                    observed_eclipse_time_errs_fold.append(best_t0_err)

                except Exception:
                    observed_eclipse_time_errs_fold.append((3 / (24 * 60)))  # set to 3minutes

                local_times_arrays.append(local_times)
                local_fluxes_arrays.append(local_fluxes)
                local_fluxerrs_arrays.append(local_fluxerrs)

        print(f"\tSaving {len(valid_T_preds)} {ecl_type} eclipse times for half-depth and folding methods to pdf ...")
        save_epoch_fits_pdf(tic_id, P, epoch_fits_hd, ecl_type, 'half_depth',
                             extra_t0s=dict(folding=[observed_eclipse_times_folding, observed_eclipse_time_errs_fold]))

        eclipse_lc_time = np.concatenate((local_times_arrays))
        eclipse_lc_flux = np.concatenate((local_fluxes_arrays))
        eclipse_lc_flux_err = np.concatenate((local_fluxerrs_arrays))

    # Cross Correlating Method
    if 'cc' in methods or 'gress' in methods:
        print(f"Getting {ecl_type} eclipse times using Cross-Correlation and Ingress/Egress methods ...")

        eclipse_lc_phase = ((eclipse_lc_time - pdgrm_results.T0) / P - 0.5) % 1  # Phased around 0.5
        phases_sorted = eclipse_lc_phase[np.argsort(eclipse_lc_phase)]
        fluxes_sorted = eclipse_lc_flux[np.argsort(eclipse_lc_phase)]

        unphased_eclipse_lc = (phases_sorted - eclipse_lc_phase[0]) * P
        initial_shift = unphased_eclipse_lc[0]
        unphased_eclipse_lc = unphased_eclipse_lc - initial_shift

        coeffs = np.polyfit([unphased_eclipse_lc[0], unphased_eclipse_lc[-1]], [fluxes_sorted[0], fluxes_sorted[-1]], 1)
        fit = np.polyval(coeffs, unphased_eclipse_lc)

        fluxes_sorted = fluxes_sorted / fit

        degree = 2
        n_points = len(unphased_eclipse_lc)
        n_knots = max(5, min(30, n_points // 10))

        # Ensure it doesn't exceed theoretical maximum knots
        max_knots = n_points - (degree + 1)
        n_knots = min(n_knots, max_knots)

        spline = LSQUnivariateSpline(unphased_eclipse_lc, fluxes_sorted, t=np.linspace(unphased_eclipse_lc[degree + 1], unphased_eclipse_lc[-(degree + 1)], n_knots), k=degree)
        # Ingress/ Egress Modeling Method

        if np.all(np.isnan(fluxes_sorted)):
            config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
            return 0, 0, 0, 0

        flux_left = fluxes_sorted[:np.nanargmin(fluxes_sorted)]
        flux_right = fluxes_sorted[np.nanargmin(fluxes_sorted):]

        noneclipse_flux = np.nanmedian(phased_lc.flux).value

        half_depth = (noneclipse_flux - fluxes_sorted[np.nanargmin(fluxes_sorted)]) / 2 + fluxes_sorted[np.nanargmin(fluxes_sorted)]
        dt_outer_flux = half_depth + 0.8 * (noneclipse_flux - half_depth)
        ingress_outer_index, _, _ = find_nearest(unphased_eclipse_lc[:np.nanargmin(fluxes_sorted)], flux_left, dt_outer_flux)
        egress_outer_index, _, _ = find_nearest(unphased_eclipse_lc[np.nanargmin(fluxes_sorted):], flux_right, dt_outer_flux)

        dt_inner_flux = half_depth - 0.8 * (noneclipse_flux - half_depth)
        ingress_inner_index, _, _ = find_nearest(unphased_eclipse_lc[:np.nanargmin(fluxes_sorted)], flux_left, dt_inner_flux)
        egress_inner_index, _, _ = find_nearest(unphased_eclipse_lc[np.nanargmin(fluxes_sorted):], flux_right, dt_inner_flux)

        egress_inner_index = egress_inner_index + np.nanargmin(fluxes_sorted)
        egress_outer_index = egress_outer_index + np.nanargmin(fluxes_sorted)

        ingress_unphased = unphased_eclipse_lc[ingress_outer_index:ingress_inner_index] - unphased_eclipse_lc[ingress_outer_index]
        ingress_fluxes = fluxes_sorted[ingress_outer_index:ingress_inner_index]

        egress_unphased = unphased_eclipse_lc[egress_inner_index:egress_outer_index] - unphased_eclipse_lc[ingress_outer_index]
        egress_fluxes = fluxes_sorted[egress_inner_index:egress_outer_index]

        if len(ingress_unphased) < 1 or len(egress_unphased) < 1:
            return 0, 0, 0, 0

        n_points = len(ingress_unphased)
        n_knots = max(5, min(30, n_points // 10))
        max_knots = n_points - (degree + 1)
        n_knots = min(n_knots, max_knots)
        ingress_spline = LSQUnivariateSpline(ingress_unphased, ingress_fluxes, t=np.linspace(ingress_unphased[degree + 1], ingress_unphased[-(degree + 1)], n_knots), k=degree)

        n_points = len(egress_unphased)
        n_knots = max(5, min(30, n_points // 10))
        max_knots = n_points - (degree + 1)
        n_knots = min(n_knots, max_knots)
        egress_spline = LSQUnivariateSpline(egress_unphased, egress_fluxes, t=np.linspace(egress_unphased[degree + 1], egress_unphased[-(degree + 1)], n_knots), k=degree)

        center_offset = (0.1 * P) - unphased_eclipse_lc[ingress_outer_index]

        # Define Ingress/Egress boundaries

        dt1_ingress = unphased_eclipse_lc[ingress_outer_index]
        dt2_ingress = unphased_eclipse_lc[ingress_inner_index]

        dt1_egress = unphased_eclipse_lc[egress_inner_index]
        dt2_egress = unphased_eclipse_lc[egress_outer_index]
        valid_T_preds = []

        for T_pred in pdgrm_results.transit_times:  # Using predicted eclipse times from TLS as a starting point

            mask = abs(eclipse_lc.time.value - T_pred) <= 0.1 * P
            local_exptime = mode(np.diff(eclipse_lc.time.value[mask])).mode
            local_expected_npts = 2 * 0.1 * P / local_exptime
            mask_left = eclipse_lc.time.value[mask] < T_pred
            mask_right = eclipse_lc.time.value[mask] > T_pred

            ingress_coverage = np.sum(mask_left) / local_expected_npts
            egress_coverage = np.sum(mask_right) / local_expected_npts

            if (np.sum(mask_left) > 1) & (np.sum(mask_right) > 1) & (ingress_coverage >= 0.1) & (egress_coverage >= 0.1):  # Make sure there's enough data points

                local_times = eclipse_lc.time.value[mask]
                local_fluxes = eclipse_lc.flux.value[mask]
                local_fluxerrs = eclipse_lc.flux_err.value[mask]
                local_sector = eclipse_lc.sector[mask]

                coeffs = np.polyfit([local_times[0], local_times[-1]], [local_fluxes[0], local_fluxes[-1]], 1)
                fit = np.polyval(coeffs, local_times)

                local_fluxes = local_fluxes / fit

                ingress_times = local_times[np.where((local_times > local_times[0] + dt1_ingress) & (local_times < local_times[0] + dt2_ingress))]
                ingress_fluxes = local_fluxes[np.where((local_times > local_times[0] + dt1_ingress) & (local_times < local_times[0] + dt2_ingress))]

                if len(ingress_times) < 1:
                    continue

                egress_times = local_times[np.where((local_times > local_times[0] + dt1_egress) & (local_times < local_times[0] + dt2_egress))]
                egress_fluxes = local_fluxes[np.where((local_times > local_times[0] + dt1_egress) & (local_times < local_times[0] + dt2_egress))]

                if len(egress_times) < 1:
                    continue

                shift_limit = 0.05
                num_trials = 1000

                shifts = np.linspace(-shift_limit, shift_limit, num_trials)
                chi2s = []
                chi2s_ingress = []
                chi2s_egress = []

                cut_times_array = []
                cut_fluxes_array = []
                model_flux_arrays = []

                for shift in shifts:

                    # Cross Correlation

                    test_times = local_times - T_pred + (0.1 * P) - shift
                    cut_times = test_times[np.where((test_times > test_times[0] + shift_limit) & (test_times < test_times[-1] - shift_limit))]
                    cut_fluxes = local_fluxes[np.where((test_times > test_times[0] + shift_limit) & (test_times < test_times[-1] - shift_limit))]
                    model_flux = spline(cut_times)
                    chi2 = np.sum((cut_fluxes - model_flux)**2 / get_lc_noise_level(local_fluxes)**2)
                    cut_times_array.append(cut_times)
                    cut_fluxes_array.append(cut_fluxes)
                    model_flux_arrays.append(model_flux)
                    chi2s.append(chi2)

                    # Ingress/Egress Modeling

                    test_times_ingress = ingress_times - T_pred + center_offset - shift
                    test_times_egress = egress_times - T_pred + center_offset - shift

                    cut_ingress_times = test_times_ingress[np.where((test_times_ingress > test_times_ingress[0] + shift_limit) & (test_times_ingress < test_times_ingress[-1] - shift_limit))]
                    cut_ingress_fluxes = ingress_fluxes[np.where((test_times_ingress > test_times_ingress[0] + shift_limit) & (test_times_ingress < test_times_ingress[-1] - shift_limit))]

                    cut_egress_times = test_times_egress[np.where((test_times_egress > test_times_egress[0] + shift_limit) & (test_times_egress < test_times_egress[-1] - shift_limit))]
                    cut_egress_fluxes = egress_fluxes[np.where((test_times_egress > test_times_egress[0] + shift_limit) & (test_times_egress < test_times_egress[-1] - shift_limit))]

                    model_flux_ingress = ingress_spline(cut_ingress_times)
                    chi2_ingress = np.sum((cut_ingress_fluxes - model_flux_ingress)**2 / get_lc_noise_level(local_fluxes)**2)

                    model_flux_egress = egress_spline(cut_egress_times)
                    chi2_egress = np.sum((cut_egress_fluxes - model_flux_egress)**2 / get_lc_noise_level(local_fluxes)**2)

                    chi2s_ingress.append(chi2_ingress)
                    chi2s_egress.append(chi2_egress)

                # CC

                best_idx = np.argmin(chi2s)
                best_shift = shifts[best_idx]
                best_t0 = T_pred + best_shift

                observed_eclipse_times_cc.append(best_t0)

                # CC Uncertainty

                try:

                    local_fluxerrs = np.ones(len(local_fluxes)) * get_lc_noise_level(local_fluxes)
                    best_t0_err = get_t0_err_shift(local_times, local_fluxes, local_fluxerrs, T_pred, P, spline, shift_limit, best_shift, 0, gress=False)

                    observed_eclipse_time_errs_cc.append(best_t0_err)

                except Exception:
                    observed_eclipse_time_errs_cc.append((3 / (24 * 60)))

                # Ingress/Egress

                best_shift_ingress = shifts[np.argmin(chi2s_ingress)]
                best_shift_egress = shifts[np.argmin(chi2s_egress)]
                best_shift = (best_shift_ingress + best_shift_egress) / 2
                best_t0 = T_pred + best_shift

                observed_eclipse_times_gress.append(best_t0)

                # Ingress Uncertainty

                try:

                    local_fluxerrs = np.ones(len(ingress_fluxes)) * get_lc_noise_level(ingress_fluxes)
                    best_t0_err_ingress = get_t0_err_shift(ingress_times, ingress_fluxes, local_fluxerrs, T_pred, P, ingress_spline, shift_limit, best_shift_ingress, center_offset, gress=True)

                    # Egress Uncertainty

                    local_fluxerrs = np.ones(len(egress_fluxes)) * get_lc_noise_level(egress_fluxes)
                    best_t0_err_egress = get_t0_err_shift(egress_times, egress_fluxes, local_fluxerrs, T_pred, P, egress_spline, shift_limit, best_shift_egress, center_offset, gress=True)

                    best_t0_err = np.sqrt(best_t0_err_ingress**2 + best_t0_err_egress**2) / 2

                    observed_eclipse_time_errs_gress.append(best_t0_err)
                except Exception:
                    observed_eclipse_time_errs_gress.append((3 / (24 * 60)))

                valid_T_preds.append(T_pred)
        print(f"\tObtained {len(valid_T_preds)} {ecl_type} eclipse times for Cross-Correlation and Ingress/Egress methods...")

    return observed_eclipse_times_halfdepth, observed_eclipse_times_folding, observed_eclipse_times_cc, observed_eclipse_times_gress, np.array(observed_eclipse_time_errs_hd), np.array(observed_eclipse_time_errs_fold), np.array(observed_eclipse_time_errs_cc), np.array(observed_eclipse_time_errs_gress)


def compute_eclipse_times(tic_id, ecl_dict, epoch_width=0.2, methods=['hd', 'fold', 'cc', 'gress', 'batman'],
                           return_global_eclipse_params=False,verbose=True):
    """
    Compute eclipse times using various methods and return the observed times and errors.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target.
    ecl_dict : dict
        Dictionary containing phased light curves, non-eclipse light curves, periods, and TLS results for primary and secondary eclipses.
        As returned by :func:`EBP_sweep.periods.separate_eclipses`.
    epoch_width : float, optional
        Width of the eclipse epoch used in batman. Default is 0.2*P to get more Ellipsoidal variation.
        Other methods strictly use 0.1*P to avoid Ellipsoidal variation.
    methods : list of str, optional
        List of methods to use for eclipse time computation.
        'hd':  half-depth method,
        'fold': phase-folding method,
        'cc': cross-correlation method,
        'gress': ingress/egress cc method,
        'batman': batman model fitting.

        Default is ['hd', 'fold', 'cc', 'gress', 'batman'].
        'hd' is always computed with 'fold' and 'cc' with 'gress'.
        The returned per-method arrays are always ordered
        ['hd', 'fold', 'cc', 'gress', 'batman'] regardless of the order
        given here, so pass a subset in that relative order (e.g. omit
        entries rather than reordering them) for the method labelling in
        `compute_oc_and_best_period` to line up correctly.
    return_global_eclipse_params : bool, optional
        If True (and 'batman' is in ``methods``), return only the pooled global
        eclipse shape parameters for the primary and secondary eclipse instead
        of running the full per-epoch timing. Useful for reusing the TESS-derived
        eclipse shape to fit sparse ground-based follow-up data (see also
        :func:`EBP_sweep.io.read_global_eclipse_params`, which reads the
        pooled shape parameters saved to disk back out, and
        ``notebooks/Follow_up_fitting.ipynb``).
    verbose : bool, optional
        If True, print progress messages. Default is True.

    Returns
    -------
    (observed_primary_arrays, observed_secondary_arrays,
                primary_errs, secondary_errs), methods_used
    or None on failure.
    """
    p = ecl_dict
    kwargs_pri = (tic_id, p['phased'], p['lc_nonsecondary'], p['P_primary'], p['results_primary'])
    kwargs_sec = (tic_id, p['phased'], p['lc_nonprimary'], p['P_secondary'], p['results_secondary'])
    if isinstance(methods, str):
        methods = [methods]
    for m in methods:
        if m not in ['hd', 'fold', 'cc', 'gress', 'batman']:
            raise ValueError(f"Method {m} is not recognized. Must be one of ['hd', 'fold', 'cc', 'gress', 'batman'].")
    bat_pri = bat_sec = []
    bat_pri_err = bat_sec_err = []

    if 'batman' in methods:
        print('Fitting primary eclipses with batman model...')
        bat_pri, bat_pri_err, indv_shape_params, global_shape_params = get_batman_eclipse_times(*kwargs_pri, 'pri', epoch_width)
        P_bat = global_shape_params['P']
        if verbose:
            print(f"\tP_bat (pri) = {P_bat:.7f} days")
        plot_shape_variation(tic_id, indv_shape_params, global_shape_params, 'pri')

        if bat_pri == 0:
            config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
            return None

        if verbose:
            print('Fitting secondary eclipses with batman model...')
        bat_sec, bat_sec_err, indv_shape_params_sec, global_shape_params_sec = get_batman_eclipse_times(*kwargs_sec, 'sec', epoch_width)
        P_bat_sec = global_shape_params_sec['P']
        if verbose:
            print(f"\tP_bat (sec) = {P_bat_sec:.7f} days")
        plot_shape_variation(tic_id, indv_shape_params_sec, global_shape_params_sec, 'sec')

        if bat_sec == 0:
            config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
            return None
        if return_global_eclipse_params:
            return global_shape_params, global_shape_params_sec

    hd_pri, fold_pri, cc_pri, gress_pri, *errs_pri = get_eclipse_times(*kwargs_pri, 'pri', methods=methods)
    if hd_pri == 0:
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None

    hd_sec, fold_sec, cc_sec, gress_sec, *errs_sec = get_eclipse_times(*kwargs_sec, 'sec', methods=methods)
    if hd_sec == 0:
        config.flag_ticid(tic_id, config.MANUAL_TICIDS_LOG)
        return None

    obs_pri = [hd_pri, fold_pri, cc_pri, gress_pri, bat_pri]
    obs_sec = [hd_sec, fold_sec, cc_sec, gress_sec, bat_sec]
    err_pri = list(errs_pri) + [bat_pri_err]
    err_sec = list(errs_sec) + [bat_sec_err]

    obs_pri = [ob for ob in obs_pri if list(ob) != []]
    obs_sec = [ob for ob in obs_sec if list(ob) != []]
    err_pri = [er for er in err_pri if list(er) != []]
    err_sec = [er for er in err_sec if list(er) != []]

    return (obs_pri, obs_sec, err_pri, err_sec), methods


def compute_oc_and_best_period(tic_id, obs_pri, obs_sec, err_pri, err_sec, 
                                P_primary, P_secondary, methods,verbose=True):
    """Compute O-C arrays, select best method, and refine periods.

    Returns a dict with OC arrays, corrected periods, best indices, etc.
    """
    P_avg = (P_primary + P_secondary) / 2
    all_obs = obs_pri + obs_sec
    O_Cs, cycles = [], []
    for obs in all_obs:
        if obs == []:
            continue
        cyc = np.round((np.array(obs) - obs[0]) / P_avg).astype(int)
        comp = obs[0] + cyc * P_avg
        OC = np.array(obs) - comp
        O_Cs.append(OC)
        cycles.append(cyc)

    OCs_pri, OCs_sec = O_Cs[:int(len(O_Cs) / 2)], O_Cs[int(len(O_Cs) / 2):]

    def _weighted_rms_and_coeffs(obs_arr, OC_arr, errs_arr):
        errs = np.array([5 if e == 3 / (24 * 60) else e * 24 * 60 for e in errs_arr])
        try:
            coeffs, covm = np.polyfit(obs_arr, OC_arr * 24 * 60, 1, w=1 / errs, cov=True)
            uncert = np.sqrt(covm[0, 0])
        except Exception:
            coeffs = np.polyfit(obs_arr, OC_arr * 24 * 60, 1, w=1 / errs)
            uncert = np.nan
        fit = np.polyval(coeffs, obs_arr)
        res = OC_arr * 24 * 60 - fit
        wrms = np.sqrt(np.sum(res ** 2 / errs ** 2) / np.sum(1 / errs ** 2))
        return wrms, coeffs, uncert

    pri_stds, pri_coeffs, pri_uncerts = [], [], []
    sec_stds, sec_coeffs, sec_uncerts = [], [], []
    for i in range(len(obs_pri)):
        ws, co, un = _weighted_rms_and_coeffs(obs_pri[i], OCs_pri[i], err_pri[i])
        pri_stds.append(ws)
        pri_coeffs.append(co)
        pri_uncerts.append(un)

    for i in range(len(obs_sec)):
        ws, co, un = _weighted_rms_and_coeffs(obs_sec[i], OCs_sec[i], err_sec[i])
        sec_stds.append(ws)
        sec_coeffs.append(co)
        sec_uncerts.append(un)

    bi_pri = select_best_index(pri_stds)  # best index for primary method
    bi_sec = select_best_index(sec_stds)  # best index for secondary method
    if verbose:
        print(f"\nBest primary method index: {bi_pri} ({methods[bi_pri]}), Best secondary method index: {bi_sec} ({methods[bi_sec]})")

    # Remove trends from both OC "lines"
    mid_coeffs = (pri_coeffs[bi_pri] + sec_coeffs[bi_sec]) / 2

    pri_Pshift = (mid_coeffs - pri_coeffs[bi_pri])[0] * P_primary / (24 * 60)
    sec_Pshift = (mid_coeffs - sec_coeffs[bi_sec])[0] * P_secondary / (24 * 60)

    P_pri_new = P_primary + pri_Pshift
    P_sec_new = P_secondary + sec_Pshift

    # Correct for P in plots
    OC_pri_corr = OCs_pri[bi_pri] * 24 * 60 - np.polyval(mid_coeffs, obs_pri[bi_pri])  # corrected O-C for primary due to period refinement
    OC_sec_corr = OCs_sec[bi_sec] * 24 * 60 - np.polyval(mid_coeffs, obs_sec[bi_sec])  # corrected O-C for secondary due to period refinement

    # Save OC Data
    primP_err = pri_uncerts[bi_pri]
    secP_err = sec_uncerts[bi_sec]

    primary_data = np.column_stack((obs_pri[bi_pri], err_pri[bi_pri], OC_pri_corr, np.zeros(len(OC_pri_corr))))
    secondary_data = np.column_stack((obs_sec[bi_sec], err_sec[bi_sec], OC_sec_corr, np.ones(len(OC_sec_corr))))

    combined_data = np.vstack((primary_data, secondary_data))
    head = "eclipse_time,eclipse_time_err,OC,flag\nflag: 0=primary, 1=secondary"
    fmt = ['%.15f', '%.15f', '%.19f', '%d']
    out_dir = config.data_path(config.OC_TXT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f'TIC{tic_id[4:]}_OCwErr_new.txt'), combined_data, fmt=fmt, delimiter=",", header=head)

    return dict(
        OC_pri_corr=OC_pri_corr, OC_sec_corr=OC_sec_corr,
        P_primary_new=P_pri_new, P_secondary_new=P_sec_new,
        best_index_primary=bi_pri, best_index_secondary=bi_sec,
        pri_uncerts=pri_uncerts, sec_uncerts=sec_uncerts,
        pri_stds=pri_stds, sec_stds=sec_stds,
    )
