"""Eclipse timing via forward-modelling with ``batman``.

This is the fifth (and most precise) eclipse-timing method used by the
pipeline: fit a single global eclipse *shape* (depth, duration, impact
parameter, limb darkening) to all pooled eclipses of a given type, then
re-fit only the mid-eclipse time ``t0`` (plus a small set of shape
parameters with tight Gaussian priors) epoch by epoch.
"""

import os

import batman
import numpy as np
from lmfit import Minimizer, Parameters, minimize
import matplotlib.pyplot as plt
from scipy.stats import mode
from uncertainties import ufloat, unumpy as unp

from . import config
from .plotting import save_epoch_fits_pdf
from .utils import get_lc_noise_level


def Tdur_to_aR(Tdur, b, Rp, P, e=0, w=90, tra_occ="tra"):
    """
    convert transit duration to scaled semi-major axis
    using eqn 41 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
    note (1+p^2) in the equation should instead be (1+p)^2

    Parameters
    -----------
    Tdur: float, ufloat, array-like;
        The transit duration in days.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.

    Returns
    --------
    aR: array-like;
        The scaled semi-major axis of the planet.
    """
    w = unp.radians(w)
    esinw = e * unp.sin(w) if tra_occ == "tra" else -e * unp.sin(w)
    ecc_fac = (1 - e ** 2) / (1 + esinw)

    numer = (1 + abs(Rp)) ** 2 - b ** 2
    numer = np.where(numer < 0, np.nan, numer) + 0
    denom = unp.sin(Tdur * np.pi * unp.sqrt(1 - e ** 2) / (P * ecc_fac ** 2)) ** 2 * ecc_fac ** 2

    aR = unp.sqrt(numer / denom + (b / ecc_fac) ** 2)

    if np.iterable(aR):
        return unp.nominal_values(aR) if all(unp.std_devs(aR) == np.zeros_like(aR)) else aR
    else:
        return unp.nominal_values(aR).item() if unp.std_devs(aR) == 0 else aR


def make_transit_params(t0=0.0, period=1.0, rp=0.1, dur=0.1, b=0.0, u=(0.3, 0.2)):
    """Create a batman.TransitParams object for a transit model."""

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = Tdur_to_aR(dur, b, rp, params.per)
    params.inc = np.degrees(np.arccos(b / params.a))
    params.ecc = 0.  # locally circular approximation for shape fitting, not the true system eccentricity
    params.w = 90.
    params.u = list(u)
    params.limb_dark = "quadratic"

    return params


def residual(params, batmodel, t, flux, err):
    """Compute the residuals between the observed flux and the model flux for given parameters."""

    values = params.valuesdict()
    flux_model = batman_flux_model(batmodel, values, t)
    res = (flux - flux_model) / err
    # Add gaussian prior constraints if defined in the parameters
    for p in params:
        u = params[p].user_data  # obtain tuple specifying the normal prior if defined
        if u and u[1] != 0:  # modify residual to account for how far the value is from mean of prior
            res_mod = (u[0] - params[p].value) / u[1]
            res = np.append(res, res_mod)
    return res


def batman_flux_model(model, param_values, t=None):
    """Compute the flux model using batman for given parameters and time array."""

    p = make_transit_params(param_values['t0'], param_values['P'], param_values['rp'], param_values['dur'],
                             param_values['b'], (param_values['u1'], param_values['u2']))
    if t is not None:
        model = batman.TransitModel(p, t)
    phi = 2 * np.pi * (model.t - param_values['t0']) / param_values['P']
    Fev = -param_values['Aev'] * np.cos(2 * phi)
    slope = param_values['slope'] * (model.t - param_values['t0'])

    return model.light_curve(p) + Fev + param_values['offset'] + slope


def fit_global_eclipse_shape(pooled_times, pooled_fluxes, pooled_fluxerrs, pooled_sectors=None,
                              param_priors=dict(t0=0, P=1, rp=0.1, dur=0.1, b=0, u1=0, u2=0, Aev=0, offset=0, slope=0),
                              tic_id=None, ecl_type=None, return_model=False):
    """
    Fit the global eclipse shape using batman and return the best-fit parameters.

    Parameters
    ----------
    pooled_times : array-like
        Times of the pooled eclipses.
    pooled_fluxes : array-like
        Fluxes of the pooled eclipses.
    pooled_fluxerrs : array-like
        Flux errors of the pooled eclipses.
    pooled_sectors : array-like
        Sector numbers of the pooled eclipses.
    tic_id : str
        TIC ID of the target.
    ecl_type : str
        Type of eclipse ('pri' or 'sec') for labeling the plots.
    param_priors : dict
        Priors for the parameters to be fitted. Keys are parameter names and values can be:
        - A single float or int for fixed parameters.
        - A tuple of (mean, std) for normal priors.
        - A tuple of (min, value, max) for uniform priors.
        - A tuple of (mean, std, min, max) for truncated normal priors.

        Parameters include: 't0', 'P', 'rp', 'dur', 'b', 'u1', 'u2', 'Aev', 'offset', 'slope'.
        Each parameter is set to fixed default values: dict(t0=0,P=1,rp=0.1,dur=0.1,b=0,u1=0,u2=0,Aev=0,offset=0,slope=0)
        unless specified in the param_priors dictionary.

    Returns
    -------
    result.params : lmfit.Parameters
        Best-fit parameters from the global eclipse shape fit."""

    exp_time = mode(np.diff(pooled_times)).mode
    model = batman.TransitModel(make_transit_params(),
                                 pooled_times,
                                 exp_time=exp_time,
                                 supersample_factor=int(np.ceil(exp_time * 24 * 60)),
                                 )

    priors = dict(t0=0, P=1, rp=0.1, dur=0.1, b=0, u1=0, u2=0, Aev=0, offset=0, slope=0)
    priors.update(param_priors)  # update the default priors with any user-specified priors

    params = Parameters()
    for key in priors.keys():
        v = priors[key]
        if isinstance(v, (float, int)):
            params.add(key, value=v, vary=False)
        elif isinstance(v, tuple) and len(v) == 2:  # normal (mean, std)
            params.add(key, value=v[0], user_data=(v[0], v[1]))
        elif isinstance(v, tuple) and len(v) == 3:  # uniform (min, value, max)
            assert v[0] <= v[1] <= v[2], f"Invalid uniform prior for parameter '{key}': {v}"
            params.add(key, value=v[1], min=v[0], max=v[2])
        elif isinstance(v, tuple) and len(v) == 4:  # truncated normal (mean, std, min, max)
            assert v[2] <= v[0] <= v[3], f"Invalid truncated normal prior for parameter '{key}': {v}"
            params.add(key, value=v[0], min=v[2], max=v[3], user_data=(v[0], v[1]))
        else:
            raise ValueError(f"Invalid prior specification for parameter '{key}': {v}")

    result = minimize(residual, params, args=(model, pooled_times, pooled_fluxes, pooled_fluxerrs),
                       method='leastsq', nan_policy='propagate')
    print(f"\tGLOBAL FIT: {result.message}")

    # Cycle through parameters, fixing one at a time, resetting each before trying the next
    if 'Could not estimate error-bars' in result.message:
        for p in ['P', 'Aev', 'b']:
            print(f"\tRetrying with {p} fixed to {params[p].value:.7f}...")
            params[p].vary = False
            result = minimize(residual, params, args=(model, pooled_times, pooled_fluxes, pooled_fluxerrs),
                               method='leastsq', nan_policy='propagate')
            print(f"\tGLOBAL FIT ({p} fixed): {result.message}")
            params[p].vary = True  # reset before trying next parameter
            if 'Could not estimate error-bars' not in result.message:
                break

    result_values = {k: result.params[k].value for k in ('t0', 'P', 'rp', 'dur', 'b', 'u1', 'u2', 'Aev', 'offset', 'slope')}
    P_bat = ufloat(result.params['P'].value, result.params['P'].stderr) if result.params['P'].stderr is not None else result.params['P'].value

    # Plotting the best-fit model for tess_data against the pooled data for each sector
    if tic_id is not None and ecl_type is not None:
        n_sectors = len(np.unique(pooled_sectors)) if pooled_sectors is not None else 1
        n_cols = 2 if n_sectors > 1 else 1
        n_rows = (n_sectors + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()  # Flatten to 1D for easy indexing

        for idx, sector in enumerate(np.unique(pooled_sectors)):
            ax = axes[idx]
            sector_mask = pooled_sectors == sector
            ax.plot(pooled_times[sector_mask], pooled_fluxes[sector_mask], 'k.')
            smooth_time = np.linspace(np.min(pooled_times[sector_mask]), np.max(pooled_times[sector_mask]), 10 * len(pooled_times[sector_mask]))
            ax.plot(smooth_time, batman_flux_model(model, result_values, t=smooth_time), 'r-')
            ax.set_title(f'Sector {int(sector)}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (BTJD)', fontsize=10)
            ax.set_ylabel('Flux', fontsize=10)
            ax.grid(True, alpha=0.3)
        # Hide any unused subplots
        for idx in range(len(np.unique(pooled_sectors)), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{tic_id} - Global eclipse shape fit {ecl_type} - P_bat={P_bat}', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        out_dir = config.fig_path(config.FIG_ECLIPSEFIT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'TIC{tic_id[4:]}_AllEclipseFit_{ecl_type}_P{P_bat.n:.4f}.jpg'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return (result.params, model) if return_model else result.params


def fit_epoch_t0(local_times, local_fluxes, local_fluxerrs, T_pred, period, shape_params, shift_limit):
    """
    Fit the eclipse time (t0) for a single epoch using batman and return the best-fit parameters and model.

    Parameters
    ----------
    local_times : array-like
        Times of the local eclipse.
    local_fluxes : array-like
        Fluxes of the local eclipse.
    local_fluxerrs : array-like
        Flux errors of the local eclipse.
    T_pred : float
        Predicted eclipse time.
    period : float
        Orbital period of the system.
    shape_params : lmfit.Parameters
        Global shape parameters from the global fit.
    shift_limit : float
        Maximum allowed shift in t0 from T_pred.

    Returns
    -------
    result : lmfit.MinimizerResult
        Best-fit parameters from the epoch t0 fit.
    smooth_time : array-like
        Time array for the best-fit model.
    bestfit_model : array-like
        Best-fit model fluxes corresponding to smooth_time.
    """
    exp_time = mode(np.diff(local_times)).mode
    model = batman.TransitModel(
        make_transit_params(T_pred, shape_params['P'].value, shape_params['rp'].value, shape_params['dur'].value, shape_params['b'].value),
        local_times, exp_time=exp_time, supersample_factor=int(np.ceil(exp_time * 24 * 60)),
    )

    params = Parameters()
    prior = shape_params
    # varying
    params.add('t0', value=T_pred, min=T_pred - shift_limit, max=T_pred + shift_limit)
    params.add('offset', value=prior['offset'].value)
    params.add('slope', value=prior['slope'].value, min=-0.1, max=0.1)
    # gaussian prior with mean and 5*stddev from the global fit
    params.add('rp', value=prior['rp'].value, user_data=(prior['rp'].value, 5 * prior['rp'].stderr))
    params.add('dur', value=prior['dur'].value, user_data=(prior['dur'].value, 5 * prior['dur'].stderr))
    params.add('Aev', value=prior['Aev'].value, user_data=(prior['Aev'].value, 5 * prior['Aev'].stderr),
                vary=prior['Aev'].stderr not in [None, 0])
    params.add('b', value=prior['b'].value, min=0, max=1.8, vary=prior['b'].stderr not in [None, 0])
    # fixed
    params.add('P', value=prior['P'].value, vary=False)
    params.add('u1', value=prior['u1'].value, vary=False)
    params.add('u2', value=prior['u2'].value, vary=False)

    minner = Minimizer(residual, params, fcn_args=(model, local_times, local_fluxes, local_fluxerrs))
    result = minner.minimize(method='leastsq')

    result_values = {k: result.params[k].value for k in ('t0', 'P', 'rp', 'dur', 'b', 'u1', 'u2', 'Aev', 'offset', 'slope')}
    smooth_time = np.linspace(result_values['t0'] - 0.2 * result_values['P'], result_values['t0'] + 0.2 * result_values['P'], int(0.4 * result_values['P'] * 24 * 60))
    bestfit_model = batman_flux_model(model, result_values, t=smooth_time)

    return result, smooth_time, bestfit_model


def get_batman_eclipse_times(tic_id, phased_lc, eclipse_lc, P, pdgrm_results, ecl_type, eclipse_cut):
    """
    Use batman to fit the eclipse times for each predicted transit time and return the observed times and errors.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target.
    phased_lc : lightkurve LightCurve
        Phased light curve data.
    eclipse_lc : lightkurve LightCurve
        Eclipse light curve data.
    P : float
        Orbital period of the system.
    pdgrm_results : object
        Results from the periodogram analysis containing predicted transit times.
    ecl_type : str
        Type of eclipse ('pri' or 'sec') for labeling the plots.
    eclipse_cut : float
        Fraction of the period to use for local eclipse fitting.

    Returns
    -------
    observed_eclipse_times_batman : list
        List of observed eclipse times from batman fitting.
    observed_eclipse_time_errs_batman : list
        List of errors on the observed eclipse times from batman fitting.
    """

    if np.all(np.isnan(pdgrm_results.transit_times)):
        return 0, 0

    local_times_arrays = []
    local_fluxes_arrays = []
    local_fluxerrs_arrays = []
    local_sector_arrays = []
    valid_T_preds = []

    for T_pred in pdgrm_results.transit_times:

        mask = abs(eclipse_lc.time.value - T_pred) <= eclipse_cut * P
        local_exptime = mode(np.diff(eclipse_lc.time.value[mask])).mode
        local_expected_npts = 2 * eclipse_cut * P / local_exptime
        mask_left = eclipse_lc.time.value[mask] < T_pred
        mask_right = eclipse_lc.time.value[mask] > T_pred

        ingress_coverage = np.sum(mask_left) / local_expected_npts
        egress_coverage = np.sum(mask_right) / local_expected_npts

        if (np.sum(mask_left) > 1) & (np.sum(mask_right) > 1) & (ingress_coverage >= 0.1) & (egress_coverage >= 0.1):

            local_times = eclipse_lc.time.value[mask]
            local_fluxes = eclipse_lc.flux.value[mask]
            local_fluxerrs = eclipse_lc.flux_err.value[mask]
            local_sector = eclipse_lc.sector[mask]
            if np.all(np.isnan(local_fluxes)):
                continue

            local_times_arrays.append(local_times)
            local_fluxes_arrays.append(local_fluxes)
            local_fluxerrs_arrays.append(np.ones_like(local_fluxes) * get_lc_noise_level(local_fluxes))
            local_sector_arrays.append(local_sector)
            valid_T_preds.append(T_pred)

    if len(local_times_arrays) < 2:
        return 0, 0, 0, 0

    pooled_times = np.concatenate(local_times_arrays)
    pooled_fluxes = np.concatenate(local_fluxes_arrays)
    pooled_fluxerrs = np.concatenate(local_fluxerrs_arrays)
    pooled_sectors = np.concatenate(local_sector_arrays)

    noneclipse_flux = np.nanmedian(phased_lc.flux).value
    depth_guess = max(noneclipse_flux - np.nanmin(pooled_fluxes), 1e-4)
    rp = np.sqrt(depth_guess)
    dur = getattr(pdgrm_results, 'duration', 0.1 * P)
    t0 = pdgrm_results.T0
    try:
        param_priors = dict(t0=(t0 - 0.1, t0, t0 + 0.1),
                             P=(P - 0.1, P, P + 0.1),
                             rp=(0.5 * rp, rp, 1.5 * rp),
                             dur=(min(0.01 * P, 0.9 * dur), dur, 0.3 * P),
                             b=(0, 0.1, 1.8),
                             u1=(0, 0.3, 2),
                             u2=(-1, 0.2, 1),
                             Aev=(-0.1, 0, 0.1),
                             offset=(-0.1, 0, 0.1),
                             slope=(-0.1, 0, 0.1)
                             )

        shape_params = fit_global_eclipse_shape(pooled_times, pooled_fluxes, pooled_fluxerrs, pooled_sectors,
                                                 param_priors, tic_id, ecl_type)
    except Exception:
        return 0, 0, 0, 0

    observed_eclipse_times_batman = []
    observed_eclipse_time_errs_batman = []
    shift_limit = 0.05 * P
    epoch_fits = []  # (times, fluxes, model_flux, t0_fit, t0_err)
    global_shape_params = dict(W=ufloat(shape_params['dur'].value, shape_params['dur'].stderr),
                                D=ufloat(shape_params['rp'].value, shape_params['rp'].stderr) ** 2,
                                b=ufloat(shape_params['b'].value, shape_params['b'].stderr) if shape_params['b'].stderr not in [0, None] else ufloat(shape_params['b'].value, 0),
                                P=ufloat(shape_params['P'].value, shape_params['P'].stderr) if shape_params['P'].stderr not in [0, None] else ufloat(shape_params['P'].value, 0)
                                )

    indv_shape_params = dict(W=[], D=[], b=[])

    print(f"\tFitting {len(valid_T_preds)} individual eclipses")
    for T_pred, local_times, local_fluxes, local_fluxerrs in zip(valid_T_preds, local_times_arrays, local_fluxes_arrays, local_fluxerrs_arrays):
        try:
            result, bestfit_time, bestfit_model = fit_epoch_t0(local_times, local_fluxes, local_fluxerrs, T_pred, P, shape_params, shift_limit)
            fit_params = result.params
            t0_fit = fit_params['t0'].value
            t0_err = fit_params['t0'].stderr
            indv_shape_params['W'].append(ufloat(fit_params['dur'].value, fit_params['dur'].stderr))
            indv_shape_params['D'].append(ufloat(fit_params['rp'].value, fit_params['rp'].stderr) ** 2)
            indv_shape_params['b'].append(ufloat(fit_params['b'].value, fit_params['b'].stderr))
            if t0_err is None:
                raise ValueError("no stderr")
            epoch_fits.append((local_times, local_fluxes, bestfit_time, bestfit_model, t0_fit, t0_err))
        except Exception:
            t0_fit = np.nan
            t0_err = np.nan
            epoch_fits.append((local_times, local_fluxes, np.full_like(local_fluxes, np.nan), np.full_like(local_fluxes, np.nan), t0_fit, t0_err))

        if np.isfinite(t0_err):
            observed_eclipse_times_batman.append(np.float64(t0_fit))
            observed_eclipse_time_errs_batman.append(np.float64(t0_err))

    if len(observed_eclipse_times_batman) < 2:
        return 0, 0, 0, 0

    save_epoch_fits_pdf(tic_id, P, epoch_fits, ecl_type, 'batman')

    return observed_eclipse_times_batman, np.array(observed_eclipse_time_errs_batman), indv_shape_params, global_shape_params
