"""Eclipse timing via forward-modelling with ``batman``.

This is the fifth (and most precise) eclipse-timing method used by the
pipeline: fit a single global eclipse *shape* (depth, duration, impact
parameter, limb darkening) to all pooled eclipses of a given type, then
re-fit only the mid-eclipse time ``t0`` (plus a small set of shape
parameters with tight Gaussian priors) epoch by epoch.
"""

import os
import pandas as pd

import batman
import numpy as np
from lmfit import Minimizer, Parameters, minimize
import matplotlib.pyplot as plt
from scipy.stats import mode
from uncertainties import ufloat, unumpy as unp

from . import config
from .plotting import save_epoch_fits_pdf
from .utils import get_lc_noise_level, red_noise_beta_factor, phase_fold


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


def residual(params, batmodel, t, flux, err, visit_idx=None):
	"""Compute the residuals between the observed flux and the model flux for given parameters."""

	values = params.valuesdict()
	flux_model = batman_flux_model(batmodel, values, t, visit_idx=visit_idx)
	res = (flux - flux_model) / err
	# Add gaussian prior constraints if defined in the parameters
	for p in params:
		u = params[p].user_data  # obtain tuple specifying the normal prior if defined
		if u and u[1] != 0:  # modify residual to account for how far the value is from mean of prior
			res_mod = (u[0] - params[p].value) / u[1]
			res = np.append(res, res_mod)
	return res


def batman_flux_model(model, param_values, t=None, visit_idx=None, return_trend=False):
	"""Compute the flux model using batman for given parameters and time array.

	visit_idx : array-like of int, optional
		Per-point 0-based visit index (one entry per point of ``model.t``/``t``) selecting
		which ``offset_<i>``/``slope_<i>`` entry of ``param_values`` applies to that point,
		so different visits can share the same eclipse shape but have independent baseline
		offset and slope. If None (default), the single global 'offset'/'slope' entries of
		``param_values`` are applied to every point, matching the original behaviour.
	"""

	p = make_transit_params(param_values['t0'], param_values['P'], param_values['rp'], param_values['dur'],
							 param_values['b'], (param_values['u1'], param_values['u2']))
	if t is not None:
		model = batman.TransitModel(p, t)
	phi = 2 * np.pi * (model.t - p.t0) / p.per
	Fev = -param_values['Aev'] * np.cos(2 * phi)

	if visit_idx is None:
		offset = param_values['offset']
		t_ref   = p.t0 + p.per * np.round((model.t.mean() - p.t0)/p.per)
		slope = param_values['slope'] * (model.t - t_ref)
	else:
		visit_idx = np.asarray(visit_idx)
		n_visits = int(visit_idx.max()) + 1
		offset_vals = np.array([param_values[f'offset_{i}'] for i in range(n_visits)])
		slope_vals = np.array([param_values[f'slope_{i}'] for i in range(n_visits)])
		offset = offset_vals[visit_idx]

		visit_centers = np.array([model.t[visit_idx == i].mean() for i in range(n_visits)])
		t_ref = np.array([p.t0 + p.per * np.round((model.t[visit_idx == i].mean() - p.t0)/p.per) for i in range(n_visits)])
		t_ref = t_ref[visit_idx]
		slope = slope_vals[visit_idx] * (model.t - t_ref)

	return model.light_curve(p) + Fev + offset + slope if not return_trend else (model.light_curve(p) + Fev + offset + slope, offset + slope)


def fit_global_eclipse_shape(pooled_times, pooled_fluxes, pooled_fluxerrs, pooled_sectors=None,
							  param_priors=dict(t0=0, P=1, rp=0.1, dur=0.1, b=0, u1=0, u2=0, Aev=0, offset=0, slope=0),
							  tic_id=None, ecl_type=None, pooled_visits=None, return_model=False, verbose=True):
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
		Sector numbers of the pooled eclipses, used only to group the diagnostic plot panels
		(one panel per unique value) when `tic_id`/`ecl_type` are given. Defaults to
		`pooled_visits` if not supplied.
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

		When `pooled_visits` is given, the 'offset' and 'slope' entries are fit independently
		per visit instead of as single shared parameters (every other parameter, including
		't0', stays one shared/global value across all visits). Their entry in `param_priors`
		is normally a single prior spec (float/tuple as above) applied to every visit; pass a
		dict instead, keyed by the labels appearing in `pooled_visits`, to give a different
		prior per visit.
	pooled_visits : array-like, optional
		Visit/dataset label for each point in `pooled_times` (e.g. one label per night or
		telescope), same length as `pooled_times`. When given, the eclipse shape ('t0', 'P',
		'rp', 'dur', 'b', 'u1', 'u2', 'Aev') is still fit jointly to all visits, but 'offset'
		and 'slope' are fit separately per unique visit label, so all visits share one eclipse
		model while each keeps its own baseline level and linear trend. Each visit's 'slope'
		is measured relative to that visit's own mean time (not the shared 't0', which may be
		days or weeks away) so 'offset' stays interpretable as the baseline level near that
		visit regardless of how far it sits from 't0'; the per-visit reference times are
		returned as a `visit_centers` attribute on `result.params` (see Returns) so later
		`batman_flux_model(..., visit_idx=...)` calls reproduce the fit exactly. If None
		(default), a single 'offset'/'slope' is fit across all pooled points relative to
		't0', matching the original single-baseline behaviour.
	verbose : bool, optional
		If True, print progress messages. Default is True.

	Returns
	-------
	result.params : lmfit.Parameters
		Best-fit parameters from the global eclipse shape fit. When `pooled_visits` is given,
		'offset'/'slope' are replaced by 'offset_<i>'/'slope_<i>' for each visit index `i`
		(0-based, in the sorted order of `np.unique(pooled_visits)`), and the returned
		`result.params` additionally carries a `.visit_centers` attribute (the per-visit
		reference time each 'slope_<i>' is measured from) and a `.visit_labels` attribute
		(the corresponding `np.unique(pooled_visits)` labels)."""

	exp_time = mode(np.diff(pooled_times)).mode
	model = batman.TransitModel(make_transit_params(),
								 pooled_times,
								 exp_time=exp_time,
								 supersample_factor=int(np.ceil(exp_time * 24 * 60)),
								 )

	priors = dict(t0=0, P=1, rp=0.1, dur=0.1, b=0, u1=0, u2=0, Aev=0, offset=0, slope=0)
	priors.update(param_priors)  # update the default priors with any user-specified priors

	if pooled_visits is not None:
		pooled_visits = np.asarray(pooled_visits)
		unique_visits, visit_idx = np.unique(pooled_visits, return_inverse=True)
		n_visits = len(unique_visits)
		if pooled_sectors is None:
			pooled_sectors = pooled_visits  # default plot grouping to visits when sectors aren't given separately
	else:
		visit_idx = None
		n_visits = 1

	params = Parameters()

	def _add_param(name, v):
		if isinstance(v, (float, int)):
			params.add(name, value=v, vary=False)
		elif isinstance(v, tuple) and len(v) == 2:  # normal (mean, std)
			params.add(name, value=v[0], user_data=(v[0], v[1]) if v[1]>0 else None)
		elif isinstance(v, tuple) and len(v) == 3:  # uniform (min, value, max)
			assert v[0] <= v[1] <= v[2], f"Invalid uniform prior for parameter '{name}': {v}"
			params.add(name, value=v[1], min=v[0], max=v[2])
		elif isinstance(v, tuple) and len(v) == 4:  # truncated normal (mean, std, min, max)
			assert v[2] <= v[0] <= v[3], f"Invalid truncated normal prior for parameter '{name}': {v}"
			params.add(name, value=v[0], min=v[2], max=v[3], user_data=(v[0], v[1]))
		else:
			raise ValueError(f"Invalid prior specification for parameter '{name}': {v}")

	for key in priors.keys():
		v = priors[key]
		if key in ('offset', 'slope') and pooled_visits is not None:
			# one independent parameter per visit; every other key stays a single shared parameter
			for i, vlabel in enumerate(unique_visits):
				vi = v[vlabel] if isinstance(v, dict) else v
				_add_param(f'{key}_{i}', vi)
		else:
			_add_param(key, v)

	fit_args = (model, pooled_times, pooled_fluxes, pooled_fluxerrs, visit_idx)
	result = minimize(residual, params, args=fit_args, method='leastsq', nan_policy='omit')
	if verbose: print(f"\tGLOBAL FIT: {result.message}")

	# Cycle through parameters, fixing one at a time, resetting each before trying the next
	if 'Could not estimate error-bars' in result.message or 'variable did not affect the fit' in result.message:
		for p in ['P', 'Aev', 'b']:
			if params[p].vary:
				if verbose: print(f"\tRetrying with {p} fixed to {params[p].value:.7f}...")
				params[p].vary = False
				result = minimize(residual, params, args=fit_args, method='leastsq', nan_policy='omit')
				if verbose: print(f"\tGLOBAL FIT ({p} fixed): {result.message}")
				params[p].vary = True  # reset before trying next parameter
				if 'Could not estimate error-bars' not in result.message and 'variable did not affect the fit' not in result.message:
					break

	result_values = {k: result.params[k].value for k in ('t0', 'P', 'rp', 'dur', 'b', 'u1', 'u2', 'Aev')}
	if pooled_visits is not None:
		result_values.update({f'offset_{i}': result.params[f'offset_{i}'].value for i in range(n_visits)})
		result_values.update({f'slope_{i}': result.params[f'slope_{i}'].value for i in range(n_visits)})
	else:
		result_values['offset'] = result.params['offset'].value
		result_values['slope'] = result.params['slope'].value

	# Inflate formal (white-noise-only) uncertainties for time-correlated ("red")
	# noise in the residuals -- see fit_epoch_t0 for why this is a post-hoc multiply
	# rather than a refit with inflated flux errors. `pooled_times` jumps between
	# widely-separated eclipse epochs, so beta is estimated within each contiguous
	# epoch window rather than across the gaps between them.
	pooled_phases = phase_fold(pooled_times, result_values['P'], result_values['t0'], phase0=-0.35)
	in_eclipse_mask = abs(pooled_phases) < 0.5*result_values["dur"]/result_values["P"]
	bestfit_flux_at_data = batman_flux_model(model, result_values, visit_idx=visit_idx)
	beta = red_noise_beta_factor(pooled_fluxes[in_eclipse_mask] - bestfit_flux_at_data[in_eclipse_mask], time=pooled_times[in_eclipse_mask])
	for pname in result.params:
		if result.params[pname].vary and result.params[pname].stderr:
			result.params[pname].stderr *= beta

	P_bat = ufloat(result.params['P'].value, result.params['P'].stderr) if result.params['P'].stderr is not None else result.params['P'].value

	# Plotting the best-fit model for tess_data against the pooled data for each sector
	if tic_id is not None and ecl_type is not None:
		n_sectors = len(np.unique(pooled_sectors)) if pooled_sectors is not None else 1
		n_cols = 2 if n_sectors > 1 else 1
		n_rows = (n_sectors + n_cols - 1) // n_cols
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=False, sharey=True)
		axes = np.atleast_2d(axes).flatten()  # Flatten to 1D for easy indexing

		for idx, sector in enumerate(np.unique(pooled_sectors)):
			ax = axes[idx]
			sector_mask = pooled_sectors == sector
			ax.plot(pooled_times[sector_mask], pooled_fluxes[sector_mask], 'k.')
			smooth_time = np.linspace(np.min(pooled_times[sector_mask]), np.max(pooled_times[sector_mask]), 10 * len(pooled_times[sector_mask]))
			if pooled_visits is not None:
				# overplot using whichever visit dominates this panel's points, so the curve
				# reflects that visit's own offset/slope rather than a mismatched one
				sector_visits, counts = np.unique(visit_idx[sector_mask], return_counts=True)
				smooth_visit_idx = np.full(smooth_time.shape, sector_visits[np.argmax(counts)])
				smooth_flux = batman_flux_model(model, result_values, t=smooth_time, visit_idx=smooth_visit_idx)
			else:
				smooth_flux = batman_flux_model(model, result_values, t=smooth_time)
			ax.plot(smooth_time, smooth_flux, 'r-')
			try:  # numeric sector numbers print as before; non-numeric visit labels (e.g. night names) print as-is
				panel_title = f'Sector {int(sector)}'
			except (TypeError, ValueError):
				panel_title = str(sector)
			ax.set_title(panel_title, fontsize=12, fontweight='bold')
			ax.set_xlabel('Time (BTJD)', fontsize=10)
			ax.set_ylabel('Flux', fontsize=10)
			ax.grid(True, alpha=0.3)
		# Hide any unused subplots
		for idx in range(len(np.unique(pooled_sectors)), len(axes)):
			axes[idx].set_visible(False)

		fig.suptitle(f'{tic_id} - Global eclipse shape fit {ecl_type} - P_bat={P_bat}', fontsize=14, fontweight='bold', y=0.995)
		fig.tight_layout()

		out_dir = config.fig_path(os.path.join(config.FIG_ECLIPSEFIT_DIR, f'TIC{tic_id[4:]}'))
		os.makedirs(out_dir, exist_ok=True)
		fig.savefig(os.path.join(out_dir, f'TIC{tic_id[4:]}_AllEclipseFit_{ecl_type}_P{P_bat.n:.4f}.jpg'), dpi=150, bbox_inches='tight')
		plt.close(fig)

	if pooled_visits is not None:
		# expose so that downstream batman_flux_model(..., visit_idx=...) calls (e.g. for
		# plotting) automatically reuse the same per-visit time reference used in the fit
		# result.params.visit_centers = visit_centers
		result.params.visit_labels = unique_visits

	return (result.params, model) if return_model else result.params


def fit_epoch_t0(local_times, local_fluxes, local_fluxerrs, T_pred, period, shape_params, shift_limit,verbose=True):
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
	verbose : bool, optional
		If True, print progress messages. Default is True.

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
	params.add('slope', value=0, min=-1, max=1)
	# gaussian prior with mean and 5*stddev from the global fit
	params.add('rp', value=prior['rp'].value, 
			user_data=(prior['rp'].value, 5 * prior['rp'].stderr))
	params.add('Aev', value=prior['Aev'].value, vary=prior['Aev'].stderr not in [None, 0],
			user_data=(prior['Aev'].value, 5 * prior['Aev'].stderr) if prior['Aev'].stderr not in [None, 0] else None)
	params.add('dur', value=prior['dur'].value, 
			user_data=(prior['dur'].value, 3 * prior['dur'].stderr))
	params.add('b', value=prior['b'].value, min=0, max=1.8, vary=prior['b'].stderr not in [None, 0],
			user_data=(prior['b'].value, 3*prior['b'].stderr) if prior['b'].stderr not in [None, 0] else None)

	params.add('P', value=prior['P'].value, vary=False)
	params.add('u1', value=prior['u1'].value, vary=exp_time*24*60 <= 5, min=0, max=2,  #vary u1 if sampling is fine enough
			user_data=(prior['u1'].value, 1 * prior['u1'].stderr) if exp_time*24*60 <= 5 else None)
	params.add('u2', value=prior['u2'].value, vary=exp_time*24*60 <= 5, min=-1, max=1,  #vary if sampling is fine enough
			user_data=(prior['u2'].value, 1 * prior['u2'].stderr) if exp_time*24*60 <= 5 else None)

	result = minimize(residual, params, args=(model, local_times, local_fluxes, local_fluxerrs),
					   method='leastsq', nan_policy='omit')

	result_values = {k: result.params[k].value for k in ('t0', 'P', 'rp', 'dur', 'b', 'u1', 'u2', 'Aev', 'offset', 'slope')}

	# Inflate formal (white-noise-only) uncertainties for time-correlated ("red")
	# noise in the residuals, which the independent-point leastsq covariance
	# doesn't account for and which otherwise leaves stderr, t0.stderr included,
	# systematically underestimated.
	in_eclipse_mask = abs(local_times - result_values['t0']) < 0.5*result_values["dur"]
	bestfit_flux_at_data = batman_flux_model(model, result_values)
	beta = red_noise_beta_factor(local_fluxes[in_eclipse_mask] - bestfit_flux_at_data[in_eclipse_mask], time=local_times[in_eclipse_mask])
	for pname in result.params:
		if result.params[pname].vary and result.params[pname].stderr:
			result.params[pname].stderr *= beta
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
		return 0, 0, 0, 0, None

	local_times_arrays = []
	local_fluxes_arrays = []
	local_fluxerrs_arrays = []
	local_sector_arrays = []
	valid_T_preds = []

	for T_pred in pdgrm_results.transit_times:

		mask = abs(eclipse_lc.time.value - T_pred) <= eclipse_cut * P
		if not np.any(mask):
			continue
		local_exptime = mode(np.diff(eclipse_lc.time.value[mask])).mode
		local_expected_npts = 2 * eclipse_cut * P / local_exptime
		mask_left = eclipse_lc.time.value[mask] < T_pred
		mask_right = eclipse_lc.time.value[mask] > T_pred

		ingress_coverage = np.sum(mask_left) / local_expected_npts
		egress_coverage = np.sum(mask_right) / local_expected_npts

		if (np.sum(mask_left) > 1) & (np.sum(mask_right) > 1) & (ingress_coverage >= 0.10) & (egress_coverage >= 0.10):

			local_times = eclipse_lc.time.value[mask]
			local_fluxes = eclipse_lc.flux.value[mask]
			# local_fluxerrs = eclipse_lc.flux_err.value[mask]
			local_fluxerrs = np.ones_like(local_fluxes) * get_lc_noise_level(local_fluxes)
			local_sector = eclipse_lc.sector[mask]
			if np.all(np.isnan(local_fluxes)):
				continue

			gd_pts = np.isfinite(local_fluxes)
			local_times_arrays.append(local_times[gd_pts])
			local_fluxes_arrays.append(local_fluxes[gd_pts])
			local_fluxerrs_arrays.append(local_fluxerrs[gd_pts])
			local_sector_arrays.append(local_sector[gd_pts])
			valid_T_preds.append(T_pred)

	if len(local_times_arrays) < 2:
		return 0, 0, 0, 0, None

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
		param_priors = dict(t0	= (t0 - 0.15*P, t0, t0 + 0.15*P),
							P	= (P - 0.1, P, P + 0.1),
							rp	= (0.5 * rp, rp, 1.5 * rp),
							dur	= (min(0.01 * P, 0.9 * dur), dur, 0.3 * P),
							b	= (0, 0.1, 2.8),
							u1	= (0, 0.3, 2),
							u2	= (-1, 0.2, 1),
							Aev	= (-0.1, 0, 0.1),
							offset	= (-0.1, 0, 0.1),
							slope	= (-0.1, 0, 0.1)
							)

		shape_params = fit_global_eclipse_shape(pooled_times, pooled_fluxes, pooled_fluxerrs, pooled_sectors,
												param_priors, tic_id, ecl_type)

		# save eclipse params to csv in out_dir
		shape_params_df = pd.DataFrame({f'{k}_{ecl_type}': [v.value, v.stderr, (v.min,v.max)] for k, v in shape_params.items()}, index=['value', 'stderr', 'bounds']).T

		out_dir = config.fig_path(os.path.join(config.FIG_ECLIPSEFIT_DIR, f'TIC{tic_id[4:]}'))
		os.makedirs(out_dir, exist_ok=True)
		csv_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_GlobalParams.csv')
		if ecl_type == 'pri':
			shape_params_df.to_csv(csv_path)
		else:
			shape_params_df.to_csv(csv_path, mode='a', header=False)
		
	except Exception:
		raise RuntimeError(f"Global eclipse shape fitting failed for TIC {tic_id} {ecl_type} eclipses. Check the light curve and parameters.")
		# return 0, 0, 0, 0

	observed_eclipse_times_batman = []
	observed_eclipse_time_errs_batman = []
	shift_limit = 0.15 * P
	epoch_fits = []  # (times, fluxes, model_times, model_flux, t0_fit, t0_err)
	global_shape_params = dict(	W=ufloat(shape_params['dur'].value, shape_params['dur'].stderr),
								D=ufloat(shape_params['rp'].value, shape_params['rp'].stderr) ** 2,
								b=ufloat(shape_params['b'].value, shape_params['b'].stderr) if shape_params['b'].stderr not in [0, None] else ufloat(shape_params['b'].value, 0),
								P=ufloat(shape_params['P'].value, shape_params['P'].stderr) if shape_params['P'].stderr not in [0, None] else ufloat(shape_params['P'].value, 0)
								)

	indv_shape_params = dict(W=[], D=[], b=[])

	print(f"\tFitting {len(valid_T_preds)} individual eclipses.", end=" ")
	valid_batman_Tpreds = []
	for T_pred, local_times, local_fluxes, local_fluxerrs in zip(valid_T_preds, local_times_arrays, local_fluxes_arrays, local_fluxerrs_arrays):
		# now with eclipse width known, discard any predicted eclipse without enough data points
		mask = abs(local_times - T_pred) < 0.6*shape_params['dur'].value #points in eclipse
		local_exptime = mode(np.diff(local_times)).mode
		local_expected_npts = np.round(shape_params['dur'].value / local_exptime)
		mask_left = local_times[mask] < T_pred
		mask_right = local_times[mask] > T_pred
		if sum(mask_left) < int(0.1*local_expected_npts) or sum(mask_right) < int(0.1*local_expected_npts):  # require at least 10% of expected points on each side of eclipse
			continue

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
		except Exception:
			t0_fit = np.nan
			t0_err = np.nan
			bestfit_time = np.full_like(local_fluxes, np.nan)
			bestfit_model = np.full_like(local_fluxes, np.nan)

		if np.isfinite(t0_err):# and t0_err < 30/(24*60):  # only consider eclipse times with errors less than 30 minutes
			observed_eclipse_times_batman.append(np.float64(t0_fit))
			observed_eclipse_time_errs_batman.append(np.float64(t0_err))
			valid_batman_Tpreds.append(t0_fit)
			epoch_fits.append((local_times, local_fluxes, bestfit_time, bestfit_model, t0_fit, t0_err))


	print(f"{len(valid_batman_Tpreds)} with sufficient datapoints")
	if len(observed_eclipse_times_batman) < 2:
		return 0, 0, 0, 0, None

	save_epoch_fits_pdf(tic_id, P, epoch_fits, ecl_type, 'batman')

	return observed_eclipse_times_batman, np.array(observed_eclipse_time_errs_batman), indv_shape_params, global_shape_params, valid_batman_Tpreds
