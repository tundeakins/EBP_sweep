"""Fitting sparse ground-based follow-up photometry against a TESS-derived eclipse shape.

A single ground-based eclipse observation rarely has enough phase coverage to
constrain a full eclipse model on its own. This module reuses the pooled
eclipse shape (depth, duration, impact parameter) already fit to TESS data by
:func:`EBP_sweep.timing.compute_eclipse_times` (with
``return_global_eclipse_params=True``) and fits only the mid-eclipse time
(plus a small nuisance baseline) to new photometry — e.g. magnitude time
series exported from AAVSO/WebObs or similar.
"""

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

from .batman_fit import batman_flux_model, fit_global_eclipse_shape
from .io import mag_to_flux


def to_bjd_tdb(time_jd_utc, ra, dec, site_name):
    """Convert JD(UTC) observation times to BJD_TDB (barycentric-corrected).

    Parameters
    ----------
    time_jd_utc : array-like
        Observation times as Julian Dates in UTC.
    ra, dec : float
        Right ascension and declination of the target, in degrees.
    site_name : str
        Name recognised by ``astropy.coordinates.EarthLocation.of_site``,
        e.g. ``'Roque de los Muchachos, La Palma'``.

    Returns
    -------
    np.ndarray
        Times converted to BJD_TDB.
    """
    star = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    observatory_location = EarthLocation.of_site(site_name)
    t = Time(time_jd_utc, format='jd', scale='utc', location=observatory_location).tdb
    ltt = t.light_travel_time(star, 'barycentric', ephemeris='builtin')
    return (t + ltt).value


def load_ground_based_photometry(filepath, ra, dec, site_name, skiprows=7, delimiter=",",
                                  usecols=(1, 2, 3), nsim=1000, err_scale=1.0, flux_floor=None):
    """Load a ground-based magnitude time series as barycentric-corrected relative flux.

    Expects a text file with (time, magnitude, magnitude_error) columns, as
    produced by e.g. AAVSO/WebObs exports.

    Parameters
    ----------
    filepath : str
        Path to the photometry file.
    ra, dec : float
        Target coordinates in degrees, used for the barycentric time correction.
    site_name : str
        Name recognised by ``astropy.coordinates.EarthLocation.of_site``.
    skiprows, delimiter, usecols : optional
        Passed straight through to ``numpy.loadtxt``.
    nsim : int, optional
        Number of Monte Carlo samples used in the magnitude-to-flux conversion (see
        :func:`EBP_sweep.io.mag_to_flux`). Default 1000.
    err_scale : float, optional
        Extra multiplicative factor applied to the normalised flux errors, to
        inflate them for a known-noisier night. Default 1.0 (no inflation).
    flux_floor : float, optional
        If given, the max-normalised flux is additionally shifted so its
        minimum equals this value — useful for placing separate nights of a
        partially-covered eclipse on a common baseline before fitting.
        Default is None (no shift).

    Returns
    -------
    time_bjd_tdb, flux, flux_err : np.ndarray
    """
    t, m, e = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter, usecols=usecols, unpack=True, dtype=float)
    flux, flux_err = mag_to_flux(m, e, nsim=nsim)

    flux_err = flux_err / np.max(flux) * err_scale
    flux = flux / np.max(flux)
    if flux_floor is not None:
        flux = flux - np.min(flux) + flux_floor

    time_bjd_tdb = to_bjd_tdb(t, ra, dec, site_name)
    return time_bjd_tdb, flux, flux_err


def add_radius_ratio(shape_params):
    """Add the radius ratio ``rp = sqrt(D)`` to a global eclipse shape dict, in place, and return it.

    ``shape_params`` is a dict of ``ufloat`` values (keys ``W``, ``D``, ``b``, ``P``)
    as returned by :func:`EBP_sweep.timing.compute_eclipse_times` when called
    with ``return_global_eclipse_params=True``; :func:`run_fit` expects an
    ``rp`` key.
    """
    shape_params['rp'] = shape_params['D'] ** 0.5
    return shape_params


def run_fit(time, flux, flux_err, shape_params, T_pred, u1=(0.63, 0.01), u2=(-0.37, 0.01)):
    """Fit a single ground-based eclipse observation against a TESS-derived global eclipse shape.

    Parameters
    ----------
    time, flux, flux_err : array-like
        The ground-based photometry, e.g. from :func:`load_ground_based_photometry`.
    shape_params : dict
        Global eclipse shape parameters (``ufloat`` values with keys ``P``,
        ``rp``, ``W``, ``b``), e.g. the output of :func:`add_radius_ratio`.
    T_pred : float
        Predicted mid-eclipse time (BJD_TDB), used to seed and bound the fit.
    u1, u2 : tuple of (mean, std), optional
        Gaussian priors on the quadratic limb-darkening coefficients. The
        defaults are broadband values for a solar-type primary; pass
        coefficients appropriate to your instrument bandpass and target
        otherwise.

    Returns
    -------
    fit_params : lmfit.Parameters
        Best-fit parameters, including ``t0``.
    smooth_time : np.ndarray
        Time array for the best-fit model curve.
    bestfit_model : np.ndarray
        Best-fit model flux corresponding to ``smooth_time``.
    """
    fit_priors = dict(t0=(T_pred - 0.1, T_pred, T_pred + 0.1),
                       offset=(-0.4, 0, 0.4),
                       slope=0,
                       P=shape_params['P'].n,
                       rp=(shape_params['rp'].n, 5 * shape_params['rp'].s),
                       dur=(shape_params['W'].n, 1 * shape_params['W'].s),
                       Aev=0,
                       b=shape_params['b'].n,
                       u1=u1,
                       u2=u2,
                       )

    fit_params, fit_model = fit_global_eclipse_shape(time, flux, flux_err,
                                                       param_priors=fit_priors,
                                                       return_model=True)
    smooth_time = np.linspace(fit_params['t0'].value - 0.5, fit_params['t0'].value + 0.5, 1000)
    bestfit_model = batman_flux_model(fit_model, fit_params, t=smooth_time)

    return fit_params, smooth_time, bestfit_model
