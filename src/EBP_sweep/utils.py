"""Generic numeric helpers shared across the pipeline.

These are small, dependency-light functions used by the period-finding,
timing and batman-fitting stages: robust statistics, outlier clipping,
nearest-value search and a couple of array-formatting utilities.
"""

import numpy as np
from scipy.stats import linregress, mode


def robust_std(data):
    """
    Compute the robust standard deviation using the Median Absolute Deviation (MAD).
    https://en.m.wikipedia.org/wiki/Median_absolute_deviation. Better resistance to outliers, unlike the standard deviation.

    Parameters
    -----------
    data: array-like
        The input data.

    Returns
    -------
    float
        The robust standard deviation.
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    robust_std = mad * 1.4826  # Scale factor for consistency with the standard deviation
    return robust_std


def get_lc_noise_level(flux, time=None, max_gap_factor=3):
    """
    Estimate the noise level of a light curve using the Median Absolute Deviation (MAD).

    Parameters
    -----------
    flux: array-like
        The flux values of the light curve.
    time: array-like, optional
        The time values of the light curve. If provided, the function will consider gaps in time to avoid overestimating noise.
    max_gap_factor: float, optional
        The factor to determine the maximum allowed gap in time. Default is 3.

    Returns
    -------
    float
        The estimated noise level.
    """
    dflux = np.diff(flux)  # take consecutive differences, removing slow trends
    # prevent large gaps in data from overestimating the noise level
    if time is not None:
        dt = np.diff(time)
        median_dt = np.nanmedian(dt)
        dflux = dflux[dt < max_gap_factor * median_dt]
    # np.sqrt(2) — corrects for the fact that differencing two noise samples with variance sigma^2
    # yields variance 2*sigma^2, so dividing by sqrt(2) recovers the per-point noise sigma
    return robust_std(dflux) / np.sqrt(2)


def outlier_clipping(x, y, yerr=None, clip=5, width=15, verbose=True, return_clipped_indices=False):
    """
    Remove outliers using a running median method. Points > clip*M.A.D are removed
    where M.A.D is the mean absolute deviation from the median in each window

    Parameters
    -----------
    x: array_like;
        dependent variable.

    y: array_like; same shape as x
        Dependent variable. Data on which to perform clipping

    yerr: array_like(x);
        errors on the dependent variable

    clip: float;
        cut off value above the median. Default is 5

    width: int;
        Number of points in window to use when computing the running median. Must be odd. Default is 15

    Returns
    --------
    x_new, y_new, yerr_new: Each and array with the remaining points after clipping

    """
    from scipy.signal import medfilt

    if clip == 0 or width < 3:
        if yerr is None:
            if return_clipped_indices:
                return x, y, np.full_like(x, False, dtype=bool)
            return x, y
        else:
            if return_clipped_indices:
                return x, y, yerr, np.full_like(x, False, dtype=bool)
            return x, y, yerr

    dd = abs(medfilt(y - 1, width) + 1 - y)  # medfilt pads with zero, so filtering at edge is better if flux level is taken to zero(y-1)
    mad = dd.mean()
    ok = dd < clip * mad

    if verbose:
        print('\nRejected {} points more than {:0.1f} x MAD from the median'.format(sum(~ok), clip))

    if yerr is None:
        if return_clipped_indices:
            return x[ok], y[ok], ~ok

        return x[ok], y[ok]

    if return_clipped_indices:
        return x[ok], y[ok], yerr[ok], ~ok

    return x[ok], y[ok], yerr[ok]


def find_nearest(time_array, flux_array, value):
    """
    Find the index of the flux value in `flux_array` that is closest to the specified `value`.

    Parameters
    -----------
    time_array: array-like
        The time values of the light curve.
    flux_array: array-like
        The flux values of the light curve.
    value: float
        The target flux value to find the nearest match for.
    Returns
    -------
    int, float, float
        The index of the nearest flux value, the corresponding time, and the nearest flux value.
    """

    flux_array = np.asarray(flux_array)
    idx = np.nanargmin(np.abs(flux_array - value))
    time = time_array[idx]
    flux = flux_array[idx]

    return idx, time, flux


def find_nearest_window(flux_array, phase_array, indices, target_flux, window_size):
    """
    Find the phase corresponding to the window with flux values nearest to the target flux.

    Parameters
    -----------
    flux_array: array-like
        The flux values of the light curve.
    phase_array: array-like
        The phase values of the light curve.
    indices: array-like
        The indices of the flux array to consider.
    target_flux: float
        The target flux value to find the nearest match for.
    window_size: int
        The size of the window to consider.

    Returns
    -------
    float
        The phase corresponding to the best matching window.
    """
    num_windows = len(indices) - window_size + 1
    avg_diff = np.zeros(num_windows)

    for i in range(num_windows):
        window_flux = flux_array[indices[i: i + window_size]]
        avg_diff[i] = np.mean(np.abs(window_flux - target_flux))

    # Find the window with the minimum average absolute difference
    min_diff_index = np.nanargmin(avg_diff)
    best_window_center_index = indices[min_diff_index + window_size // 2]  # Get center index of window

    return phase_array[best_window_center_index]


def get_oversampling_factor(time_array):
    """
    Determine the oversampling factor based on the most common time difference between consecutive points.

    Parameters
    -----------
    time_array: array-like
        The time values of the light curve.

    Returns
    -------
    int
        The oversampling factor.
    """
    exp_time = mode(np.diff(time_array)).mode  # get the most common time difference between consecutive points
    return int(np.ceil(exp_time * 24 * 60))  # number of points to go down to ~1minute cadence


def format_ranges(arr):
    """
    Format an array of integers into a string of ranges.

    Parameters
    -----------
    arr: array-like
        The array of integers to format.

    Returns
    -------
    str
        A string representation of the ranges.
    """
    ranges = []
    start = arr[0]
    end = arr[0]

    for num in arr[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append((start, end))
            start = end = num
    ranges.append((start, end))  # Add the final range

    # Format the ranges
    return ', '.join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)


def select_best_index(stds, floor=1e-4, fallback_index=2):
    """
    Select the best index based on the standard deviations.

    Parameters
    ----------
    stds : array-like
        Array of standard deviations.
    floor : float, optional
        Minimum acceptable standard deviation. Default is 1e-4.
    fallback_index : int, optional
        Index to return if no standard deviation exceeds the floor. Default is 2.

    Returns
    -------
    int
        Index of the best standard deviation.
    """

    for idx in np.argsort(stds):
        if stds[idx] > floor:
            return idx

    return fallback_index


def estimate_time_uncertainty(time, flux, flux_err, half_depth):
    """
    Estimate the timing uncertainty of an eclipse using linear regression on the flux values near the half-depth.
    """

    idx = np.argsort(np.abs(flux - half_depth))[:4]
    x = time[idx]
    y = flux[idx]
    yerr = flux_err[idx]

    slope, intercept, _, _, _ = linregress(x, y)  # Perform linear regression

    avg_flux_err = np.mean(yerr)
    time_err = avg_flux_err / np.abs(slope)

    return time_err
