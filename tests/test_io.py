import numpy as np

from EBP_sweep.io import mag_to_flux


def test_mag_to_flux_brighter_magnitude_gives_higher_flux():
    np.random.seed(0)
    fluxes, fluxes_err = mag_to_flux(m=[10.0, 11.0], merr=[0.01, 0.01], nsim=2000)
    assert fluxes[0] > fluxes[1]  # smaller (brighter) magnitude -> more flux
    assert np.all(fluxes_err > 0)


def test_mag_to_flux_output_shape_matches_input():
    np.random.seed(0)
    m = [12.0, 12.5, 13.0]
    merr = [0.02, 0.02, 0.03]
    fluxes, fluxes_err = mag_to_flux(m, merr, nsim=500)
    assert len(fluxes) == len(m)
    assert len(fluxes_err) == len(m)
