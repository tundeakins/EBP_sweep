import os

import numpy as np
import pytest

from EBP_sweep import config
from EBP_sweep.io import mag_to_flux, read_global_eclipse_params


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


def _write_global_params_csv(figures_dir, tic_id):
    """Recreate the CSV format written by EBP_sweep.batman_fit.get_batman_eclipse_times."""
    out_dir = os.path.join(figures_dir, config.FIG_ECLIPSEFIT_DIR, f'TIC{tic_id[4:]}')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_GlobalParams.csv')
    with open(csv_path, "w") as f:
        f.write(",value,stderr,bounds\n")
        f.write('t0_pri,100.0,0.001,"(99.9, 100.1)"\n')
        f.write('P_pri,5.0,,"(4.9, 5.1)"\n')  # stderr blank -> a fixed (non-varied) parameter
        f.write('t0_sec,102.5,0.002,"(102.4, 102.6)"\n')
    return csv_path


def test_read_global_eclipse_params_splits_into_primary_and_secondary_and_strips_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", str(tmp_path))
    _write_global_params_csv(str(tmp_path), "TIC 999999999")

    pri_params, sec_params = read_global_eclipse_params("TIC 999999999")

    assert pri_params["t0"].nominal_value == pytest.approx(100.0)
    assert pri_params["t0"].std_dev == pytest.approx(0.001)
    assert sec_params["t0"].nominal_value == pytest.approx(102.5)
    assert sec_params["t0"].std_dev == pytest.approx(0.002)
    # 'P_pri' -> 'P', not 'P_pri' -- the eclipse-type suffix is stripped
    assert "P" in pri_params and "P_pri" not in pri_params


def test_read_global_eclipse_params_gives_fixed_parameters_zero_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", str(tmp_path))
    _write_global_params_csv(str(tmp_path), "TIC 999999999")

    pri_params, _ = read_global_eclipse_params("TIC 999999999")

    # 'P_pri' was written with a blank stderr (a fixed, non-varied parameter);
    # ufloat can't hold an uncertainty of None, so it comes back as 0 instead
    assert pri_params["P"].nominal_value == pytest.approx(5.0)
    assert pri_params["P"].std_dev == 0


def test_read_global_eclipse_params_raises_if_never_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        read_global_eclipse_params("TIC 000000000")
