import numpy as np
import pytest

batman = pytest.importorskip("batman")

from EBP_sweep.batman_fit import Tdur_to_aR, batman_flux_model, make_transit_params


def test_make_transit_params_round_trips_basic_fields():
    params = make_transit_params(t0=1.0, period=3.0, rp=0.1, dur=0.15, b=0.2, u=(0.3, 0.2))
    assert params.t0 == 1.0
    assert params.per == 3.0
    assert params.rp == 0.1
    assert params.limb_dark == "quadratic"
    assert params.u == [0.3, 0.2]
    # a central impact parameter (b=0) should give an inclination near 90 deg
    edge_on = make_transit_params(b=0.0)
    assert edge_on.inc == pytest.approx(90.0, abs=1e-6)


def test_tdur_to_ar_increases_scaled_semimajor_axis_for_shorter_duration():
    aR_long = Tdur_to_aR(Tdur=0.3, b=0.0, Rp=0.1, P=3.0)
    aR_short = Tdur_to_aR(Tdur=0.1, b=0.0, Rp=0.1, P=3.0)
    # a shorter transit duration at fixed period implies a larger a/R*
    assert aR_short > aR_long


def test_batman_flux_model_dips_at_mid_eclipse():
    t0, period = 0.0, 2.0
    params = make_transit_params(t0=t0, period=period, rp=0.15, dur=0.1, b=0.0)
    time = np.linspace(-0.5, 0.5, 2001)
    model = batman.TransitModel(params, time)

    param_values = dict(t0=t0, P=period, rp=0.15, dur=0.1, b=0.0, u1=0.3, u2=0.2,
                         Aev=0.0, offset=0.0, slope=0.0)
    flux = batman_flux_model(model, param_values)

    out_of_eclipse = flux[np.abs(time) > 0.2]
    assert flux[np.argmin(np.abs(time))] < out_of_eclipse.min()
