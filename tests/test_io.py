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


@pytest.fixture
def lc_io(tmp_path, monkeypatch):
    from EBP_sweep import io
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(io, "plot_all_sectors", lambda *args: None)
    return io


def _archive_result(io, curves):
    from astropy.table import Table

    class Search:
        def __init__(self, curves):
            self.curves = curves
            self.table = Table({"sequence_number": [lc.sector for lc in curves]})

        def __getitem__(self, keep):
            return Search([lc for lc, selected in zip(self.curves, keep) if selected])

        def download_all(self, **kwargs):
            return io.lk.LightCurveCollection(self.curves)

    return Search(curves)


def _curve(io, sector=1):
    lc = io.lk.TessLightCurve(time=np.arange(20) * 0.02 + 1500,
                            flux=np.ones(20), flux_err=np.full(20, 0.01),
                            meta={"SECTOR": sector})
    lc["sap_bkg"] = np.ones(20)
    return lc


@pytest.mark.parametrize("author", ["QLP", "SPOC"])
def test_archive_preferred_over_eleanor(lc_io, monkeypatch, author):
    calls = []
    def search(*args, **kwargs):
        calls.append(kwargs["author"])
        return _archive_result(lc_io, [_curve(lc_io)] if kwargs["author"] == author else [])
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", search)
    monkeypatch.setattr(lc_io, "_load_eleanor_lcs", lambda *a: pytest.fail("Unexpected FFI extraction"))
    lc = lc_io.load_and_clean_lc("TIC 123")
    assert len(lc) == 20  # constant background must not discard every cadence
    assert calls == (["QLP"] if author == "QLP" else ["QLP", "SPOC"])


def test_eleanor_fallback_conversion_and_sector_filter(lc_io, monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", lambda *a, **kw: _archive_result(lc_io, [_curve(lc_io, 1)]))
    extracted = []
    def target_data(source, **kwargs):
        extracted.append(source.sector)
        return SimpleNamespace(time=np.array([1500., 1501., 1502., 1503.]),
                               corr_flux=np.array([100., 100., np.nan, 100.]),
                               flux_err=np.ones(4), quality=np.array([0, 1, 0, 0]),
                               flux_bkg=np.ones(4))
    def multi_sectors(**kwargs):
        assert kwargs == {"tic": 123, "sectors": [2, 3]}
        return [SimpleNamespace(sector=s) for s in (1, 2, 3)]
    monkeypatch.setitem(sys.modules, "eleanor", SimpleNamespace(multi_sectors=multi_sectors, TargetData=target_data))
    lc = lc_io.load_and_clean_lc("TIC 123", sector_bounds=(2, 3))
    assert extracted == [2, 3]
    assert list(lc.time.value) == [1500., 1503., 1500., 1503.]
    assert lc.time.format == "btjd" and lc.time.scale == "tdb"
    np.testing.assert_allclose(lc.flux.value, 1.)
    np.testing.assert_allclose(lc.flux_err.value, 0.01)
    assert set(lc["sector"]) == {2, 3}
    assert lc.meta["AUTHOR"] == "eleanor"


@pytest.mark.parametrize("failure", [ImportError("eleanor"), RuntimeError("no FFI coverage")])
def test_eleanor_failure_logged(lc_io, monkeypatch, failure):
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", lambda *a, **kw: _archive_result(lc_io, []))
    def fail(*args):
        raise failure
    monkeypatch.setattr(lc_io, "_load_eleanor_lcs", fail)
    assert lc_io.load_and_clean_lc("TIC 123") is None
    assert os.path.exists(config.data_path(config.ELEANOR_TICIDS_LOG))
    assert str(failure) in open(config.data_path(config.ERRORS_LOG)).read()


def test_failed_search_does_not_trigger_eleanor(lc_io, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("archive unavailable")
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", fail)
    monkeypatch.setattr(lc_io, "_load_eleanor_lcs", lambda *a: pytest.fail("Unexpected extraction"))
    assert lc_io.load_and_clean_lc("TIC 123") is None


def test_eleanor_disabled(lc_io, monkeypatch):
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", lambda *a, **kw: _archive_result(lc_io, []))
    monkeypatch.setattr(lc_io, "_load_eleanor_lcs", lambda *a: pytest.fail("Unexpected extraction"))
    assert lc_io.load_and_clean_lc("TIC 123", eleanor_fallback=False) is None


def test_eleanor_keeps_successful_sectors(lc_io, monkeypatch):
    import sys
    from types import SimpleNamespace
    def extract(source, **kwargs):
        if source.sector == 1:
            raise RuntimeError("sector failed")
        return SimpleNamespace(time=np.array([1500., 1501.]), corr_flux=np.ones(2),
                               flux_err=np.ones(2), quality=np.zeros(2))
    monkeypatch.setitem(sys.modules, "eleanor", SimpleNamespace(
        multi_sectors=lambda **kw: [SimpleNamespace(sector=s) for s in (1, 2)], TargetData=extract))
    monkeypatch.setattr(lc_io.lk, "search_lightcurve", lambda *a, **kw: _archive_result(lc_io, []))
    lc = lc_io.load_and_clean_lc("TIC 123")
    assert len(lc) == 2
    assert set(lc["sector"]) == {2}
    assert "sector failed" in open(config.data_path(config.ERRORS_LOG)).read()
