"""Fit ground-based follow-up photometry using a TESS-derived eclipse shape.

Workflow:

1. Run the TESS pipeline for a target up through eclipse separation.
2. Fit the pooled TESS eclipses with the batman model to get a precise
   global eclipse shape (depth, duration, impact parameter) per eclipse type.
3. Load each night of ground-based photometry (e.g. an AAVSO/WebObs export
   of magnitudes) as barycentric-corrected relative flux.
4. Fit each night against the TESS-derived shape, varying only the
   mid-eclipse time and a small linear baseline.

Adjust TIC_ID, the target coordinates, the observatory site name, and the
photometry file paths / predicted eclipse times for your own target before
running this.

    python examples/ground_based_followup.py
"""

from EBP_sweep import followup, periods, plotting, timing
from EBP_sweep.io import load_and_clean_lc

TIC_ID = "TIC 343127696"
QUALITY_BITMASK = "hard"

# Target coordinates and observatory, for the barycentric time correction.
TARGET_RA, TARGET_DEC = 334.52585, 55.90433  # degrees
OBSERVATORY_SITE = "Roque de los Muchachos, La Palma"

# One entry per night of ground-based photometry: file path, predicted
# mid-eclipse time (BJD_TDB), whether it's a primary ('pri') or secondary
# ('sec') eclipse, and an optional flux floor to normalise partial nights
# onto a common baseline (see EBP_sweep.followup.load_ground_based_photometry).
NIGHTS = [
    dict(label="2026-07-28", path="../MARVEL_data/TIC_343127696_V0743_Cep_webobs_20260728.txt",
         ecl_type="pri", T_pred=2461249.5, err_scale=2.0, flux_floor=0.8),
    dict(label="2026-08-11", path="../MARVEL_data/TIC_343127696_V0743_Cep_webobs_20260811.txt",
         ecl_type="pri", T_pred=2461263.53, err_scale=1.0, flux_floor=0.8),
    dict(label="2026-08-12", path="../MARVEL_data/TIC_343127696_V0743_Cep_webobs_20260812.txt",
         ecl_type="sec", T_pred=2461265.75, err_scale=1.0, flux_floor=None),
]


def main():
    # 1-2. Run the TESS pipeline and get the global eclipse shape per eclipse type.
    good_lc = load_and_clean_lc(TIC_ID, QUALITY_BITMASK)
    P_TLS, results_TLS = periods.find_orbital_period(TIC_ID, good_lc)
    flat_lc, t0, phased, o_factor = periods.prepare_flat_lc(TIC_ID, good_lc, P_TLS, results_TLS)
    ecl = periods.separate_eclipses(TIC_ID, flat_lc, phased, P_TLS, t0, o_factor)

    pri_shape, sec_shape = timing.compute_eclipse_times(
        TIC_ID, ecl, epoch_width=0.2, methods="batman", return_global_eclipse_params=True)
    followup.add_radius_ratio(pri_shape)
    followup.add_radius_ratio(sec_shape)
    shape_by_type = dict(pri=pri_shape, sec=sec_shape)

    # 3-4. Load and fit each night of ground-based photometry.
    fits = []
    for night in NIGHTS:
        time, flux, flux_err = followup.load_ground_based_photometry(
            night["path"], TARGET_RA, TARGET_DEC, OBSERVATORY_SITE,
            err_scale=night["err_scale"], flux_floor=night["flux_floor"])

        shape_params = shape_by_type[night["ecl_type"]]
        fit_params, smooth_time, bestfit_model = followup.run_fit(
            time, flux, flux_err, shape_params, T_pred=night["T_pred"])

        t0_fit = fit_params["t0"].value
        t0_err = fit_params["t0"].stderr
        print(f"{night['label']}: t0 = {t0_fit:.5f} +/- {t0_err:.5f} BJD_TDB")

        fits.append(dict(label=night["label"], time=time, flux=flux, flux_err=flux_err,
                          model_time=smooth_time, model_flux=bestfit_model,
                          t0=t0_fit, t0_err=t0_err))

    fig = plotting.plot_followup_fits(TIC_ID, fits)
    fig.savefig(f"followup_fits_TIC{TIC_ID[4:]}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
