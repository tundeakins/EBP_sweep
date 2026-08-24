"""Run the pipeline for a single TIC ID, step by step.

This mirrors working through the pipeline stage by stage in a notebook —
useful for inspecting intermediate products (the cleaned light curve, the
measured period, the separated eclipses, ...) before moving to the next
stage. For a one-line run, see ``run_target`` at the bottom, or
``batch_targets.py`` for processing several targets in a loop.

Run from the repository root with the `ellc_env`-equivalent environment
active (see the README for the dependency list):

    python examples/single_target.py
"""

from EBP_sweep import periods, plotting, reporting, timing
from EBP_sweep.io import load_and_clean_lc
from EBP_sweep.pipeline import EclipsingBinaryTarget, run_target

TIC_ID = "TIC 343127696"  # Example TIC ID; replace with the desired target
QUALITY_BITMASK = "hard"  # 'none', 'default', 'hard', or 'hardest'


def run_step_by_step():
    # 1. Download, stitch, and clean the light curve.
    good_lc = load_and_clean_lc(TIC_ID, QUALITY_BITMASK)

    # 2. Find and refine the orbital period with BLS + TLS.
    P_TLS, results_TLS = periods.find_orbital_period(TIC_ID, good_lc)
    print(f"Found orbital period: {P_TLS} days")

    # 3. Flatten the light curve sector by sector and phase-fold it.
    flat_lc, t0, phased, o_factor = periods.prepare_flat_lc(TIC_ID, good_lc, P_TLS, results_TLS)

    # 4. Separate primary and secondary eclipses.
    ecl = periods.separate_eclipses(TIC_ID, flat_lc, phased, P_TLS, t0, o_factor)

    plotting.plot_subfigures(
        TIC_ID,
        ecl["phased_primary"], ecl["phased_secondary"],
        ecl["phased_secondary_on_primary"], ecl["phased_primary_on_secondary"],
        ecl["primary_xlims"], ecl["secondary_xlims"],
    )

    # 5. Compute eclipse times with all five methods (HD, fold, CC, gress, batman).
    (obs_pri, obs_sec, err_pri, err_sec), methods = timing.compute_eclipse_times(
        TIC_ID, ecl, epoch_width=0.2, methods=["hd", "fold", "cc", "gress", "batman"])

    # 6. Compute O-C values and pick the best-scatter method per eclipse type.
    oc = timing.compute_oc_and_best_period(
        TIC_ID, obs_pri, obs_sec, err_pri, err_sec,
        ecl["P_primary"], ecl["P_secondary"], methods)

    bi_pri, bi_sec = oc["best_index_primary"], oc["best_index_secondary"]
    primary_method, secondary_method = methods[bi_pri], methods[bi_sec]

    plotting.plot_best_oc(
        TIC_ID,
        obs_pri[bi_pri], obs_sec[bi_sec],
        oc["OC_pri_corr"], oc["OC_sec_corr"],
        [e * 24 * 60 for e in err_pri[bi_pri]],
        [e * 24 * 60 for e in err_sec[bi_sec]],
    )

    reporting.create_DV_summary_pdf(
        TIC_ID,
        oc["P_primary_new"], oc["pri_uncerts"][bi_pri],
        oc["P_secondary_new"], oc["sec_uncerts"][bi_sec],
        flat_lc,
        primary_method, secondary_method,
        oc["pri_stds"][bi_pri], oc["sec_stds"][bi_sec],
    )

    print(oc["P_primary_new"], oc["P_secondary_new"])


def run_with_the_class():
    """Same result as `run_step_by_step`, using the stateful pipeline object."""
    target = EclipsingBinaryTarget(TIC_ID, quality_bitmask=QUALITY_BITMASK)
    target.download()
    target.find_period()
    target.flatten()
    target.separate_eclipses()
    target.compute_timing()
    target.compute_oc()
    target.plot_diagnostics()
    target.save_summary()
    print(target.oc["P_primary_new"], target.oc["P_secondary_new"])
    return target


def run_one_liner():
    """Same result again, in one call."""
    return run_target(TIC_ID, quality_bitmask=QUALITY_BITMASK)


if __name__ == "__main__":
    run_step_by_step()
