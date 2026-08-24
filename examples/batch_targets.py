"""Run the full pipeline for several targets in a loop.

For batch processing many targets, ``run_many`` handles per-target failures
gracefully: if a target is flagged for manual inspection at some stage (too
few eclipses, ambiguous period, etc.) it is logged to a CSV file under
``EBP_sweep.config.DATA_DIR`` and the loop moves on to the next target.

    python examples/batch_targets.py
"""

from EBP_sweep.pipeline import run_many

TIC_LIST = ["TIC 343127696", "TIC 81741369"]  # Replace with actual TIC IDs


def main():
    results = run_many(TIC_LIST, quality_bitmask="default")

    for tic_id, target in results.items():
        if target is not None and target.oc is not None:
            print(f"{tic_id}: P_primary={target.oc['P_primary_new']:.6f} d, "
                  f"P_secondary={target.oc['P_secondary_new']:.6f} d")
        else:
            print(f"{tic_id}: failed — see the CSV logs under Data/ for the reason")


if __name__ == "__main__":
    main()
