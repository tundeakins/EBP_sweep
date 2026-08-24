"""Command-line entry point: ``ebp-sweep run TIC_ID [TIC_ID ...]``."""

import argparse

from .pipeline import run_many


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ebp-sweep",
        description="Measure eclipse timing variations for one or more TESS eclipsing binaries.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline for one or more TIC IDs.")
    run_parser.add_argument("tic_ids", nargs="+", help="TIC ID(s), e.g. 'TIC 343127696' (quote each one).")
    run_parser.add_argument("--quality-bitmask", default="default",
                             choices=["none", "default", "hard", "hardest"],
                             help="lightkurve quality bitmask controlling which cadences are discarded (default: %(default)s).")
    run_parser.add_argument("--mask-outliers", action="store_true",
                             help="Mask outlier points during light-curve cleaning.")
    run_parser.add_argument("--no-plots", action="store_true", help="Skip saving diagnostic plots.")
    run_parser.add_argument("--no-summary", action="store_true", help="Skip saving the PDF summary.")

    args = parser.parse_args(argv)

    if args.command == "run":
        results = run_many(
            args.tic_ids,
            quality_bitmask=args.quality_bitmask,
            mask_outliers=args.mask_outliers,
            make_plots=not args.no_plots,
            make_summary=not args.no_summary,
        )
        n_ok = sum(1 for t in results.values() if t is not None and t.oc is not None)
        print(f"\nCompleted {n_ok}/{len(args.tic_ids)} target(s) successfully.")


if __name__ == "__main__":
    main()
