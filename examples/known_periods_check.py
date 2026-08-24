"""Sanity-check the TLS period search against a set of literature periods.

Runs only the light-curve download and period-finding stages (skipping
flattening / eclipse separation / timing) for a list of targets with known
published periods, and reports how closely the TLS period search recovers
them. Useful as a quick regression check when tuning
``EBP_sweep.periods.find_orbital_period``.

    python examples/known_periods_check.py
"""

import csv

from EBP_sweep.io import load_and_clean_lc
from EBP_sweep.periods import find_orbital_period

# TIC ID -> published orbital period (days).
TIC_PERIOD_PAPER = {
    'TIC 81741369': 10.0920,
    'TIC 293225466': 129.3214,
    'TIC 274182408': 6.554,
    'TIC 457579488': 14.6925,
    'TIC 124282654': 11.3293,
    'TIC 286310830': 8.3687,
    'TIC 51629874': 6.465,
    'TIC 123951716': 23.1065,
    'TIC 396170777': 6.5656,
    'TIC 139699256': 5.952,
    'TIC 56023695': 8.177,
    'TIC 292357653': 23.1476,
    'TIC 82893635': 28.5504,
    'TIC 196989952': 7.281,
    'TIC 146204045': 14.192,
    'TIC 283651681': 7.62,
    'TIC 61656788': 4.3028,
    'TIC 444544588': 16.349,
    'TIC 252497283': 5.04,
    'TIC 167699456': 18.74,
    'TIC 343127696': 4.672,
    'TIC 253270207': 6.11,
    'TIC 158330804': 11.355,
    'TIC 115396972': 19.809,
    'TIC 399127035': 5.448,
    'TIC 355503224': 6.9247,
    'TIC 95622298': 4.5123,
    'TIC 64366964': 4.459,
    'TIC 198242678': 4.6298,
    'TIC 84546771': 4.2837,
    'TIC 291751499': 5.1957,
    'TIC 167756615': 19.181,
    'TIC 441496809': 5.7542,
    'TIC 165615442': 5.475,
    'TIC 121092916': 8.2198,
    'TIC 342356517': 11.9666,
}


def main():
    quality_bitmask = "hardest"  # 'none', 'default', 'hard', or 'hardest'
    tls_results = {}

    for tic_id, paper_period in TIC_PERIOD_PAPER.items():
        try:
            good_lc = load_and_clean_lc(tic_id, quality_bitmask)
            P_TLS, _ = find_orbital_period(tic_id, good_lc)
            tls_results[tic_id] = P_TLS
            print(f"{tic_id}: TLS={P_TLS}, paper={paper_period:.2f}")
        except Exception as e:
            print(f"{tic_id}: failed — {e}")
            tls_results[tic_id] = None

    with open("test_period.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TIC ID", "TLS Period", "Paper Period"])
        for tic_id, period in tls_results.items():
            writer.writerow([tic_id, period, TIC_PERIOD_PAPER.get(tic_id)])

    print(f"\n{'TIC ID':20s} {'TLS Period':10s} {'Paper Period':12s}")
    print("-" * 45)
    for tic_id, period in tls_results.items():
        period_str = f"{period:10.2f}" if period is not None else f"{'n/a':>10s}"
        print(f"{tic_id:20s} {period_str} {TIC_PERIOD_PAPER[tic_id]:12.2f}")


if __name__ == "__main__":
    main()
