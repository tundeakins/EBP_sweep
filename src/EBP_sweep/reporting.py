"""PDF data-validation summaries for a completed target."""

import csv
import os

import numpy as np
from astroquery.mast import Catalogs
from fpdf import FPDF

from . import config
from .utils import format_ranges


def make_pdf(tic_id, P_primary, primP_err, P_secondary, secP_err, flat_lc, primary_method, secondary_method, primary_scatter, secondary_scatter):
    """
    Generate a PDF summary for a given TIC ID with primary and secondary period information.

    Parameters
    -----------
    tic_id: str
        The TIC ID of the target.
    P_primary: float
        The primary period of the binary system.
    primP_err: float
        The error in the primary period.
    P_secondary: float
        The secondary period of the binary system.
    secP_err: float
        The error in the secondary period.
    flat_lc: LightCurve
        The flattened TESS light curve object.
    primary_method: str
        The method used to determine the primary T0.
    secondary_method: str
        The method used to determine the secondary T0.
    primary_scatter: float
        The scatter in the primary eclipse timing.
    secondary_scatter: float
        The scatter in the secondary eclipse timing.

    Returns
    -------
    None
    """
    # Set up Page

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', '', 16)

    # Layout contents

    # title
    pdf.cell(40, 10, f'{tic_id}')

    # figure 1
    oc_plot = os.path.join(config.fig_path(config.FIG_OC_DIR), f'TIC{tic_id[4:]}_OCwErr.png')
    pdf.image(oc_plot, x=1, y=20, w=150)

    # info block

    catalog_data = Catalogs.query_object(f"{tic_id}", catalog="TIC")
    ra = catalog_data[0]['ra']
    dec = catalog_data[0]['dec']

    P_avg = (P_primary + P_secondary) / 2
    per_diff = round(((abs(P_primary - P_secondary) / P_avg)) * 100, 7)

    text_lines = [
        f"Primary Period: {round(P_primary, 7)} days",
        f"Secondary Period: {round(P_secondary, 7)} days",
        f"Period Percent Difference: {per_diff:.2e}%",
        f"Number of Sectors: {len(np.unique(flat_lc.sector).value)}",
        f"Sectors: {format_ranges(np.unique(flat_lc.sector).value)}",
        f"RA: {ra}",
        f"Dec: {dec}",
        f"Primary T0 Method: {primary_method}",
        f"Secondary T0 Method: {secondary_method}",
        f"Primary Scatter: {round(primary_scatter, 10)} min",
        f"Secondary Scatter: {round(secondary_scatter, 10)} min"
    ]
    block_text = "\n".join(text_lines)

    pdf.set_font('Times', '', 10)
    pdf.set_xy(142, 32)  # set Y as needed
    pdf.multi_cell(w=61, h=8, txt=block_text, border=1, align='L')

    # figure 2
    panel_plot = os.path.join(config.fig_path(config.FIG_4PANEL_DIR), f'TIC{tic_id[4:]}_4panel_new.png')
    pdf.image(panel_plot, x=5, y=140, w=200)

    # Save Info to CSV
    os.makedirs(config.data_path(), exist_ok=True)
    with open(config.data_path(config.GOOD_PERIODS_LOG), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([tic_id, P_primary, primP_err, P_secondary, secP_err, per_diff])

    # Output File
    out_dir = config.fig_path(config.FIG_DVSUMMARY_DIR)
    os.makedirs(out_dir, exist_ok=True)
    pdf.output(os.path.join(out_dir, f'TIC{tic_id[4:]}_dv_wErr.pdf'), 'F')


def create_DV_summary_pdf(tic_id,
                           P_pri_new, primP_err,
                           P_sec_new, secP_err,
                           flat_lc,
                           primary_method, secondary_method,
                           pri_std, sec_std):
    """Create a summary PDF for the target with all relevant plots and information."""
    # this can fail due to connection error. retry a few times before giving up
    for attempt in range(3):
        try:
            make_pdf(tic_id, P_pri_new, primP_err, P_sec_new, secP_err,
                     flat_lc, primary_method, secondary_method, pri_std, sec_std)
            print(f"create_DV_summary_pdf: Successfully created PDF for {tic_id}")
            break  # success
        except Exception as e:
            print(f"create_DV_summary_pdf: Attempt {attempt + 1} failed for {tic_id}: {e}")
            if attempt == 2:
                print(f"create_DV_summary_pdf: Failed to create PDF for {tic_id} after 3 attempts.")
    # make_pdf(tic_id, P_pri_new, primP_err, P_sec_new, secP_err, flat_lc, primary_method, secondary_method, pri_std, sec_std)
