"""Diagnostic and summary plots produced at each pipeline stage.

Every function here saves a figure to disk under :data:`EBP_sweep.config.FIGURES_DIR`
(creating subdirectories as needed) rather than returning a figure object, so
that long batch runs over many targets can be monitored from the output
directory without holding figures in memory.
"""

import os

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from . import config

# Fixed color + marker per timing method, independent of which subset is
# passed in for a given target, so a method's identity on the plot never
# shifts with which other methods happen to be present (colorblind-safe
# categorical set; see EBP_sweep's dataviz palette notes).
METHOD_STYLE = {
    'hd':     dict(color='#2a78d6', marker='o'),
    'fold':   dict(color='#eb6834', marker='o'),
    'cc':     dict(color='#1baf7a', marker='o'),
    'gress':  dict(color='#eda100', marker='o'),
    'batman': dict(color='#e87ba4', marker='o'),
}


def plot_subfigures(tic_id, phased_primary, phased_secondary, phased_secondary_on_primary, 
                    phased_primary_on_secondary, xlim_primary, xlim_secondary, verbose=True):
    """
    Plot the four subfigures for the eclipses phased on primary and secondary periods."""

    fig, ((subfig1, subfig2), (subfig3, subfig4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharey=True)

    unique_sectors = np.unique(np.array(phased_primary.sector))

    original_cmap = plt.cm.viridis

    # Number of colors in the original colormap
    num_colors = original_cmap.N

    start_index = 0.0
    end_index = 0.8

    # Create the sliced colormap
    sliced_colors = original_cmap(np.linspace(start_index, end_index, num=num_colors))
    sliced_cmap = ListedColormap(sliced_colors)

    colors = [sliced_cmap(i) for i in np.linspace(0, 1, 108)]  # Change for increasing number of TESS sectors
    # colors = sliced_cmap(unique_sectors)
    # sector_color_map = {sector: colors[i] for i, sector in enumerate(unique_sectors)}

    # Subfigure 1: Primary Eclipse Phased on Primary Period

    for sector in unique_sectors:

        mask = phased_primary['sector'] == sector

        norm = (sector - min(unique_sectors)) / (max(unique_sectors) - min(unique_sectors))
        alpha_value = 0.1 + 0.9 * (1 - norm)
        size_scaled = 3 - 1 * norm

        subfig1.plot(phased_primary.time.value[mask], phased_primary.flux.value[mask], 'o', color=colors[sector], ms=size_scaled, alpha=alpha_value, label=f"Sector {sector}")

    subfig1.set_title('Primary Eclipse Phased on Primary Period', fontsize=12)
    subfig1.set_xlim(xlim_primary)
    subfig1.set_xticklabels([])
    subfig1.set_ylabel('Normalised Flux', fontsize=12)
    subfig1.tick_params(axis='y', labelsize=12)
    subfig1.grid(True)

    # Subfigure 2: Secondary Eclipse Phased on Primary Period

    for sector in unique_sectors:

        mask = phased_secondary_on_primary['sector'] == sector

        norm = (sector - min(unique_sectors)) / (max(unique_sectors) - min(unique_sectors))
        alpha_value = 0.1 + 0.9 * (1 - norm)
        size_scaled = 3 - 1 * norm

        subfig2.plot(phased_secondary_on_primary.time.value[mask], phased_secondary_on_primary.flux.value[mask], 'o', color=colors[sector], ms=size_scaled, alpha=alpha_value, label=f"Sector {sector}")

    subfig2.set_title('Secondary Eclipse Phased on Primary Period', fontsize=12)
    subfig2.set_xlim(xlim_secondary)
    subfig2.set_xticklabels([])
    subfig2.tick_params(axis='y', labelsize=12)
    subfig2.grid(True)

    # Subfigure 3: Primary Eclipse Phased on Secondary Period

    for sector in unique_sectors:

        mask = phased_primary_on_secondary['sector'] == sector

        norm = (sector - min(unique_sectors)) / (max(unique_sectors) - min(unique_sectors))
        alpha_value = 0.1 + 0.9 * (1 - norm)
        size_scaled = 3 - 1 * norm

        subfig3.plot(phased_primary_on_secondary.time.value[mask], phased_primary_on_secondary.flux.value[mask], 'o', color=colors[sector], ms=size_scaled, alpha=alpha_value, label=f"Sector {sector}")

    subfig3.set_title('Primary Eclipse Phased on Secondary Period', fontsize=12)
    subfig3.set_xlim(xlim_primary)
    subfig3.set_xlabel('Phase', fontsize=12)
    subfig3.set_ylabel('Normalised Flux', fontsize=12)
    subfig3.tick_params(axis='both', labelsize=12)
    subfig3.grid(True)

    # Subfigure 4: Secondary Eclipse Phased on Secondary Period

    for sector in unique_sectors:

        mask = phased_secondary['sector'] == sector

        norm = (sector - min(unique_sectors)) / (max(unique_sectors) - min(unique_sectors))
        alpha_value = 0.1 + 0.9 * (1 - norm)
        size_scaled = 3 - 1 * norm

        subfig4.plot(phased_secondary.time.value[mask], phased_secondary.flux.value[mask], 'o', color=colors[sector], ms=size_scaled, alpha=alpha_value, label=f"Sector {sector}")

    subfig4.set_title('Secondary Eclipse Phased on Secondary Period', fontsize=12)
    subfig4.set_xlim(xlim_secondary)
    subfig4.set_xlabel('Phase', fontsize=12)
    subfig4.tick_params(axis='x', labelsize=12)
    subfig4.grid(True)

    legend_handles = []

    min_sector = min(unique_sectors)
    max_sector = max(unique_sectors)

    for sector in unique_sectors:
        norm = (sector - min_sector) / (max_sector - min_sector)
        size_scaled = 3 - 1 * norm  # SAME formula as plots

        legend_handles.append(
            Line2D(
                [], [],
                linestyle='None',
                marker='o',
                color=colors[sector],
                markersize=size_scaled,  # <-- THIS is the key
                alpha=1,
                label=f"Sector {sector}"
            )
        )

    fig.legend(
        handles=legend_handles,
        fontsize=9,
        markerscale=1,      # <-- prevent automatic scaling
        bbox_to_anchor=(1.11, 0.9)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir = config.fig_path(config.FIG_4PANEL_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_4panel_new.png')
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    if verbose: print(f"\tSaved 4-panel plot to {out_path}")


def plot_best_oc(tic_id, observed_primary_eclipse_times, observed_secondary_eclipse_times, OCs_primary, OCs_secondary, primary_eclipse_err, secondary_eclipse_err):
    """Plot the best O-C diagram with error bars for primary and secondary eclipses."""

    fig = plt.figure(figsize=(8, 5))
    plt.title(f"TIC {tic_id[4:]} O-C Diagram", fontsize=14)
    plt.errorbar(observed_primary_eclipse_times, OCs_primary, yerr=primary_eclipse_err, fmt='o', color='b', ecolor='b', capsize=3, alpha=0.75,
                 markeredgecolor='k', label='Primary Eclipses')
    plt.errorbar(observed_secondary_eclipse_times, OCs_secondary, yerr=secondary_eclipse_err, fmt='o', color='r', ecolor='r', capsize=3,
                 markeredgecolor='k', label='Secondary Eclipses')

    plt.axhline(0, color='gray', linestyle='dashed', linewidth=2)
    plt.xlabel("Time - 2457000 [BTJD days]", fontsize=12)
    plt.ylabel("O - C (minutes)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    out_dir = config.fig_path(config.FIG_OC_DIR)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'TIC{tic_id[4:]}_OCwErr.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_method_comparison(tic_id, obs_pri, obs_sec, err_pri, err_sec, methods,
                            Tref_pri, Tref_sec, Pref, pri_stds, sec_stds, pri_red_chi2, sec_red_chi2,
                            best_index_primary=None, best_index_secondary=None):
    """Compare each timing method's O-C scatter side by side.

    Every method's eclipse times are folded onto the *same* (Tref, Pref)
    reference ephemeris (one row per eclipse type, one column per method),
    rather than each method's own best-fit line. Detrending each method
    separately would absorb any systematic offset between methods into the
    fit and hide it; referencing them all to one ephemeris is what makes a
    timing offset or precision difference between methods visible at all.
    """
    n = len(methods)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), sharey='col', squeeze=False)

    rows = [
        ('Primary', obs_pri, err_pri, Tref_pri, pri_stds, pri_red_chi2, best_index_primary),
        ('Secondary', obs_sec, err_sec, Tref_sec, sec_stds, sec_red_chi2, best_index_secondary),
    ]

    for row, (label, obs_list, err_list, Tref, stds, red_chi2, best_index) in enumerate(rows):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            style = METHOD_STYLE.get(method, dict(color='0.4', marker='o'))

            obs = np.asarray(obs_list[col])
            err_min = np.asarray(err_list[col]) * 24 * 60
            cyc = np.round((obs - obs[0]) / Pref)
            OC = (obs - (obs[0] + Pref * cyc)) * 24 * 60

            ax.errorbar(obs, OC, yerr=err_min, fmt=style['marker'], color=style['color'],
                        ecolor=style['color'], markeredgecolor='k', mew=0.4, ms=5,
                        capsize=2, alpha=0.85, linestyle='none')
            ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)
            if red_chi2[col] is None:
                ax.text(0.05, 0.05, f"$\\sigma$={stds[col]:.2f} min", 
                        transform=ax.transAxes,fontsize=8, color='0.3', va='bottom', ha='left')
            else:
                ax.text(0.05, 0.05, f"$\\sigma$={stds[col]:.2f} min, $\\chi^2_\\mathrm{{red}}$={red_chi2[col]:.2f}", 
                        transform=ax.transAxes,fontsize=8, color='0.3', va='bottom', ha='left')
            ax.grid(alpha=0.3)

            if row == 0:
                ax.set_title(method, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{label}\nO - C (minutes)", fontsize=10)
            if best_index is not None and col == best_index:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(1.8)

    fig.supxlabel("Time - 2457000 [BTJD days]", fontsize=12)
    fig.suptitle(f"TIC {tic_id[4:]} — eclipse-timing method comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = config.fig_path(config.FIG_METHODCOMP_DIR)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'TIC{tic_id[4:]}_method_comparison.png'), dpi=250)
    plt.close(fig)
    print(f"Saved method comparison figure to {os.path.join(out_dir, f'TIC{tic_id[4:]}_method_comparison.png')}")


def save_all_sectors_multipage_pdf(tic_id):
    """Merge the raw and detrended and phasefolded all-sector JPGs into a multi-page PDF."""
    base_path = os.path.join(config.fig_path(config.FIG_ALLSECTORS_DIR), f'TIC{tic_id[4:]}')
    image_paths = [
        f'{base_path}_All_Sectors.jpg',
        f'{base_path}_All_Sectorsdetrended.jpg',
        f'{base_path}_Allphased_TLS.jpg'
    ]

    image_paths = [path for path in image_paths if os.path.exists(path)]

    pdf_path = f'{base_path}_All_Sectors_multipage.pdf'
    images = []
    try:
        for path in image_paths:
            images.append(Image.open(path).convert('RGB'))
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
        print(f"\tCombined raw, detrended, and phased data in multipage PDF: {pdf_path}")
    finally:
        for image in images:
            image.close()
    [os.remove(img) for img in image_paths if os.path.exists(img)]


def plot_all_sectors(tic_id, lc_collection, append_str="", verbose=True):
    """
    Create a subplot image showing time vs flux for all sectors.

    Parameters:
    -----------
    tic_id : str
        TIC ID string (e.g., 'TIC 343127696')
    lc_collection : lightkurve LightCurve
        Stitched light curve data with sector information
    """
    try:
        n_sectors = len(lc_collection)

        # Calculate subplot grid dimensions
        n_cols = 2
        n_rows = (n_sectors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=False, sharey=True)
        axes = np.atleast_2d(axes).flatten()  # Flatten to 1D for easy indexing

        for idx, lc in enumerate(lc_collection):
            ax = axes[idx]
            sector_lc = lc
            sector = sector_lc.sector if isinstance(sector_lc.sector, int) else sector_lc.sector[0]
            if hasattr(lc, 'masked_points'):
                sector_mask = lc.masked_points
            else:
                sector_mask = np.full_like(sector_lc.time.value, False, dtype=bool)

            ax.plot(sector_lc.time.value, sector_lc.flux.value, 'o-', markersize=2, linewidth=0.5)
            ax.plot(sector_lc.time.value[sector_mask], sector_lc.flux.value[sector_mask], 'rx', markersize=3, linewidth=0.5)
            if append_str == "":
                ax.set_title(f'Sector {int(sector)} - masked {np.sum(sector_mask)}/{len(sector_lc.time.value)} points', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'Sector {int(sector)} {append_str}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (BTJD)', fontsize=10)
            ax.set_ylabel('Flux', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_sectors, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{tic_id} - All Sectors Light Curve ({append_str})', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()

        out_dir = config.fig_path(config.FIG_ALLSECTORS_DIR)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_All_Sectors{append_str}.jpg')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if verbose: 
            print(f"\tSaved all sectors plot to {out_path}")

    except Exception as e:
        print(f"Error creating all sectors plot: {e}")


def save_epoch_fits_pdf(tic_id, P, epoch_fits, ecl_type, method, extra_t0s=None):
    """
    Save per-epoch fit subplots into a multipage PDF, 4 columns x 5 rows per page.

    Parameters
    ----------
    tic_id : str
        TIC ID string (e.g., 'TIC 343127696')
    P : float
        Orbital period.
    epoch_fits : list
        List of per-epoch fit results.
    ecl_type : str
        Eclipse type ('pri' or 'sec').
    method : str
        Fitting method name.
    extra_t0s : dict, optional
        Extra t0 values to plot.
    """
    N_COLS, N_ROWS = 4, 5
    per_page = N_COLS * N_ROWS
    n_epochs = len(epoch_fits)
    n_pages = max(1, (n_epochs + per_page - 1) // per_page)

    out_dir = config.fig_path(os.path.join(config.FIG_EPOCHTIMES_DIR, f'TIC{tic_id[4:]}'))
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_EpochFits_{ecl_type}_{method}_P{P:.4f}.pdf')
    cols, lin_sty = ['g', 'b', 'k', 'm'], ['--', ':', '-.', '-']
    xt0s = {}
    if extra_t0s is not None:
        for k in extra_t0s:
            if (len(extra_t0s[k][0]) == n_epochs) and (len(extra_t0s[k][1]) == n_epochs):
                xt0s[k] = extra_t0s[k]
            else:
                print(f"Warning: extra_t0s[{k}] has length {len(extra_t0s[k][0])}, expected {n_epochs} epochs. Not plotting t0s of this method.")

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            end = min(start + per_page, n_epochs)
            n_this = end - start

            fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 15), sharey=True)
            axes = axes.flatten()

            for i, (times, fluxes, model_time, model_flux, t0_fit, t0_err) in enumerate(epoch_fits[start:end]):
                ax = axes[i]
                ax_title = f'{method}_t0={t0_fit:.4f}(+/-{t0_err*24*60:.1f}mins)' if t0_fit is not None else ''
                ax.plot(times, fluxes, 'k.', ms=3, alpha=0.6, label='Data')
                ax.plot(model_time, model_flux, 'r-', lw=1.5, label='Model')
                if t0_fit is not None:
                    ax.axvline(t0_fit, color='c', lw=0.8, ls='-', label=method)

                for j, k in enumerate(xt0s):
                    ax.axvline(xt0s[k][0][start + i], color=cols[j], lw=0.8, ls=lin_sty[j], label=k)
                    ax_title += f'\n{k+"_t0":13s}={xt0s[k][0][start + i]:.4f}(+/-{xt0s[k][1][start + i]*24*60:.1f}mins)'

                ax.set_title(ax_title, fontsize=7)
                ax.tick_params(labelsize=6)
                try:
                    ax.set_xlim([t0_fit - 0.1 * P, t0_fit + 0.1 * P])
                except Exception:
                    pass
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=6, loc='lower right', framealpha=0.05)

            for i in range(n_this, per_page):
                axes[i].set_visible(False)

            fig.suptitle(f'TIC{tic_id[4:]}  —  Epoch fits (P={P:.4f} d)  {method} —  page {page+1}/{n_pages}', fontsize=11)
            fig.subplots_adjust(hspace=0.4, wspace=0.1)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\tSaved epoch fits to {pdf_path}")


def plot_shape_variation(tic_id, pri_shape_params, pri_global_shape_params,
                         sec_shape_params, sec_global_shape_params):
    """Save primary and secondary eclipse-shape variations as a two-page PDF.

    Parameters
    ----------
    tic_id : str
        TIC ID of the target.
    pri_shape_params, sec_shape_params : dict
        Individual primary and secondary eclipse shape parameters (W, D, b).
    pri_global_shape_params, sec_global_shape_params : dict
        Pooled primary and secondary eclipse shape parameters (W, D, b).
    """
    out_dir = config.fig_path(config.FIG_VARIATION_DIR)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f'TIC{tic_id[4:]}_shape_variation.pdf')

    with PdfPages(pdf_path) as pdf:
        _plot_shape_variation_page(
            pri_shape_params, pri_global_shape_params, 'pri', pdf,
        )
        _plot_shape_variation_page(
            sec_shape_params, sec_global_shape_params, 'sec', pdf,
        )


def _plot_shape_variation_page(indv_shape_params, global_shape_params,
                               eclipse_type, pdf):
    """Add one eclipse-type shape-variation figure to an open PDF."""
    W, W_all = indv_shape_params['W'], global_shape_params['W']
    D, D_all = indv_shape_params['D'], global_shape_params['D']
    b, b_all = indv_shape_params['b'], global_shape_params['b']

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax[0].set_title(f"{eclipse_type.capitalize()} Eclipse Width Variation")
    ax[0].errorbar(np.arange(len(W)), [w.n * 24 for w in W], yerr=[w.s * 24 for w in W], fmt='o', ecolor="gray")
    ax[0].axhline(24 * W_all.n, color='b', linestyle='--')
    ax[0].axhspan(24 * (W_all.n - W_all.s), 24 * (W_all.n + W_all.s), color='cyan', alpha=0.2)
    ax[0].set_ylabel("Eclipse width (hours)")
    ax[0].grid(True)

    ax[1].set_title(f"{eclipse_type.capitalize()} Eclipse Depth Variation")
    ax[1].errorbar(np.arange(len(D)), [Dp.n * 100 for Dp in D], yerr=[Dp.s * 100 for Dp in D], fmt='o', ecolor="gray")
    ax[1].axhline(100 * D_all.n, color='b', linestyle='--')
    ax[1].axhspan(100 * (D_all.n - D_all.s), 100 * (D_all.n + D_all.s), color='cyan', alpha=0.2)
    ax[1].set_ylabel("Eclipse depth (%)")
    ax[1].grid(True)

    ax[2].set_title(f"{eclipse_type.capitalize()} Eclipse Impact Parameter Variation")
    ax[2].errorbar(np.arange(len(b)), [bp.n for bp in b], yerr=[bp.s for bp in b], fmt='o', ecolor="gray")
    ax[2].axhline(b_all.n, color='b', linestyle='--')
    ax[2].axhspan(b_all.n - b_all.s, b_all.n + b_all.s, color='cyan', alpha=0.2)
    ax[2].set_xlabel("Eclipse number")
    ax[2].set_ylabel("Impact parameter")
    ax[2].grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
