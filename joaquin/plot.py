import warnings

import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

phot_to_label = {
    'GAIAEDR3_PHOT_BP_MEAN_MAG': 'BP',
    'GAIAEDR3_PHOT_RP_MEAN_MAG': 'RP',
    'GAIAEDR3_PHOT_G_MEAN_MAG': 'G',
    'phot_bp_mean_mag': 'BP',
    'phot_rp_mean_mag': 'RP',
    'phot_g_mean_mag': 'G',
    'w1mpro': 'W1',
    'w2mpro': 'W2',
    'w3mpro': 'W3',
    'w4mpro': 'W4',
    'J': 'J',
    'H': 'H',
    'K': 'K'
}


def simple_corner(X, labels=None, color_by=None, axes=None,
                  colorbar=False, **style):
    if X.shape[1] > X.shape[0]:
        raise ValueError("I don't believe you")

    if color_by is None:
        plotfunc = 'plot'
        style.setdefault('marker', 'o')
        style.setdefault('mew', style.pop('markeredgewidth', 0))
        style.setdefault('ls', style.pop('linestyle', 'none'))
        style.setdefault('ms', style.pop('markersize', 2.))
    else:
        plotfunc = 'scatter'
        style.setdefault('marker', 'o')
        style.setdefault('lw', style.pop('linewidth', 0))
        style.setdefault('s', 5)
        style.setdefault('c', color_by)

    nside = X.shape[1] - 1

    # Some magic numbers for pretty axis layout.
    # Stolen from corner.py!
    K = X.shape[1]
    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    if axes is None:
        fig, axes = plt.subplots(nside, nside,
                                 figsize=(dim, dim),  # (3*nside, 3*nside),
                                 sharex='col', sharey='row',
                                 constrained_layout=True)
    else:
        fig = axes.flat[0].figure

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    cs = None
    for i in range(nside):
        for j in range(nside):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
            else:
                cs = getattr(ax, plotfunc)(X[:, j], X[:, i+1], **style)

    if labels is not None:
        for i in range(nside):
            axes[i, 0].set_ylabel(labels[i+1])

        for j in range(nside):
            axes[-1, j].set_xlabel(labels[j])

    return_stuff = [fig, axes]

    if colorbar and color_by is not None and cs is not None:
        cb = fig.colorbar(cs, ax=axes)
        return_stuff.append(cb)

    return return_stuff


def plot_hr_cmd(parent_stars, stars,
                idx0, other_idx,
                teff_logg_bins=None, cmd_bins=None):

    style_main = dict(ls='none', marker='o', mew=0.6, ms=6.,
                      color='tab:blue', zorder=100,
                      mec='gold')
    style_neighbors = dict(ls='none', marker='o', mew=0, ms=2.,
                           alpha=0.75, color='tab:orange', zorder=10)

    if teff_logg_bins is None:
        teff_logg_bins = (
            np.linspace(3000, 9000, 128),
            np.linspace(-0.5, 5.75, 128))

    if cmd_bins is None:
        cmd_bins = (np.linspace(-0.5, 2, 128),
                    np.linspace(-6, 10, 128))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    ax.hist2d(parent_stars['TEFF'], parent_stars['LOGG'],
              bins=teff_logg_bins, norm=mpl.colors.LogNorm(),
              cmap='Greys')

    ax.plot(stars['TEFF'][idx0],
            stars['LOGG'][idx0],
            **style_main)

    ax.plot(stars['TEFF'][other_idx],
            stars['LOGG'][other_idx],
            **style_neighbors)

    ax.set_xlim(teff_logg_bins[0].max(),
                teff_logg_bins[0].min())
    ax.set_ylim(teff_logg_bins[1].max(),
                teff_logg_bins[1].min())

    ax.set_xlabel(r'$T_{\rm eff}$')
    ax.set_ylabel(r'$\log g$')

    # ---

    ax = axes[1]

    color = ('J', 'K')
    mag = 'H'

    dist_mask, = np.where((parent_stars['GAIAEDR3_PARALLAX'] /
                           parent_stars['GAIAEDR3_PARALLAX_ERROR']) > 5)
    plx = parent_stars['GAIAEDR3_PARALLAX'][dist_mask] * u.mas
    distmod = coord.Distance(
        parallax=plx).distmod.value
    ax.hist2d((parent_stars[color[0]] - parent_stars[color[1]])[dist_mask],
              parent_stars[mag][dist_mask] - distmod,
              bins=cmd_bins, norm=mpl.colors.LogNorm(),
              cmap='Greys')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distmod = coord.Distance(parallax=stars['GAIAEDR3_PARALLAX']*u.mas,
                                 allow_negative=True).distmod.value
    ax.plot((stars[color[0]] - stars[color[1]])[idx0],
            (stars[mag] - distmod)[idx0],
            **style_main)

    ax.plot((stars[color[0]] - stars[color[1]])[other_idx],
            (stars[mag] - distmod)[other_idx],
            **style_neighbors)

    ax.set_xlim(cmd_bins[0].min(), cmd_bins[0].max())
    ax.set_ylim(cmd_bins[1].max(), cmd_bins[1].min())

    ax.set_xlabel('$J - K$')
    ax.set_ylabel('$M_H$')

    return fig
