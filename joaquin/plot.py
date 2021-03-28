import matplotlib.pyplot as plt
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
