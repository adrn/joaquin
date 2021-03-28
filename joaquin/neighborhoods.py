import numpy as np
from .plot import phot_to_label


def stretch(x, vmin=None, vmax=None):
    x = np.array(x)

    if vmin is None:
        vmin = np.min(x)

    if vmax is None:
        vmax = np.max(x)

    return (x - vmin) / (vmax - vmin)


def get_neighborhood_X(data, spec_good_mask, color_names=None,
                       apply_stretch=True):
    (X, *_), idx_map = data.get_Xy(['spec'],
                                   spec_good_mask=spec_good_mask)

    color_names = [
        ('GAIAEDR3_PHOT_BP_MEAN_MAG', 'GAIAEDR3_PHOT_RP_MEAN_MAG'),
        ('GAIAEDR3_PHOT_RP_MEAN_MAG', 'w1mpro'),
        ('J', 'K'),
        ('J', 'w1mpro'),
        ('H', 'w2mpro'),
        ('w1mpro', 'w2mpro')
    ]
    color_labels = [f'{phot_to_label[x1]}-{phot_to_label[x2]}'
                    for x1, x2 in color_names]

    color_X = data.get_colors(color_names)

    if apply_stretch:
        for i in range(color_X.shape[1]):
            color_X[:, i] = stretch(
                color_X[:, i],
                *np.nanpercentile(color_X[:, i], [5, 95]))  # MAGIC NUMBERs

    X = np.hstack((X, color_X))

    good_stars = data.stars[data.stars_mask]
    assert X.shape[0] == len(good_stars)

    return X, color_labels, good_stars
