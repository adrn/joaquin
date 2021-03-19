import numpy as np
from .config import phot_names


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
}


def get_color_mask(stars):
    G_J = stars['GAIAEDR3_PHOT_G_MEAN_MAG'] - stars['J']
    J_K = stars['J'] - stars['K']

    poly = np.poly1d(np.polyfit(G_J, J_K, deg=1))
    dcolor = J_K - poly(G_J)

    # MAGIC NUMBERS: see Parent-sample-cleaning.ipynb
    dcolor_mask = np.abs(dcolor - np.median(dcolor)) < 6 * np.std(dcolor)
    dcolor_mask &= (clean_stars['H'] - clean_stars['w2mpro']) > -0.5
    dcolor_mask &= (clean_stars['w1mpro'] - clean_stars['w2mpro']) > -1

    return dcolor_mask


def get_parent_sample(stars):
    """
    Many magic numbers set heuristically in Parent-sample-cleaning.ipynb
    """
    phot_mask = np.isin(stars['TELESCOPE'], ['apo25m', 'lco25m'])

    phot_mask = np.ones(len(stars), dtype=bool)
    for name in phot_names:
        phot_mask &= ((np.isfinite(stars[name]) &
                      (stars[name] > 0) &
                      (stars[name] < 22))  # MAGIC NUMBER

    # TODO: this assumes 2MASS photometry is in there...
    for band in ['J', 'H', 'K']:
        phot_mask &= ((stars[f'{band}_ERR'] > 0) &
                      (stars[f'{band}_ERR'] < 0.1))

    # TODO: this assumes WISE photometry is in there...
    phot_mask &= np.char.startswith(stars['ph_qual'].astype(str), 'AA')

    # clean bad photometry from color-color selections:
    colcol_mask = get_color_mask(stars)

    # require SNR > 40
    snr_mask = stars['SNR'] > 40

    # TODO: remove binaries?

    return stars[phot_mask & colcol_mask]
