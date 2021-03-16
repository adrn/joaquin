import numpy as np

from .filters import nufft_lowpass


default_phot_names = [
    'GAIAEDR3_PHOT_G_MEAN_MAG',
    'GAIAEDR3_PHOT_BP_MEAN_MAG',
    'GAIAEDR3_PHOT_RP_MEAN_MAG',
    'J', 'H', 'K',
    'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro'
]


def get_lsf_features(lsf_hdul):
    """BAG O' HACKS"""

    lsf = lsf_hdul[0].data[:, 7]  # Because: What other pixel would you choose?
    pix = np.arange(len(lsf))

    locs = [500, 2500, 4000, 5600, 6600, 7900]
    half_size = 50

    vals = []
    for loc in locs:
        mask = np.abs(pix - loc) < half_size
        vals.append(np.mean(lsf[mask]))

    return np.array(vals)


def get_phot_features(star, phot_names=None):
    if phot_names is None:
        phot_names = default_phot_names

    vals = []
    for name in phot_names:
        vals.append(star[name])
    return np.array(vals)


def get_spec_features(star_hdul):
    pix = np.arange(star_hdul[1].header['NAXIS1'])
    wvln = 10 ** (star_hdul[1].header['CRVAL1'] +
                  pix * star_hdul[1].header['CDELT1'])
    ln_wvln = np.log(wvln)
    flux = star_hdul[1].data
    err = star_hdul[2].data

    mask = (flux <= 0) | (err > (3 * np.median(err))) | (~np.isfinite(flux))
    ln_flux = np.full_like(flux, np.nan)
    ln_flux[~mask] = np.log(flux[~mask])

    new_ln_flux = nufft_lowpass(ln_wvln, ln_flux,
                                fcut=0.5 * 22500,
                                bad_mask=mask)

    return new_ln_flux
