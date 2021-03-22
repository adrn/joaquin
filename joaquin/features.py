import numpy as np

from .filters import nufft_lowpass
from .config import phot_names, dr


def get_lsf_features(lsf_hdul):
    """BAG O' HACKS"""

    if dr == 'dr17':
        lsf = lsf_hdul[0].data[:, 7]  # MAGIC NUMBER
    else:
        lsf = lsf_hdul[1].data[7]  # MAGIC NUMBER
    pix = np.arange(len(lsf))

    locs = [500, 2500, 4000, 5600, 6600, 7900]
    half_size = 50

    vals = []
    for loc in locs:
        mask = np.abs(pix - loc) < half_size
        vals.append(np.mean(lsf[mask]))

    return np.array(vals)


def get_phot_features(star):
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

    mask = ((flux <= 0) |
            (err > (5 * np.median(err))) |  # MAGIC NUMBER
            (err == 0) |
            (~np.isfinite(flux)))
    ln_flux = np.full_like(flux, np.nan)
    ln_flux[~mask] = np.log(flux[~mask])

    new_ln_flux = nufft_lowpass(ln_wvln, ln_flux,
                                fcut=0.5 * 22500,
                                bad_mask=mask)

    return new_ln_flux, mask
