import numpy as np

from .filters import nufft_lowpass
from .config import all_phot_names


def get_lsf_features(lsf):
    pix = np.arange(len(lsf))

    # BAG O' HACKS
    locs = [500, 2500, 4000, 5600, 6600, 7900]  # MAGIC NUMBERs
    half_size = 50

    vals = []
    for loc in locs:
        mask = np.abs(pix - loc) < half_size
        vals.append(np.mean(lsf[mask]))

    return np.array(vals)


def get_phot_features(star):
    vals = []
    for name in all_phot_names:
        vals.append(star[name])
    return np.array(vals)


def get_spec_features(wvln, flux, err, lowpass=True):
    ln_wvln = np.log(wvln)

    mask = ((flux <= 0) |
            (err > (5 * np.median(err))) |  # MAGIC NUMBER
            (err == 0) |
            (~np.isfinite(flux)))
    ln_flux = np.full_like(flux, np.nan)
    ln_flux[~mask] = np.log(flux[~mask])

    if lowpass:
        new_ln_flux = nufft_lowpass(ln_wvln, ln_flux,
                                    fcut=0.5 * 22500,
                                    bad_mask=mask)
    else:
        new_ln_flux = ln_flux
        new_ln_flux[mask] = 0.

    return new_ln_flux, mask
