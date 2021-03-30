from astropy.io import fits
import numpy as np

from .config import APOGEE_CACHE_PATH, dr, reduction


def get_aspcapstar_path(star):
    filename = f"aspcapStar-{reduction}-{star['APOGEE_ID']}.fits"
    local_path = (APOGEE_CACHE_PATH /
                  dr /
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)
    return local_path


def get_aspcapstar(star):
    local_path = get_aspcapstar_path(star)
    with fits.open(local_path) as hdul:
        pix = np.arange(hdul[1].header['NAXIS1'])
        wvln = 10 ** (hdul[1].header['CRVAL1'] +
                      pix * hdul[1].header['CDELT1'])
        flux = hdul[1].data
        err = hdul[2].data

    return wvln, flux, err


def get_lsf_path(star):
    if star['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif star['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    filename = f"a{sorp}StarLSF-{star['APOGEE_ID']}.fits"
    local_path = (APOGEE_CACHE_PATH /
                  dr /
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)

    return local_path


def get_lsf(star):
    local_path = get_lsf_path(star)
    with fits.open(local_path) as hdul:
        if dr == 'dr17':
            lsf = hdul[0].data[:, 7]
        else:
            lsf = hdul[1].data[7]

    pix = np.arange(len(lsf))

    return pix, lsf
