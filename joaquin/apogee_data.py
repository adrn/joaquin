# Third-party
from astropy.io import fits
import numpy as np
from tqdm.auto import tqdm

# Joaquin
from .logger import logger
from .features import get_phot_features, get_lsf_features, get_spec_features


def get_aspcapstar_path(config, star):
    filename = f"aspcapStar-{config.apogee_reduction}-{star['APOGEE_ID']}.fits"
    local_path = (config.apogee_cache_path /
                  config.apogee_dr /
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)
    return local_path


def get_aspcapstar(config, star):
    local_path = get_aspcapstar_path(config, star)
    with fits.open(local_path) as hdul:
        pix = np.arange(hdul[1].header['NAXIS1'])
        wvln = 10 ** (hdul[1].header['CRVAL1'] +
                      pix * hdul[1].header['CDELT1'])
        flux = hdul[1].data
        err = hdul[2].data

    return wvln, flux, err


def get_lsf_path(config, star):
    if star['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif star['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    filename = f"a{sorp}StarLSF-{star['APOGEE_ID']}.fits"
    local_path = (config.apogee_cache_path /
                  config.apogee_dr /
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)

    return local_path


def get_lsf(config, star):
    local_path = get_lsf_path(config, star)
    with fits.open(local_path) as hdul:
        if config.apogee_dr == 'dr17':
            lsf = hdul[0].data[:, 7]
        else:
            lsf = hdul[1].data[7]

    pix = np.arange(len(lsf))

    return pix, lsf


def make_apogee_X(config, stars, progress=True, X_dtype=np.float32,
                  spec_fill_value=0.):

    if progress:
        iter_ = tqdm
    else:
        iter_ = iter

    if stars is None:
        raise ValueError(
            "Input `stars` is None! You must pass a table of allStar data "
            "using the `stars` argument to the initializer")

    # First, figure out how many features we have:
    for star in stars:
        try:
            wvln, flux, err = get_aspcapstar(config, star)
            pix, lsf = get_lsf(config, star)

            phot_f = get_phot_features(star, config.phot_names)
            lsf_f = get_lsf_features(lsf)
            spec_f, mask = get_spec_features(wvln, flux, err,
                                             fill_value=spec_fill_value)

        except Exception:  # noqa
            continue

        Nlsf = len(lsf_f)
        Nphot = len(phot_f)
        Nspec = len(spec_f)

        break

    else:
        raise RuntimeError("Failed to determine number of features")

    Nstars = len(stars)
    Nfeatures = Nphot + Nlsf + Nspec
    X = np.full((Nstars, Nfeatures), np.nan, dtype=X_dtype)
    spec_bad_masks = np.full((Nstars, Nspec), True, dtype=bool)
    for i, star in enumerate(iter_(stars)):
        try:
            wvln, flux, err = get_aspcapstar(config, star)
            pix, lsf = get_lsf(config, star)
        except Exception as e:
            logger.log(1,
                       "failed to get aspcapStar or apStarLSF data for "
                       f"star {i}\n{e}")
            continue

        try:
            phot_f = get_phot_features(star, config.phot_names)
            lsf_f = get_lsf_features(lsf)
            spec_f, spec_mask = get_spec_features(wvln, flux, err,
                                                  fill_value=spec_fill_value)
        except Exception as e:
            logger.log(1, f"failed to get features for star {i}\n{e}")
            continue

        phot_idx = np.arange(Nphot, dtype=int)

        last = phot_idx[-1] + 1
        lsf_idx = np.arange(last, last + Nlsf, dtype=int)

        last = lsf_idx[-1] + 1
        spec_idx = np.arange(last, last + Nspec, dtype=int)

        X[i] = np.concatenate((phot_f, lsf_f, spec_f))
        spec_bad_masks[i] = spec_mask

    idx_map = {
        'phot': phot_idx,
        'lsf': lsf_idx,
        'spec': spec_idx
    }

    return X, idx_map, wvln, spec_bad_masks
