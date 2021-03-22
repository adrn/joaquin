from astropy.io import fits

from .config import APOGEE_CACHE_PATH, dr, reduction


def get_aspcapstar(star):
    filename = f"aspcapStar-{reduction}-{star['APOGEE_ID']}.fits"
    local_path = (APOGEE_CACHE_PATH /
                  dr /
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)
    return fits.open(local_path)


def get_lsf(star):
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

    return fits.open(local_path)
