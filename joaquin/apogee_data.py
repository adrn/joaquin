from astropy.io import fits

from .config import APOGEE_CACHE_PATH, dr


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


def get_aspcapstar(star):
    filename = f"aspcapStar-{dr}-{star['APOGEE_ID']}.fits"
    local_path = (APOGEE_CACHE_PATH /
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
                  star['TELESCOPE'] /
                  star['FIELD'].strip() /
                  filename)

    return fits.open(local_path)
