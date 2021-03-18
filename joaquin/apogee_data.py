from astropy.io import fits

from .config import dr, reduction, stars_, cache_path
from .data_helpers import download_file


SAS_URL = "https://data.sdss.org/sas/"


def get_aspcapstar(star, overwrite=False):
    filename = f"aspcapStar-{dr}-{star['APOGEE_ID']}.fits"
    local_path = cache_path / 'aspcap' / star['FIELD'].strip() / filename
    local_path.parent.mkdir(exist_ok=True, parents=True)

    url = (SAS_URL +
           f"apogeework/apogee/spectro/aspcap/{dr}/{reduction}/" +
           f"{star['TELESCOPE'].strip()}/" +
           f"{star['FIELD'].strip()}/" +
           filename)

    download_file(url, local_path, overwrite=overwrite)

    return fits.open(local_path)


def get_lsf(star, overwrite=False):
    if star['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif star['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    filename = f"a{sorp}StarLSF-{star['APOGEE_ID']}.fits"
    local_path = cache_path / 'aspcap' / star['FIELD'].strip() / filename
    local_path.parent.mkdir(exist_ok=True, parents=True)

    url = (SAS_URL +
           f"apogeework/apogee/spectro/redux/{dr}/{stars_}/" +
           f"{star['TELESCOPE'].strip()}/" +
           f"{star['FIELD'].strip()}/" +
           filename)

    download_file(url, local_path, overwrite=overwrite)

    return fits.open(local_path)
