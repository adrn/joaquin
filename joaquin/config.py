import os
import pathlib

dr = 'dr16'
# dr = 'dr17'  # TODO
APOGEE_CACHE_PATH = pathlib.Path(os.environ.get(
    "APOGEE_CACHE_PATH",
    pathlib.Path.home() / ".apogee")).expanduser().resolve()

# Load authentication for SDSS
sdss_auth_file = pathlib.Path('~/.sdss.login').expanduser()
if sdss_auth_file.exists():
    with open(sdss_auth_file, 'r') as f:
        sdss_auth = f.readlines()
    sdss_auth = tuple([s.strip() for s in sdss_auth if len(s.strip()) > 0])
else:
    sdss_auth = None

phot_names = [
    'GAIAEDR3_PHOT_G_MEAN_MAG',
    'GAIAEDR3_PHOT_BP_MEAN_MAG',
    'GAIAEDR3_PHOT_RP_MEAN_MAG',
    'J', 'H', 'K',
    'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro'
]
