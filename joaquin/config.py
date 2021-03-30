import os
import pathlib

###############################################################################
# Joaquin

# For zoning the neighborhoods:
zone1_K = 2048
zone2_K = 8192

# For cross-validation:
Kfold_K = 8

###############################################################################
# DATA / CACHING

# What APOGEE data reduction are we using?
dr = 'dr16'
reduction = 'r12'
# dr = 'dr17'  # TODO
# reduction = '??'

# Path to a /.../apogee folder to download the data to
APOGEE_CACHE_PATH = pathlib.Path(os.environ.get(
    "APOGEE_CACHE_PATH",
    pathlib.Path.home() / ".apogee")).expanduser().resolve()

JOAQUIN_CACHE_PATH = pathlib.Path(os.environ.get(
    "JOAQUIN_CACHE_PATH",
    pathlib.Path(__file__).parent.parent / 'cache'))
root_cache_path = JOAQUIN_CACHE_PATH / dr

# Load authentication for SDSS
sdss_auth_file = pathlib.Path('~/.sdss.login').expanduser()
if sdss_auth_file.exists():
    with open(sdss_auth_file, 'r') as f:
        sdss_auth = f.readlines()
    sdss_auth = tuple([s.strip() for s in sdss_auth if len(s.strip()) > 0])
else:
    sdss_auth = None

# All photometric data column names:
all_phot_names = [
    'GAIAEDR3_PHOT_G_MEAN_MAG',
    'GAIAEDR3_PHOT_BP_MEAN_MAG',
    'GAIAEDR3_PHOT_RP_MEAN_MAG',
    'J', 'H', 'K',
    'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro'
]

###############################################################################
# Logging configuration

logger_level = 1  # show all messages
# logger_level = 20  # INFO
