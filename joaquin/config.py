import os
import pathlib

###############################################################################
# Joaquin

# Photometric colors used in the neighborhood construction:
neighborhood_color_names = [
    ('GAIAEDR3_PHOT_BP_MEAN_MAG', 'GAIAEDR3_PHOT_RP_MEAN_MAG'),
    ('GAIAEDR3_PHOT_RP_MEAN_MAG', 'w1mpro'),
    ('H', 'w2mpro'),
    ('w1mpro', 'w2mpro')
]

# Number of PCA components to use for making the neighborhoods:
neighborhood_n_components = 8

# Size of the larger neighborhood around each "stoop":
neighborhood_size = 8192

# Size of the testing subsample (the "block") around each "stoop":
block_size = 2048

# For cross-validation:
Kfold_K = 8

# Number of PCA components to use to patch the missing spectral pixels:
patching_n_components = 8

###############################################################################
# APOGEE DATA / CACHING

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
