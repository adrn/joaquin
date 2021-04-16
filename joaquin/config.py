import os
import pathlib

###############################################################################
# Joaquin

# Photometric colors used in the neighborhood construction:
neighborhood_color_names = [
    ('phot_bp_mean_mag', 'phot_rp_mean_mag'),
    ('phot_rp_mean_mag', 'w1mpro'),
    ('H', 'w2mpro'),
    ('w1mpro', 'w2mpro')
]

# Photometric measurements used by the model
phot_names = [
    'phot_g_mean_mag',
    'phot_bp_mean_mag',
    'phot_rp_mean_mag',
    'J', 'H', 'K',
    'w1mpro', 'w2mpro'
]

# Number of PCA components to use for making the neighborhoods:
neighborhood_n_components = 8

# Maximum size of the larger neighborhood around each "stoop":
max_neighborhood_size = 32768

# Size of the testing subsample (the "block") around each "stoop":
block_size = 1024

# For cross-validation:
Kfold_K = 4

# Number of PCA components to use to patch the missing spectral pixels:
patching_n_components = 8

###############################################################################
# APOGEE DATA / CACHING

# What APOGEE data reduction are we using?
dr = 'dr17'
reduction = 'dr17'

# Path to a /.../apogee folder to download the data to
APOGEE_CACHE_PATH = pathlib.Path(os.environ.get(
    "APOGEE_CACHE_PATH",
    pathlib.Path.home() / ".apogee")).expanduser().resolve()

JOAQUIN_OUTPUT_PATH = os.environ.get("JOAQUIN_OUTPUT_PATH")
if JOAQUIN_OUTPUT_PATH is not None:
    JOAQUIN_OUTPUT_PATH = pathlib.Path(JOAQUIN_OUTPUT_PATH)

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
    'phot_g_mean_mag',
    'phot_bp_mean_mag',
    'phot_rp_mean_mag',
    'J', 'H', 'K',
    'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro'
]

###############################################################################
# Logging configuration

logger_level = 1  # show all messages
# logger_level = 20  # INFO
