import os
import pathlib

dr = 'dr17'
reduction = 'turbo20'
stars_ = 'stars.beta'

APOGEE_CACHE_PATH = pathlib.Path(os.environ.get(
    "APOGEE_CACHE_PATH",
    pathlib.Path.home() / ".apogee")).expanduser().resolve()

cache_path = APOGEE_CACHE_PATH / dr / reduction
cache_path.mkdir(parents=True, exist_ok=True)

# Load authentication for SDSS
sdss_auth_file = pathlib.Path('~/.sdss.login').expanduser()
if sdss_auth_file.exists():
    with open(sdss_auth_file, 'r') as f:
        sdss_auth = f.readlines()
    sdss_auth = tuple([s.strip() for s in sdss_auth if len(s.strip()) > 0])
else:
    sdss_auth = None
