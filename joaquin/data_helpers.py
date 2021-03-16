import requests
from .config import sdss_auth


def _authcheck():
    if sdss_auth is None:
        raise RuntimeError("No SDSS authentication information available. "
                           "Create a ~/.sdss.login file with the standard "
                           "SDSS username and password (one per line).")


def download_file(url, local_path, overwrite=False):
    if not local_path.exists() or overwrite:
        r = requests.get(url, auth=sdss_auth)

        if not r.ok:
            raise RuntimeError(f"Failed to download file from {url}: {r}")

        with open(local_path, 'wb') as f:
            f.write(r.content)

    return local_path
