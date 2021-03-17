import pathlib
import hashlib
import pickle

import numpy as np
from tqdm.auto import tqdm

from .features import get_lsf_features, get_phot_features, get_spec_features
from .apogee_data import get_aspcapstar, get_lsf
from .logger import logger


class DesignMatrix:

    def __init__(self, stars, overwrite=False, progress=True, cache_path=None):
        if cache_path is None:
            cache_path = 'cache'
        cache_path = pathlib.Path(cache_path)
        this_cache_path = (cache_path / 'designmatrix').resolve()
        this_cache_path.mkdir(exist_ok=True, parents=True)

        name = hashlib.sha256(
            (','.join(stars['APOGEE_ID'])).encode('utf-8')).hexdigest()
        cache_file = this_cache_path / f'{name}.pkl'

        if not cache_file.exists() or overwrite:
            logger.debug(
                'Design matrix cache file not found, or being overwritten for '
                f'{len(stars)} stars')
            Xyivar, idx_map, failures = self._make_Xy(stars, progress=progress)

            with open(cache_file, 'wb') as f:
                pickle.dump((Xyivar, idx_map, failures), f)

        else:
            logger.debug(
                'Design matrix cache file found: loading pre-cached design '
                f'matrix from {str(cache_file)}')
            with open(cache_file, 'rb') as f:
                (Xyivar, idx_map, failures) = pickle.load(f)

        self.idx_map = idx_map
        self._X = Xyivar[0]
        self._y = Xyivar[1]
        self._y_ivar = Xyivar[2]

        self._good_stars_mask = np.ones(len(stars), dtype=bool)
        if len(failures) > 0:
            logger.debug(
                f'failed to get spectral features for {len(failures)} stars')
            self._good_stars_mask[failures] = False
        self.stars = stars[self._good_stars_mask]

    def _make_Xy(self, stars, progress=True):
        if progress:
            iter_ = tqdm
        else:
            iter_ = iter

        X = None
        del_idx = []
        for i, star in enumerate(iter_(stars)):
            star_hdul = get_aspcapstar(star)
            lsf_hdul = get_lsf(star)

            lsf_f = get_lsf_features(lsf_hdul)
            phot_f = get_phot_features(star)
            try:
                spec_f = get_spec_features(star_hdul)
            except:  # noqa
                logger.log(0, f"failed to get spectrum features for star {i}")
                del_idx.append(i)
                star_hdul.close()
                lsf_hdul.close()
                continue

            if X is None:
                Nlsf = len(lsf_f)
                Nphot = len(phot_f)
                Nspec = len(spec_f)

                X = np.full((len(stars),
                             Nlsf + Nphot + Nspec),
                            np.nan)

                lsf_idx = np.arange(Nlsf, dtype=int)
                phot_idx = np.arange(Nlsf,
                                     Nlsf + Nphot,
                                     dtype=int)
                spec_idx = np.arange(Nlsf + Nphot,
                                     Nlsf + Nphot + Nspec,
                                     dtype=int)

            X[i] = np.concatenate((lsf_f, phot_f, spec_f))

            star_hdul.close()
            lsf_hdul.close()

        y = stars['GAIAEDR3_PARALLAX']
        y_ivar = 1 / stars['GAIAEDR3_PARALLAX_ERROR'] ** 2

        X = np.delete(X, del_idx, axis=0)
        y = np.delete(y, del_idx)
        y_ivar = np.delete(y_ivar, del_idx)

        idx_map = {
            'lsf': lsf_idx,
            'phot': phot_idx,
            'spec': spec_idx
        }

        return (X, y, y_ivar), idx_map, np.array(del_idx)

    def get_sub_Xy(self, terms=['lsf', 'phot', 'spec']):
        idx = []
        for name in terms:
            idx.append(self.idx_map[name])
        idx = np.concatenate(idx)
        return self._X[:, idx], self._y, self._y_ivar
