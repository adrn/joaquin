import pathlib
import hashlib
import pickle

import numpy as np
from tqdm.auto import tqdm

from .features import get_lsf_features, get_phot_features, get_spec_features
from .apogee_data import get_aspcapstar, get_lsf
from .logger import logger


def make_Xy(stars, progress=True, lowpass=True):
    if progress:
        iter_ = tqdm
    else:
        iter_ = iter

    # First, figure out how many features we have:
    for star in stars:
        try:
            star_hdul = get_aspcapstar(star)
            lsf_hdul = get_lsf(star)

            lsf_f = get_lsf_features(lsf_hdul)
            phot_f = get_phot_features(star)
            spec_f, mask = get_spec_features(star_hdul)

        except Exception:  # noqa
            continue

        finally:
            star_hdul.close()
            lsf_hdul.close()

        Nlsf = len(lsf_f)
        Nphot = len(phot_f)
        Nspec = len(spec_f)
        break
    else:
        raise RuntimeError("Failed to determine number of features")

    Nstars = len(stars)
    Nfeatures = Nphot + Nlsf + Nspec
    X = np.full((Nstars, Nfeatures), np.nan)
    spec_masks = np.full((Nstars, Nspec), np.nan)
    for i, star in enumerate(iter_(stars)):
        try:
            star_hdul = get_aspcapstar(star)
            lsf_hdul = get_lsf(star)
        except Exception as e:
            logger.debug(f"failed to load data file for star {i}:\n{e}")
            continue

        lsf_f = get_lsf_features(lsf_hdul)
        phot_f = get_phot_features(star)
        try:
            spec_f, spec_mask = get_spec_features(star_hdul, lowpass=lowpass)
        except:  # noqa
            logger.log(0, f"failed to get spectrum features for star {i}")
            continue
        finally:
            star_hdul.close()
            lsf_hdul.close()

        phot_idx = np.arange(Nphot, dtype=int)

        last = phot_idx[-1] + 1
        lsf_idx = np.arange(last, last + Nlsf, dtype=int)

        last = lsf_idx[-1] + 1
        spec_idx = np.arange(last, last + Nspec, dtype=int)

        X[i] = np.concatenate((phot_f, lsf_f, spec_f))
        spec_masks[i] = spec_mask

        star_hdul.close()
        lsf_hdul.close()

    y = stars['GAIAEDR3_PARALLAX']
    y_ivar = 1 / stars['GAIAEDR3_PARALLAX_ERROR'] ** 2

    all_spec_mask = np.sum(spec_masks, axis=0) / len(spec_masks)

    idx_map = {
        'lsf': lsf_idx,
        'phot': phot_idx,
        'spec': spec_idx
    }

    return (X, y, y_ivar), idx_map, all_spec_mask


class JoaquinData:

    def __init__(self, X, y, y_ivar, idx_map, spec_mask_vals=None,
                 spec_mask_thresh=None):

        self.X = np.array(X)
        self.y = np.array(y)
        self.y_ivar = np.array(y_ivar)
        self.idx_map = idx_map

        assert self.X.shape[0] == self.y.shape[0] == self.y_ivar.shape[0]

        self._spec_mask_thresh = spec_mask_thresh
        if spec_mask_vals is None:
            spec_mask_vals = np.zeros(len(self.idx_map['spec']))
        self._spec_mask_vals = np.array(spec_mask_vals)

        if spec_mask_thresh is not None:
            self._spec_good_mask = self._spec_mask_vals < spec_mask_thresh
        else:
            self._spec_good_mask = np.ones(len(self._spec_mask_vals),
                                           dtype=bool)
        self.idx_map['spec'] = self.idx_map['spec'][self._spec_good_mask]

    @classmethod
    def from_stars(cls, stars, cache_path, lowpass=True, overwrite=False,
                   progress=True, **kwargs):

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
            Xyivar, idx_map, spec_mask = make_Xy(stars, progress=progress,
                                                 lowpass=lowpass)

            with open(cache_file, 'wb') as f:
                pickle.dump((Xyivar, idx_map, spec_mask), f)

        else:
            logger.debug(
                'Design matrix cache file found: loading pre-cached design '
                f'matrix from {str(cache_file)}')
            with open(cache_file, 'rb') as f:
                (Xyivar, idx_map, spec_mask) = pickle.load(f)

        good_stars_mask = np.all(np.isfinite(Xyivar[0]), axis=1)
        if not np.all(good_stars_mask):
            nfailures = len(good_stars_mask) - good_stars_mask.sum()
            logger.debug(
                f'failed to get spectral features for {nfailures} stars')

        return cls(
            X=Xyivar[0],
            y=Xyivar[1],
            y_ivar=Xyivar[2],
            idx_map=idx_map,
            spec_mask_vals=spec_mask,
            **kwargs), good_stars_mask

    def get_sub_Xy(self, terms=['phot', 'lsf', 'spec']):
        if isinstance(terms, str):
            terms = [terms]

        idx = []
        new_idx_map = {}
        start = 0
        for name in terms:
            idx.append(self.idx_map[name])

            new_idx_map[name] = np.arange(start,
                                          start + len(self.idx_map[name]))
            start = start + len(self.idx_map[name])

        idx = np.concatenate(idx)
        return self.X[:, idx], self.y, self.y_ivar, new_idx_map

    def __getitem__(self, slice_):
        cls = self.__class__
        return cls(self.X[slice_], self.y[slice_], self.y_ivar[slice_],
                   self.idx_map)
