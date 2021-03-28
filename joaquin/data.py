import pathlib

import astropy.table as at
import h5py
import numpy as np
from tqdm.auto import tqdm

from .features import get_lsf_features, get_phot_features, get_spec_features
from .apogee_data import get_aspcapstar, get_lsf
from .logger import logger
from .config import all_phot_names, root_cache_path


class JoaquinData:

    def __init__(self, stars=None, cache_file=None, lowpass=True,
                 overwrite=False, progress=True):

        if cache_file is not None:
            cache_file = pathlib.Path(cache_file)

            if not cache_file.suffix:  # filename only passed in
                cache_file = (root_cache_path / 'data' /
                              cache_file.with_suffix(".hdf5"))

            cache_file = cache_file.resolve()
            logger.debug(f"Cache file parsed to: {str(cache_file)}")
            cache_file.parent.mkdir(exist_ok=True, parents=True)

            if not cache_file.exists() or overwrite:
                logger.debug(
                    'Design matrix data cache file not found, or being '
                    f'overwritten for {len(stars)} stars')
                self._make_Xy(stars, progress=progress, lowpass=lowpass)
                self._write(cache_file)

            else:
                logger.debug(
                    'Design matrix cache file found: loading pre-cached design '
                    f'matrix data from {str(cache_file)}')
                self._read(cache_file)

        else:
            self._make_Xy(stars, progress=progress, lowpass=lowpass)

    def _init(self, X, y, y_ivar, spec_mask_vals, idx_map, stars, **kwargs):

        self._X = X
        self._y = y
        self._y_ivar = y_ivar
        self._spec_mask_vals = spec_mask_vals
        self._idx_map = idx_map

        self.stars = stars

        stars_mask = np.all(np.isfinite(self._X), axis=1)
        if not np.all(stars_mask):
            nfailures = len(stars_mask) - stars_mask.sum()
            logger.debug(
                f'failed to get spectral features for {nfailures} stars')
        self.stars_mask = stars_mask

        for key, val in kwargs.items():
            setattr(self, key, val)

        assert self._X.shape[0] == self._y.shape[0] == self._y_ivar.shape[0]

    def _make_Xy(self, stars, progress=True, lowpass=True):
        if progress:
            iter_ = tqdm
        else:
            iter_ = iter

        if stars is None:
            raise ValueError(
                "Input `stars` is None! You must pass a table of allStar data "
                "using the `stars` argument to the initializer")

        # First, figure out how many features we have:
        for star in stars:
            try:
                star_hdul = get_aspcapstar(star)
                lsf_hdul = get_lsf(star)

                lsf_f = get_lsf_features(lsf_hdul)
                phot_f = get_phot_features(stars)
                spec_f, mask = get_spec_features(star_hdul, lowpass=lowpass)

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
            except Exception as e:
                logger.log(
                    1, f"failed to load aspcapStar file for star {i}:\n{e}")
                continue

            try:
                lsf_hdul = get_lsf(star)
            except Exception as e:
                logger.log(
                    1, f"failed to load apStarLSF file for star {i}:\n{e}")
                continue

            lsf_f = get_lsf_features(lsf_hdul)
            phot_f = get_phot_features(star)
            try:
                spec_f, spec_mask = get_spec_features(star_hdul,
                                                      lowpass=lowpass)
            except:  # noqa
                logger.log(1, f"failed to get spectrum features for star {i}")
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

        spec_mask_vals = np.nansum(spec_masks, axis=0) / len(spec_masks)

        idx_map = {
            'lsf': lsf_idx,
            'phot': phot_idx,
            'spec': spec_idx
        }

        self._init(X, y, y_ivar, spec_mask_vals, idx_map, stars,
                   lowpass=lowpass, all_phot_names=all_phot_names)

    def _read(self, filename):
        data = {}
        with h5py.File(filename, 'r') as f:
            for k in ['X', 'y', 'y_ivar', 'spec_mask_vals']:
                data[k] = f[k][:]

            data['idx_map'] = {}
            for k in f['idx_map'].keys():
                data['idx_map'][k] = f['idx_map'][k][:]

            data['stars'] = at.Table.read(f['stars'])

            for k in f.attrs.keys():
                data[k] = f.attrs[k]

        self._init(**data)

    def _write(self, filename):
        meta = {'lowpass': self.lowpass,
                'all_phot_names': self.all_phot_names}

        with h5py.File(filename, 'w') as f:
            f.create_dataset('X', data=self._X)
            f.create_dataset('y', data=self._y)
            f.create_dataset('y_ivar', data=self._y_ivar)
            f.create_dataset('spec_mask_vals', data=self._spec_mask_vals)

            g = f.create_group('idx_map')
            for key, idx in self._idx_map.items():
                g.create_dataset(key, data=idx)

            self.stars.write(f, path='stars', serialize_meta=False)

            for k in meta:
                f.attrs[k] = meta[k]

    def get_Xy(self, terms=['phot', 'lsf', 'spec'],
               phot_names=None,
               spec_mask_thresh=0.25, spec_good_mask=None,
               good_stars_only=True):

        if isinstance(terms, str):
            terms = [terms]

        if phot_names is None:
            phot_names = all_phot_names

        if spec_good_mask is None:
            logger.debug('Using cached spectral feature mask values')
            spec_good_mask = self._spec_mask_vals < spec_mask_thresh

        idx_map = self._idx_map.copy()
        idx_map['spec'] = idx_map['spec'][spec_good_mask]

        _, idx1, idx2 = np.intersect1d(all_phot_names, phot_names,
                                       return_indices=True)
        idx_map['phot'] = idx_map['phot'][idx1[idx2.argsort()]]

        idx = []
        new_idx_map = {}
        start = 0
        for name in terms:
            idx.append(idx_map[name])

            new_idx_map[name] = np.arange(start,
                                          start + len(idx_map[name]))
            start = start + len(idx_map[name])
        idx = np.concatenate(idx)

        if good_stars_only:
            nidx = self.stars_mask
        else:
            nidx = np.ones(len(self.stars_mask), dtype=bool)

        return ((self._X[nidx][:, idx], self._y[nidx], self._y_ivar[nidx]),
                new_idx_map)

    def get_colors(self, colors, good_stars_only=True):
        """
        colors : iterable of tuples of strings
        """
        color_X = []
        for band1, band2 in colors:
            i1 = self._idx_map['phot'][all_phot_names.index(band1)]
            i2 = self._idx_map['phot'][all_phot_names.index(band2)]
            color_X.append(self._X[:, i1] - self._X[:, i2])

        if good_stars_only:
            nidx = self.stars_mask
        else:
            nidx = np.ones(len(self.stars_mask), dtype=bool)

        return np.stack(color_X, axis=1)[nidx]