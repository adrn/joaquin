import pathlib

import astropy.table as at
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm, trange

from .apogee_data import get_aspcapstar, get_lsf
from .features import get_lsf_features, get_phot_features, get_spec_features
from .logger import logger
from .config import all_phot_names as default_phot_names, root_cache_path

__all__ = ['JoaquinData']


class JoaquinData:

    def __init__(self, stars, X, idx_map, spec_wvln, spec_bad_masks=None,
                 all_phot_names=None):

        self.stars = stars
        self.X = np.array(X)
        self.spec_wvln = np.array(spec_wvln)
        self.spec_bad_masks = spec_bad_masks
        self._idx_map = idx_map

        if self.spec_bad_masks is not None:
            self.spec_bad_masks = np.array(self.spec_bad_masks)

        if all_phot_names is None:
            all_phot_names = default_phot_names
        self.all_phot_names = all_phot_names

        stars_mask = np.all(np.isfinite(self.X), axis=1)
        if not np.all(stars_mask):
            nfailures = len(stars_mask) - stars_mask.sum()
            raise ValueError(
                f'Input feature matrix, X, has {nfailures} stars with '
                'non-finite values. You must pass in a cleaned feature matrix '
                'instead.')

        assert len(self._idx_map['spec']) == len(self.spec_wvln)
        assert len(self._idx_map['phot']) == len(self.all_phot_names)
        assert self.X.shape[0] == len(stars)

    @classmethod
    def _parse_cache_file(cls, cache_file):
        cache_file = pathlib.Path(cache_file)

        if not cache_file.suffix:  # only file basename passed in
            cache_file = (root_cache_path / 'data' /
                          cache_file.with_suffix(".hdf5"))

        cache_file = cache_file.resolve()
        logger.debug(f"Cache file parsed to: {str(cache_file)}")
        cache_file.parent.mkdir(exist_ok=True, parents=True)

        return cache_file

    @classmethod
    def from_stars(cls, stars, cache_file=None, overwrite=False,
                   progress=True):

        if cache_file is not None:
            cache_file = cls._parse_cache_file(cache_file)

        if cache_file is not None and cache_file.exists() and not overwrite:
            logger.debug(
                'Design matrix cache file found: loading pre-cached design '
                f'matrix data from {str(cache_file)}')
            obj = cls.read(cache_file)

            if stars is not None:
                # TODO: hard-coded ID column name (APSTAR_ID)
                ids1 = np.unique(obj.stars['APSTAR_ID']).astype(str)
                ids2 = np.unique(stars['APSTAR_ID']).astype(str)
                if not np.all(ids1 == ids2):
                    raise RuntimeError(
                        "Input `stars` table is not consistent with the "
                        "previously cached `cache_file` stars! Use "
                        "`overwrite=True` to overwrite the existing cache "
                        "file."
                    )

            return obj

        # From here on, whether the cache file exists or not, we will build the
        # sample from the input stars!
        X, idx_map, wvln, spec_masks = make_X(stars, progress=progress)

        obj = cls(stars, X, idx_map, wvln, spec_bad_masks=spec_masks)

        if cache_file is not None or overwrite:
            logger.debug(
                'Design matrix data cache file not found, or being '
                f'overwritten for {len(stars)} stars')
            obj.write(cache_file)

        return obj

    @classmethod
    def read(cls, cache_file):
        cache_file = cls._parse_cache_file(cache_file)

        data = {}
        with h5py.File(cache_file, 'r') as f:
            for k in ['X', 'spec_wvln', 'spec_bad_masks']:
                if k in f.keys():
                    data[k] = f[k][:]
                else:
                    logger.debug(f"Did not find dataset '{k}' in cache file")

            data['idx_map'] = {}
            for k in f['idx_map'].keys():
                data['idx_map'][k] = f['idx_map'][k][:]

            data['stars'] = at.Table.read(f['stars'])

            data['all_phot_names'] = np.array(f.attrs['all_phot_names'])

        return cls(**data)

    def write(self, cache_file):
        cache_file = self._parse_cache_file(cache_file)

        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('X', data=self.X)
            f.create_dataset('spec_wvln', data=self.spec_wvln)

            if self.spec_bad_masks is not None:
                f.create_dataset('spec_bad_masks', data=self.spec_bad_masks)

            g = f.create_group('idx_map')
            for key, idx in self._idx_map.items():
                g.create_dataset(key, data=idx)

            self.stars.write(f, path='stars', serialize_meta=False)

            f.attrs['all_phot_names'] = list(self._all_phot_names)

    def _replicate(self, X, stars=None, **kwargs):
        if stars is None:
            stars = self.stars

        kwargs.setdefault('idx_map', self._idx_map),
        kwargs.setdefault('spec_wvln', self.spec_wvln)
        kwargs.setdefault('spec_bad_masks', self.spec_bad_masks)
        kwargs.setdefault('all_phot_names', self.all_phot_names)

        return self.__class__(stars, X, **kwargs)

    def __getitem__(self, slc):

        if not isinstance(slc, slice):
            try:
                slc = np.array(slc)
            except Exception:
                super().__getitem__(slc)

            if not np.issubdtype(slc.dtype, np.integer):
                raise KeyError("Input slice array must have an integer dtype")

        return self._replicate(X=self.X[slc],
                               stars=self.stars[slc],
                               spec_bad_masks=self.spec_bad_masks[slc])

    ###########################################################################
    # Operations on the feature matrix

    def lowpass_filter_spec(self, fcut_factor=1., progress=True,
                            fill_value=0., copy=True):
        from .filters import nufft_lowpass

        if progress:
            range_ = trange
        else:
            range_ = range

        # TODO: APOGEE resolution hard-coded
        fcut = 0.5 * 22500 * fcut_factor
        ln_wvln = np.log(self.spec_wvln)
        spec_X, _ = self.get_X('spec')

        if copy:
            new_spec_X = np.full_like(spec_X, np.nan)
        else:
            new_spec_X = spec_X

        mask = None
        for i in range_(self.X.shape[0]):
            if self.spec_bad_masks is not None:
                mask = self.spec_bad_masks[i]

            new_spec_X[i] = nufft_lowpass(ln_wvln,
                                          spec_X[i],
                                          fcut=fcut,
                                          bad_mask=mask)
            new_spec_X[i][mask] = fill_value

        if copy:
            new_X = self.X.copy()
            new_X[:, self._idx_map['spec']] = new_spec_X
            return self._replicate(X=new_X)

        else:
            self.X[:, self._idx_map['spec']] = new_spec_X
            return self

    def fill_masked_spec_pixels(self, global_spec_bad_mask=None,
                                spec_mask_thresh=None, fill_value=0.,
                                copy=True):

        if self.spec_bad_masks is not None:
            spec_bad_masks = self.spec_bad_masks.copy()
        else:
            spec_bad_masks = np.zeros(
                (len(self.stars), self.n_features('spec')), dtype=bool)

        if global_spec_bad_mask is not None and spec_mask_thresh is not None:
            raise ValueError(
                "You may only specify one of `global_spec_bad_mask` or "
                "`spec_mask_thresh`, not both.")

        if global_spec_bad_mask is not None:
            global_spec_bad_mask = np.array(global_spec_bad_mask)

            if global_spec_bad_mask.ndim > 1:
                raise ValueError("The global bad spectral feature mask must "
                                 "be one-dimensional. To set the 2D mask, "
                                 "override the `spec_bad_masks` attribute.")

        if spec_mask_thresh is not None:
            spec_mask_vals = spec_bad_masks.sum(axis=0)
            global_spec_bad_mask = spec_mask_vals >= spec_mask_thresh

        spec_bad_masks |= global_spec_bad_mask

        if copy:
            X = self.X.copy()
            X[spec_bad_masks] = fill_value
            return self._replicate(X=X)

        else:
            self.X[spec_bad_masks] = fill_value
            return self

    def patch_spec(self, stuff, patching_n_components=None):
        from .config import (patching_n_components
                             as default_patching_n_components)

        if patching_n_components is None:
            patching_n_components = default_patching_n_components

        pca = PCA(n_components=patching_n_components)

        subX_pca = pca.fit_transform(self.X)
        tmp_patched = pca.inverse_transform(subX_pca)

        subX_patched = self.X.copy()
        subX_patched[subX_patched == 0] = tmp_patched[subX_patched == 0]

        return self._replicate(X=subX_patched)

    ###########################################################################
    # Feature matrix utilities

    def get_X(self, terms=['phot', 'lsf', 'spec'],
              phot_names=None):

        if isinstance(terms, str):
            terms = [terms]

        if phot_names is None:
            phot_names = default_phot_names

        idx_map = self._idx_map.copy()
        _, idx1, idx2 = np.intersect1d(self.all_phot_names, phot_names,
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

        return self.X[:, idx], new_idx_map

    def put_X(self, sub_X, term, copy=True):
        pass

    def get_neighborhood_X(self, color_names=None, renormalize_colors=True):
        from .config import neighborhood_color_names
        from .filters import renormalize

        X, idx_map = self.get_X('spec')

        if color_names is None:
            color_names = neighborhood_color_names

        color_X = self.get_colors(color_names)

        if renormalize_colors:
            for i in range(color_X.shape[1]):
                color_X[:, i] = renormalize(
                    color_X[:, i],
                    *np.nanpercentile(color_X[:, i], [5, 95]))  # MAGIC NUMBERs

        X = np.hstack((X, color_X))

        return X

    ###########################################################################
    # Convenience attributes and misc

    def n_features(self, term):
        try:
            N = len(self._idx_map[term])
        except KeyError:
            raise KeyError(
                f"{term} is not a valid feature matrix component name! For "
                f"this dataset, it must be one of: {list(self._idx_map.keys())}"
            )
        return N

    def get_colors(self, colors, good_stars_only=True):
        """
        colors : iterable of tuples of strings
        """
        color_X = []
        for band1, band2 in colors:
            i1 = self._idx_map['phot'][self.all_phot_names.index(band1)]
            i2 = self._idx_map['phot'][self.all_phot_names.index(band2)]
            color_X.append(self._X[:, i1] - self._X[:, i2])

        if good_stars_only:
            nidx = self.stars_mask
        else:
            nidx = np.ones(len(self.stars_mask), dtype=bool)

        return np.stack(color_X, axis=1)[nidx]


def make_X(stars, progress=True, X_dtype=np.float32):
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
            wvln, flux, err = get_aspcapstar(star)
            pix, lsf = get_lsf(star)

            phot_f = get_phot_features(star)
            lsf_f = get_lsf_features(lsf)
            spec_f, mask = get_spec_features(wvln, flux, err,
                                             lowpass=False)

        except Exception:  # noqa
            continue

        Nlsf = len(lsf_f)
        Nphot = len(phot_f)
        Nspec = len(spec_f)

        break

    else:
        raise RuntimeError("Failed to determine number of features")

    Nstars = len(stars)
    Nfeatures = Nphot + Nlsf + Nspec
    X = np.full((Nstars, Nfeatures), np.nan, dtype=X_dtype)
    spec_bad_masks = np.full((Nstars, Nspec), True, dtype=bool)
    for i, star in enumerate(iter_(stars)):
        try:
            wvln, flux, err = get_aspcapstar(star)
            pix, lsf = get_lsf(star)
        except Exception as e:
            logger.log(1,
                       "failed to get aspcapStar or apStarLSF data for "
                       f"star {i}\n{e}")
            continue

        try:
            phot_f = get_phot_features(star)
            lsf_f = get_lsf_features(lsf)
            spec_f, spec_mask = get_spec_features(wvln, flux, err,
                                                  lowpass=False)
        except Exception as e:
            logger.log(1, f"failed to get features for star {i}\n{e}")
            continue

        phot_idx = np.arange(Nphot, dtype=int)

        last = phot_idx[-1] + 1
        lsf_idx = np.arange(last, last + Nlsf, dtype=int)

        last = lsf_idx[-1] + 1
        spec_idx = np.arange(last, last + Nspec, dtype=int)

        X[i] = np.concatenate((phot_f, lsf_f, spec_f))
        spec_bad_masks[i] = spec_mask

    idx_map = {
        'phot': phot_idx,
        'lsf': lsf_idx,
        'spec': spec_idx
    }

    return X, idx_map, wvln, spec_bad_masks
