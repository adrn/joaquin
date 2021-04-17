# Standard library
import pathlib

# Third-party
import astropy.table as at
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import trange

# Joaquin
from .logger import logger

__all__ = ['JoaquinData']


class JoaquinData:

    def __init__(self, stars, X, idx_map, spec_wvln, spec_bad_masks=None,
                 phot_names=None):
        """The main data storage container for passing to a Joaquin model

        Parameters
        ----------


        TODO
        ----
        This could cache just the spec (and metadata, e.g., LSF) part of the
        feature matrix, and reconstruct the full feature matrix with the
        photometry and anything else the user wants at runtime...

        """

        self.stars = stars
        self.X = np.array(X)
        self.spec_wvln = np.array(spec_wvln)
        self.spec_bad_masks = spec_bad_masks
        self._idx_map = idx_map

        if self.spec_bad_masks is not None:
            self.spec_bad_masks = np.array(self.spec_bad_masks)

        self._phot_names = phot_names
        if self._phot_names is not None:
            self._phot_names = np.array(self._phot_names)

        stars_mask = np.all(np.isfinite(self.X), axis=1)
        if not np.all(stars_mask):
            nfailures = len(stars_mask) - stars_mask.sum()
            raise ValueError(
                f'Input feature matrix, X, has {nfailures} stars with '
                'non-finite values. You must pass in a cleaned feature matrix '
                'instead.')

        assert self.X.shape[0] == len(stars)
        assert len(self._idx_map['spec']) == len(self.spec_wvln)

        if 'phot' in self._idx_map:
            if self._phot_names is None:
                raise TypeError(
                    "If photometric data is incuded in the feature matrix, "
                    "you must pass in the names of the bands using "
                    "'phot_names'")
            assert len(self._idx_map['phot']) == len(self._phot_names)

    @classmethod
    def _parse_cache_file(cls, cache_file):
        cache_file = pathlib.Path(cache_file)

        if not cache_file.suffix or cache_file.suffix != '.hdf5':
            raise ValueError(
                "Cache file must have a .hdf5 file type / extension.")

        cache_file = cache_file.resolve()
        logger.debug(f"Cache file parsed to: {str(cache_file)}")
        cache_file.parent.mkdir(exist_ok=True, parents=True)

        return cache_file

    @classmethod
    def from_stars(cls, config, stars, stars_to_X_func, cache_file=None,
                   overwrite=False, progress=True):

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
                if ids1.shape != ids2.shape or not np.all(ids1 == ids2):
                    logger.warning(
                        "Input `stars` table is not consistent with the "
                        "previously cached `cache_file` stars. This is likely "
                        "because some stars were masked due to lack of data. "
                        "But you might want to check to make sure! Or use "
                        "`overwrite=True` to overwrite the existing cache "
                        "file."
                    )

            return obj

        # From here on, whether the cache file exists or not, we will build the
        # sample from the input stars!
        X, idx_map, wvln, spec_masks = stars_to_X_func(config, stars,
                                                       progress=progress)

        stars_mask = np.all(np.isfinite(X), axis=1)
        obj = cls(stars[stars_mask], X[stars_mask], idx_map, wvln,
                  spec_bad_masks=spec_masks[stars_mask],
                  phot_names=config.phot_names)

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

            data['phot_names'] = np.array(f.attrs['phot_names']).astype(str)

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

            f.attrs['phot_names'] = list(self._phot_names.astype('S'))

    def _replicate(self, X, stars=None, **kwargs):
        if stars is None:
            stars = self.stars

        kwargs.setdefault('idx_map', self._idx_map),
        kwargs.setdefault('spec_wvln', self.spec_wvln)
        kwargs.setdefault('spec_bad_masks', self.spec_bad_masks)
        kwargs.setdefault('phot_names', self._phot_names)

        return self.__class__(stars, X, **kwargs)

    def __getitem__(self, slc):

        if not isinstance(slc, slice):
            try:
                slc = np.array(slc)
            except Exception:
                super().__getitem__(slc)

            if (not np.issubdtype(slc.dtype, np.integer) and
                    not np.issubdtype(slc.dtype, np.bool)):
                raise KeyError("Input slice array must have an integer or "
                               "boolean dtype")

        kw = dict(
            X=self.X[slc],
            stars=self.stars[slc],
        )
        if self.spec_bad_masks is not None:
            kw['spec_bad_masks'] = self.spec_bad_masks[slc]

        return self._replicate(**kw)

    def __len__(self):
        return len(self.stars)

    ###########################################################################
    # Operations on the feature matrix

    def fill_masked_spec_pixels(self, spec_mask_thresh=None, fill_value=0.,
                                copy=True):

        if self.spec_bad_masks is not None:
            spec_bad_masks = self.spec_bad_masks
        else:
            spec_bad_masks = np.zeros(
                (len(self.stars), self.n_features('spec')), dtype=bool)

        if spec_mask_thresh is not None:
            spec_mask_vals = spec_bad_masks.sum(axis=0) / len(spec_bad_masks)
            global_spec_bad_mask = spec_mask_vals >= spec_mask_thresh
            spec_bad_masks = spec_bad_masks | global_spec_bad_mask

        spec_X, _ = self.get_X('spec')
        spec_X[spec_bad_masks] = fill_value

        if copy:
            new_X = self.X.copy()
            new_X[:, self._idx_map['spec']] = spec_X
            return self._replicate(X=new_X)

        else:
            raise NotImplementedError("TODO")

            # Note: This indexing order doesn't work for in-place...
            # self.X[:, self._idx_map['spec']] = spec_X
            # return self

    def mask_spec_pixels(self, spec_bad_mask, copy=True):
        spec_bad_mask = np.array(spec_bad_mask)

        if spec_bad_mask.ndim > 1:
            raise ValueError("The global bad spectral feature mask must be "
                             "one-dimensional. To set the 2D mask, override "
                             "the `spec_bad_masks` attribute.")

        new_spec_X = self.X[:, self._idx_map['spec'][~spec_bad_mask]]
        return self.put_X('spec', new_spec_X,
                          spec_wvln=self.spec_wvln[~spec_bad_mask],
                          copy=copy)

    def patch_spec(self, patching_n_components=None):
        from .config import (patching_n_components
                             as default_patching_n_components)

        if patching_n_components is None:
            patching_n_components = default_patching_n_components

        pca = PCA(n_components=patching_n_components)

        spec_X = self.get_X('spec')[0].copy()
        subX_pca = pca.fit_transform(spec_X)
        tmp_patched = pca.inverse_transform(subX_pca)

        spec_X[spec_X == 0] = tmp_patched[spec_X == 0]

        return self.put_X('spec', sub_X=spec_X)

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

            if mask is not None:
                new_spec_X[i][mask] = fill_value

        return self.put_X('spec', sub_X=new_spec_X, copy=copy)

    ###########################################################################
    # Feature matrix utilities

    def get_X(self, terms=['ones', 'phot', 'lsf', 'spec'],
              phot_names=None):

        if isinstance(terms, str):
            terms = [terms]
        terms = list(terms)

        if phot_names is None:
            phot_names = self._phot_names

        idx_map = self._idx_map.copy()
        _, idx1, idx2 = np.intersect1d(self._phot_names, phot_names,
                                       return_indices=True)
        idx_map['phot'] = idx_map['phot'][idx1[idx2.argsort()]]

        X = None
        idx = []
        new_idx_map = {}
        start = 0

        if 'ones' in terms:
            # Insert the ones column:
            new_idx_map['ones'] = np.array([0])
            start = 1
            X = np.ones((len(self), 1))
            terms.pop(terms.index('ones'))

        for name in terms:
            idx.append(idx_map[name])

            new_idx_map[name] = np.arange(start,
                                          start + len(idx_map[name]))
            start = start + len(idx_map[name])
        idx = np.concatenate(idx)

        if X is None:
            X = self.X[:, idx]
        else:
            X = np.hstack((X, self.X[:, idx]))

        return X, new_idx_map

    def put_X(self, term, sub_X, copy=True, **kwargs):
        terms = ['phot', 'lsf', 'spec']
        terms.pop(terms.index(term))
        X, idx_map = self.get_X(terms)

        idx_map[term] = X.shape[1] + np.arange(sub_X.shape[1])
        X = np.hstack((X, sub_X))

        if copy:
            return self._replicate(X=X, idx_map=idx_map, **kwargs)

        else:
            self._idx_map = idx_map
            self.X = X
            for k in kwargs:
                setattr(self, k, kwargs[k])
            return self

    def get_neighborhood_X(self, color_names, renormalize_colors=True):
        from .filters import renormalize

        X, idx_map = self.get_X('spec')

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

    def get_colors(self, colors):
        """
        colors : iterable of tuples of strings
        """
        color_X = []
        for band1, band2 in colors:
            i1 = self._idx_map['phot'][np.argwhere(
                self._phot_names == band1)[0][0]]
            i2 = self._idx_map['phot'][np.argwhere(
                self._phot_names == band2)[0][0]]
            color_X.append(self.X[:, i1] - self.X[:, i2])

        return np.stack(color_X, axis=1)
