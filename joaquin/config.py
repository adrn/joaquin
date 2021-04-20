# Standard library
from dataclasses import dataclass, fields
import pathlib

# Third-party
import yaml

# Project
# from .logger import logger

__all__ = ['Config']


@dataclass
class Config:
    output_path: (str, pathlib.Path) = None

    apogee_cache_path: (str, pathlib.Path) = None
    apogee_dr: str = None
    apogee_reduction: str = None

    parallax_zpt: float = None

    phot_names: list = None

    neighborhood_pca_components: int = None
    neighborhood_color_names: list = None
    max_neighborhood_size: int = None
    block_size: int = None
    Kfold_K: int = None
    patching_pca_components: int = None

    optimize_train_maxiter: int = None

    seed: int = None

    def __init__(self, filename):
        self._load_validate_config_values(filename)
        # self._cache = {}

    def _load_validate_config_values(self, filename):
        filename = pathlib.Path(filename).expanduser().absolute()
        if not filename.exists():
            raise IOError(f"Config file {str(filename)} does not exist.")

        with open(filename, 'r') as f:
            vals = yaml.safe_load(f.read())

        # Validate types:
        kw = {}
        for field in fields(self):
            default = field.default
            if field.name == 'output_path':
                default = filename.parent

            val = vals.get(field.name, None)
            if val is None:
                val = default

            if val is not None and (field.name.endswith('_file')
                                    or field.name.endswith('_path')):
                val = pathlib.Path(val)

            kw[field.name] = val

        # Normalize paths:
        for k, v in kw.items():
            if isinstance(v, pathlib.Path):
                kw[k] = v.expanduser().absolute()

        # Validate:
        allowed_None_names = ['seed']
        for field in fields(self):
            val = kw[field.name]
            if (not isinstance(val, field.type)
                    and field.name not in allowed_None_names):
                msg = (f"Config field '{field.name}' has type {type(val)}, "
                       f"but should be one of: {field.type}")
                raise ValueError(msg)

            setattr(self, field.name, val)

        # Make sure output file path exists:
        self.output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # File paths
    #

    @property
    def parent_sample_source_file(self):
        return self.output_path / 'parent-sample.fits'

    @property
    def parent_sample_cache_file(self):
        return self.output_path / 'parent-sample.hdf5'

    @property
    def neighborhood_index_file(self):
        return self.output_path / 'good_parent_neighborhood_indices.npy'

    @property
    def parent_closest_stoop_file(self):
        return self.output_path / 'parent_closest_stoop.npz'

    @property
    def plot_path(self):
        path = self.output_path / 'plots'
        path.mkdir(exist_ok=True)
        return path

    # -------------------------------------------------------------------------
    # Special methods
    #
    # def __getstate__(self):
    #     """Ensure that the cache does not get pickled with the object"""
    #     state = {k: v for k, v in self.__dict__.items() if k != '_cache'}
    #     return state.copy()
