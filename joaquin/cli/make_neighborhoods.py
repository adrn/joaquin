# Standard library
import os
import warnings
import pickle

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange

from scipy.spatial import cKDTree
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KernelDensity

# Joaquin
from joaquin.data import JoaquinData
from joaquin.config import Config
from joaquin.plot import simple_corner, phot_to_label, plot_hr_cmd


def make_neighborhoods(config_file, overwrite=False, seed=None):
    c = Config(config_file)
    pass
