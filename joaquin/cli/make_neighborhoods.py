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
from joaquin.logger import logger


def make_neighborhoods(config_file, overwrite=False, seed=None):
    conf = Config(config_file)

    plot_path = conf.plot_path / 'neighborhoods'
    plot_path.mkdir(exist_ok=True)

    # Random number generator, using seed from config file
    rng = np.random.default_rng(seed=conf.seed)

    # TODO: set up logging to file if log directory specified in config!

    # Load parent data - this takes a bit
    logger.debug("Loading parent data set...")
    parent = JoaquinData.read(conf.parent_sample_cache_file)
    parent = parent[np.all(np.isfinite(parent.X), axis=1)]
    logger.debug(f"Parent data loaded: {len(parent)} stars")

    # Define a global spectral mask based on the fraction of pixels over the
    # full parent sample that are masked:
    # MAGIC NUMBER / TODO: 0.25 should probably be a config setting
    global_spec_mask = (parent.spec_bad_masks.sum(axis=0) / len(parent)) > 0.25
    logger.debug(f"{global_spec_mask.sum()} masked spectral pixels out of "
                 f"{len(global_spec_mask)} total spectral pixels")

    # TODO: these numbers or the filter used here should be configurable
    idx = np.argwhere(
        (parent.stars['SNR'] > 200) &   # Strict spectral S/N cut
        (parent.stars['ruwe'] < 1.2) &  # Strict RUWE cut
        (parent.stars['N_COMPONENTS'] == 1)  # No SB2's
    ).ravel()

    # TODO: this number (size) should be configurable
    idx = rng.choice(idx, size=16384, replace=False)
    subset = parent[idx]

    # Spectroscopic HR diagram of the subset stars:
    fig, ax = plt.subplots(figsize=(6, 6))

    teff_logg_bins = (
        np.linspace(3000, 9000, 128),
        np.linspace(-0.5, 5.75, 128))
    ax.hist2d(parent.stars['TEFF'],
              parent.stars['LOGG'],
              bins=teff_logg_bins,
              norm=mpl.colors.LogNorm(),
              cmap='Greys')

    ax.plot(subset.stars['TEFF'],
            subset.stars['LOGG'],
            ls='none', marker='o', mew=0, ms=2.,
            color='tab:blue', alpha=0.75)

    ax.set_xlim(teff_logg_bins[0].max(),
                teff_logg_bins[0].min())
    ax.set_ylim(teff_logg_bins[1].max(),
                teff_logg_bins[1].min())

    ax.set_xlabel('TEFF')
    ax.set_ylabel('LOGG')

    fig.tight_layout()
    fig.savefig(plot_path / 'subset-logg-teff.png', dpi=200)
    plt.close(fig)

    # Construct the neighborhood feature matrix for the neighborhood node sample
    # using colors and spectral pixels:
    color_labels = [f'{phot_to_label[x1]}-{phot_to_label[x2]}'
                    for x1, x2 in conf.neighborhood_color_names]
    tmp = subset.mask_spec_pixels(global_spec_mask)
    subset_X = tmp.get_neighborhood_X(conf.neighborhood_color_names)

    # Corner plot of colors for the subset:
    tmp = subset_X[:, -len(color_labels):]
    fig, _, cb = simple_corner(
        tmp,
        labels=color_labels,
        color_by=subset.stars['LOGG'], vmin=0.5, vmax=5.5,
        colorbar=True)
    cb.ax.set_aspect(40)
    fig.savefig(plot_path / 'neighborhood-colors.png', dpi=200)
    plt.close(fig)

    # Run PCA on the neighborhood node features and project the subset feature
    # matrix onto the PCA basis:
    pca = IncrementalPCA(n_components=conf.neighborhood_pca_components,
                         batch_size=1024)  # HACK
    projected_X = pca.fit_transform(subset_X)
    projected_X /= pca.singular_values_

    # This hacky step removes extreme outliers
    mean = np.mean(projected_X, axis=0)
    std = np.std(projected_X, axis=0)
    bad_mask = np.any(np.abs(projected_X - mean) > 5*std, axis=1)

    neighborhood_node_X = subset_X[~bad_mask]
    neighborhood_node_stars = subset.stars[~bad_mask]

    pca = IncrementalPCA(n_components=conf.neighborhood_pca_components,
                         batch_size=1024)  # HACK
    node_projected_X = pca.fit_transform(neighborhood_node_X)
    node_projected_X /= pca.singular_values_

    logger.debug("Neighborhood PCA cumulative explained variance: "
                 f"{np.cumsum(pca.explained_variance_ratio_)}")

    # Plot the spectroscopic parameters, colored by PCA component
    fig, axes = plt.subplots(3, 3,
                             figsize=(10, 10),
                             sharex=True, sharey=True)

    for i in range(pca.n_components):
        ax = axes.flat[i]
        ax.scatter(neighborhood_node_stars['TEFF'],
                   neighborhood_node_stars['LOGG'],
                   c=node_projected_X[:, i], s=6)
        ax.text(teff_logg_bins[0].max() - 100,
                teff_logg_bins[1].min() + 0.1,
                f'PCA feature {i}', va='top', ha='left')

    for i in range(pca.n_components, len(axes.flat)):
        axes.flat[i].set_visible(False)

    ax.set_xlim(teff_logg_bins[0].max(),
                teff_logg_bins[0].min())
    ax.set_ylim(teff_logg_bins[1].max(),
                teff_logg_bins[1].min())

    fig.tight_layout()
    fig.savefig(plot_path / 'neighborhood-logg-teff-pca.png', dpi=200)
    plt.close(fig)

    # Plot the projected subset data colored by spectroscopic parameters
    things = {
        'TEFF': (3000, 6500),
        'LOGG': (0.5, 5.5),
        'M_H': (-2, 0.5)
    }
    for name, (vmin, vmax) in things.items():
        fig, axes, cb = simple_corner(
            node_projected_X,
            color_by=neighborhood_node_stars[name],
            colorbar=True,
            vmin=vmin, vmax=vmax,
            labels=[f'PCA {i}'
                    for i in range(pca.n_components_)])
        cb.ax.set_aspect(40)
        axes.flat[0].set_title(f'color: {name}')

        fig.savefig(plot_path / f'neighborhood-pca-{name}.png', dpi=200)
        plt.close(fig)

    # Use the neighborhood sample to define the stoops and neighborhoods. We do this by estimating the density (in projected data space),

    tmp = parent.mask_spec_pixels(global_spec_mask)
    parent_X = tmp.get_neighborhood_X(conf.neighborhood_color_names)
