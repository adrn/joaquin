# Standard library
import pickle

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d

# Joaquin
from joaquin import Joaquin
from joaquin.data import JoaquinData
from joaquin.config import Config
from joaquin.logger import logger
from joaquin.plot import phot_to_label
from joaquin.crossval import get_Kfold_indices


def worker(task):
    worker_id, conf, data, root_plot_path, seed = task

    rng = np.random.default_rng(seed)

    plot_path = root_plot_path / str(worker_id)
    cache_path = conf.output_path / 'cache' / str(worker_id)
    plot_path.mkdir(exist_ok=True)
    cache_path.mkdir(exist_ok=True, parents=True)

    # TODO: 0.25 is a MAGIC NUMBER and should be configurable
    spec_bad_mask = (data.spec_bad_masks.sum(axis=0) / len(data.stars)) > 0.25
    patched_data, patch_pca = data.patch_spec(conf.patching_pca_components)

    # Mask out the globally bad pixels
    patched_data.spec_bad_masks = None
    patched_data = patched_data.mask_spec_pixels(spec_bad_mask)

    # Need to save "spec_bad_mask", and the PCA used to do patching so
    # that we can apply the model to stars that aren't in the neighborhood, but
    # for which this stoop is the nearest stoop
    np.save(cache_path / 'spec_bad_mask.npy', spec_bad_mask)
    with open(cache_path / 'patch_pca.pkl', 'wb') as f:
        pickle.dump(patch_pca, f)

    # Colors the HRD by index in the neighborhood, which is a proxy for how
    # close a star is to the stoop
    fig = plot_max_neighborhood_by_rank(patched_data)
    fig.savefig(plot_path / 'neighborhood_index_color.png', dpi=200)
    plt.close(fig)

    # How many spectral pixels were patched by the PCA patching?
    tmp, _ = data.get_X('spec')
    npix_fixed = (tmp[:, ~spec_bad_mask] == 0).sum()
    tmp_patched, _ = patched_data.get_X('spec')
    assert (tmp_patched == 0).sum() == 0

    logger.info(f"Neighborhood {worker_id}: {npix_fixed} pixels patched, "
                f"~{npix_fixed/tmp.shape[0]:.0f} pixels patched per star")

    # Low-pass filter the spectra:
    lowpass_data = patched_data.lowpass_filter_spec()

    fig = plot_2D_mean_diff(lowpass_data[(lowpass_data.stars['SNR'] > 300) &
                                         (lowpass_data.stars['LOGG'] > -1)])
    fig.savefig(plot_path / '2D_mean_spec_diff.png', dpi=200)
    plt.close(fig)

    # Cross-validate L2_ivar and training area size:
    data = lowpass_data
    train_mask = np.argwhere(
        lowpass_data.stars['parallax_error'] < 0.1  # TODO: hard-coded
    ).ravel()
    L2_ivar_vals = 10 ** np.arange(0., 5+1, 0.5)  # TODO: hard-coded
    train_sizes = np.array([4096, 8192, 16384, 32768])  # TODO: hard-coded

    train_lls = np.full((len(train_sizes), conf.Kfold_K, len(L2_ivar_vals)),
                        np.nan)
    test_lls = np.full((len(train_sizes), conf.Kfold_K, len(L2_ivar_vals)),
                       np.nan)

    for i, train_size in enumerate(train_sizes):
        train_idxs, test_idxs = get_Kfold_indices(
            K=conf.Kfold_K,
            train_mask=train_mask[:train_size],
            block_size=conf.block_size,
            rng=rng
        )

        for k in np.arange(conf.Kfold_K):
            train_idx = train_idxs[k]
            test_idx = test_idxs[k]

            test_block = data[test_idx]
            test_X, _ = test_block.get_X(phot_names=conf.phot_names)
            test_y = test_block.stars['parallax']
            test_y_ivar = 1 / test_block.stars['parallax_error'] ** 2

            train_block = data[train_idx]
            train_X, idx_map = train_block.get_X(phot_names=conf.phot_names)
            train_y = train_block.stars['parallax']
            train_y_ivar = 1 / train_block.stars['parallax_error'] ** 2

            joa = Joaquin(
                train_X,
                train_y,
                train_y_ivar,
                idx_map)

            test_joa = Joaquin(
                test_X,
                test_y,
                test_y_ivar,
                idx_map)

            for j, L2_ivar in enumerate(L2_ivar_vals):
                frozen = {'L2_ivar': L2_ivar,
                          'parallax_zpt': conf.parallax_zpt}

                init_beta = joa.init_beta(**frozen)

                test_lls[i, k, j] = test_joa.ln_likelihood(beta=init_beta,
                                                           **frozen)[0]
                train_lls[i, k, j] = joa.ln_likelihood(beta=init_beta,
                                                       **frozen)[0]

    # Mean (could sum here) the cross-validation scores
    train_ll = np.mean(train_lls, axis=1)
    test_ll = np.mean(test_lls, axis=1)

    # Plot the cross-validation scores:
    L2_ivar_vals_2d, train_sizes_2d = np.meshgrid(L2_ivar_vals, train_sizes)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, ll, name in zip(axes, [test_ll, train_ll], ['test', 'train']):
        cs = ax.scatter(L2_ivar_vals_2d, train_sizes_2d,
                        c=ll,
                        vmin=np.percentile(ll, 25),
                        vmax=np.percentile(ll, 99.5),
                        marker='s', s=500, cmap='Spectral')
        ax.set_title(name)
        fig.colorbar(cs, ax=ax, aspect=20)
        ax.set_ylabel('train size')

    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.set_xlabel('L2 ivar')
    fig.savefig(plot_path / 'crossval_L2ivar_trainsize.png', dpi=200)
    plt.close(fig)

    cross_val_L2_ivar = L2_ivar_vals_2d.ravel()[test_ll.argmax()]
    cross_val_train_size = train_sizes_2d.ravel()[test_ll.argmax()]
    logger.info(f"Neighborhood {worker_id}: "
                f"Best L2_ivar, train_size = {cross_val_L2_ivar}, "
                f"{cross_val_train_size}\n"
                f"Best L2 stddev: {1 / np.sqrt(cross_val_L2_ivar):.3f}")

    frozen = {'L2_ivar': cross_val_L2_ivar,
              'parallax_zpt': conf.parallax_zpt}

    train_data = data[train_mask[:cross_val_train_size]]
    train_X, idx_map = train_data.get_X(phot_names=conf.phot_names)
    train_y = train_data.stars['parallax']
    train_y_ivar = 1 / train_data.stars['parallax_error'] ** 2

    joa = Joaquin(
        train_X,
        train_y,
        train_y_ivar,
        idx_map,
        frozen=frozen)

    init = joa.init(parallax_zpt=frozen.get('parallax_zpt', conf.parallax_zpt),
                    pack=False)
    res = joa.optimize(init=init,
                       options={'maxiter': conf.optimize_train_maxiter})
    fit_pars = joa.unpack_pars(res.x)
    logger.debug(f"Neighborhood {worker_id}: Optimize result \n{res}")

    for key in ['phot', 'lsf', 'spec']:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax = axes[0]
        if key == 'phot':
            xx = [phot_to_label[x] for x in conf.phot_names]
        else:
            xx = np.arange(len(idx_map[key]))

        ax.plot(xx, init['beta'][idx_map[key]])
        ax.plot(xx, fit_pars['beta'][idx_map[key]])

        axes[1].plot(
            xx,
            fit_pars['beta'][idx_map[key]] - init['beta'][idx_map[key]])

        axes[0].set_title(key)
        axes[1].set_ylabel('optimized - init', fontsize=14)

        fig.savefig(plot_path / f'beta-{key}.png', dpi=200)
        plt.close(fig)

    # ---
    key = 'spec'

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    xx = np.arange(len(idx_map[key]))

    ax.plot(xx, init['beta'][idx_map[key]])
    ax.plot(xx, fit_pars['beta'][idx_map[key]])

    axes[1].plot(xx,
                 fit_pars['beta'][idx_map[key]] - init['beta'][idx_map[key]])

    axes[0].set_title(key)
    axes[1].set_ylabel('optimized - init', fontsize=14)
    axes[0].set_xlim(2500, 3500)
    fig.savefig(plot_path / 'beta-spec-zoomed.png', dpi=200)
    plt.close(fig)

    # ---

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                             sharex=True, sharey=True)

    stars = train_data.stars
    stars['H_W2'] = stars['H'] - stars['w2mpro']
    pred_plx = joa.model_y(train_X, **fit_pars)

    for color_by, ax in zip(['LOGG', 'TEFF', 'M_H', 'H_W2'], axes.flat):
        c = stars[color_by]
        ax.scatter(train_y + fit_pars['parallax_zpt'],
                   pred_plx,
                   c=c,
                   marker='o', s=2, alpha=0.75)
        ax.set_title(color_by)

    ax.set_xlim(-0.5, train_y.max() + 0.1)
    ax.set_ylim(ax.get_xlim())

    for ax in axes[-1]:
        ax.set_xlabel('Gaia parallax')
    for ax in axes[:, 0]:
        ax.set_ylabel('Joaquin parallax')

    fig.tight_layout()
    fig.savefig(plot_path / 'training_plx_vs_plx.png', dpi=200)
    plt.close(fig)


def run_pipeline(config_file, pool, neighborhood_index=None):
    conf = Config(config_file)

    plot_path = conf.plot_path / 'pipeline'
    plot_path.mkdir(exist_ok=True)

    logger.debug("Loading parent sample cache file...")
    parent = JoaquinData.read(conf.parent_sample_cache_file)

    logger.debug("Parent sample cache file loaded - selecting finite data...")
    parent = parent[np.all(np.isfinite(parent.X), axis=1)]

    # Load all parent sample neighborhood indices:
    # (generated in 2-Neighborhoods-PCA.ipynb)
    logger.debug("Loading neighborhood index file...")
    neighborhood_indices = np.load(conf.neighborhood_index_file)
    stoop_ids = np.arange(len(neighborhood_indices))

    if neighborhood_index is not None:
        neighborhood_indices = [neighborhood_indices[neighborhood_index]]
        stoop_ids = [neighborhood_index]

    tasks = [
        (stoop_ids[i], conf, parent[neighborhood_indices[i]], plot_path)
        for i in range(len(neighborhood_indices))]

    seedseq = np.random.SeedSequence(conf.seed)
    seeds = seedseq.spawn(len(tasks))
    tasks = [tuple(t) + (s,) for t, s in zip(tasks, seeds)]

    logger.info(f'Done preparing tasks: {len(tasks)} neighborhoods to process')
    for r in pool.map(worker, tasks):
        pass


def plot_max_neighborhood_by_rank(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             sharex=True,
                             constrained_layout=True)

    axes[0].scatter(data.stars['TEFF'][0],
                    data.stars['LOGG'][0],
                    s=6, color='tab:green', zorder=100)

    stat = binned_statistic_2d(
        data.stars['TEFF'],
        data.stars['LOGG'],
        np.arange(len(data.stars)),
        bins=(np.linspace(3000, 8500, 256),
              np.linspace(-0.5, 5.5, 256)))
    axes[0].pcolormesh(stat.x_edge, stat.y_edge, stat.statistic.T)

    axes[0].set_xlim(stat.x_edge.max(), stat.x_edge.min())
    axes[0].set_ylim(stat.y_edge.max(), stat.y_edge.min())

    axes[0].set_xlabel('TEFF')
    axes[0].set_ylabel('LOGG')

    # ----

    axes[1].scatter(data.stars['TEFF'][0],
                    data.stars['M_H'][0],
                    s=6, color='tab:green', zorder=100)

    stat = binned_statistic_2d(
        data.stars['TEFF'],
        data.stars['M_H'],
        np.arange(len(data.stars)),
        bins=(np.linspace(3000, 8500, 256),
              np.linspace(-2.5, 0.6, 256)))
    axes[1].pcolormesh(stat.x_edge, stat.y_edge, stat.statistic.T)

    axes[1].set_ylim(-2.5, 0.5)

    axes[1].set_xlabel('TEFF')
    axes[1].set_ylabel('M_H')
    return fig


def plot_2D_mean_diff(data):
    tmp, _ = data.get_X('spec')

    fig, ax = plt.subplots(figsize=(10, 10 * tmp.shape[0] / tmp.shape[1]))

    diff = tmp[data.stars['LOGG'].argsort()] - np.median(tmp, axis=0)
    ax.imshow(diff, origin='lower',
              vmin=np.percentile(diff.ravel(), 1),
              vmax=np.percentile(diff.ravel(), 99),
              cmap='RdBu')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('wavelength')
    ax.set_ylabel('spectra - mean, ordered by LOGG')

    fig.tight_layout()
    return fig
