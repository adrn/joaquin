import numpy as np

# Joaquin
from .joaquin import Joaquin


def get_Kfold_indices(K, train_mask, block_size=None, rng=None):
    """Split indices in the training sample into train and test batches.

    If a ``block_size`` is provided, only the first ``block_size`` stars (i.e.
    the stars in the block) are K-folded and all of the other stars (i.e. the
    zone2 neighborhood stars) are always included in the training set.

    Parameters
    ----------
    K : int
        The number of K-fold batches.
    train_mask : array-like
        If this has a bool dtype, it is converted into an integer index array
        using `numpy.argwhere()`.
    block_size : int (optional)
    rng : `numpy.random.Generator`

    Returns
    -------
    train_indices : ndarray
    test_indices : ndarray
    """

    if rng is None:
        rng = np.random.default_rng()

    if train_mask.dtype is np.dtype(bool):
        train_idx = np.argwhere(train_mask).ravel()
    else:
        train_idx = train_mask

    if block_size is not None:
        assert block_size < len(train_mask)

        # Now split into block and zone 2 stars: the K-fold will
        # only happen on the block stars, and the zone 2 stars
        # will be appended to all blocks
        block_idx = train_idx[:block_size].copy()
        rng.shuffle(block_idx)

        zone2_idx = train_idx[block_size:]
        batch_size = block_size // K

    else:
        # No block / zone2 crap
        block_idx = train_idx.copy()
        rng.shuffle(block_idx)
        zone2_idx = []
        batch_size = len(train_idx) // K

    train_batches = []
    test_batches = []
    for k in range(K):
        if k == K-1:
            test_batch = block_idx[k*batch_size:]
        else:
            test_batch = block_idx[k*batch_size:(k+1)*batch_size]

        train_batch = np.concatenate(
            (block_idx[~np.isin(block_idx, test_batch)], zone2_idx))

        test_batches.append(test_batch)
        train_batches.append(train_batch)

    assert np.all(np.array([
        len(train_batches[i]) + len(test_batches[i])
        for i in range(len(train_batches))]) == len(train_idx))

    return train_batches, test_batches


def Kfold_train_test_split(config, data, K, **kwargs):
    """Generate K train/test data subsets using ``get_Kfold_indices()``

    Parameters
    ----------
    config : `joaquin.config.Config`
    data : `joaquin.data.JoaquinData`
    K : int
        The number of K-fold batches.
    **kwargs
        Passed to ``get_Kfold_indices()``, like ``train_mask`` and
        ``block_size``.
    """
    train_idxs, test_idxs = get_Kfold_indices(K=K, **kwargs)

    for k in range(K):
        train_idx = train_idxs[k]
        test_idx = test_idxs[k]

        test_block = data[test_idx]
        test_X, _ = test_block.get_X(phot_names=config.phot_names)
        test_y = test_block.stars['parallax']
        test_y_ivar = 1 / test_block.stars['parallax_error'] ** 2

        train_block = data[train_idx]
        train_X, idx_map = train_block.get_X(phot_names=config.phot_names)
        train_y = train_block.stars['parallax']
        train_y_ivar = 1 / train_block.stars['parallax_error'] ** 2

        train_joa = Joaquin(
            train_X,
            train_y,
            train_y_ivar,
            idx_map)

        test_joa = Joaquin(
            test_X,
            test_y,
            test_y_ivar,
            idx_map)

        yield train_joa, test_joa
