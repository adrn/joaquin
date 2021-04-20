import numpy as np


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
