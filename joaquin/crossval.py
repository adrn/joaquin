import numpy as np


def get_Kfold_indices(K, train_mask, block_size, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    if train_mask.dtype is np.dtype(bool):
        train_idx = np.argwhere(train_mask).ravel()
    else:
        train_idx = train_mask

    assert block_size < len(train_mask)

    # Now split into block and zone 2 stars: the K-fold will
    # only happen on the block stars, and the zone 2 stars
    # will be appended to all blocks
    block_idx = train_idx[:block_size].copy()
    rng.shuffle(block_idx)

    zone2_idx = train_idx[block_size:]

    batch_size = block_size // K
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
