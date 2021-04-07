import numpy as np


def get_Kfold_indices(N, K, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    idx = np.arange(N)
    rng.shuffle(idx)

    batch_size = N // K
    train_batches = []
    test_batches = []
    for k in range(K):
        if k == K-1:
            batch = idx[k*batch_size:]
        else:
            batch = idx[k*batch_size:(k+1)*batch_size]

        test_batches.append(batch)
        train_batches.append(idx[~np.isin(idx, batch)])

    assert np.all(np.array([len(train_batches[i]) + len(test_batches[i])
                            for i in range(len(train_batches))]) == N)

    return train_batches, test_batches
