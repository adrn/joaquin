# Third-party
import numpy as np
import pytest

# Project
from ..crossval import get_Kfold_indices


@pytest.mark.parametrize("K", [2, 4, 7])
@pytest.mark.parametrize(
    "train_mask",
    [np.ones(132, dtype=bool),  # MAGIC NUMBER
     np.arange(173, dtype=int)])  # MAGIC NUMBER
@pytest.mark.parametrize("block_size", [None, 32])
def test_get_Kfold_indices(K, train_mask, block_size):
    rng = np.random.default_rng(seed=42)

    train_is, test_is = get_Kfold_indices(K=K, train_mask=train_mask,
                                          block_size=block_size, rng=rng)

    assert len(np.unique(np.concatenate(train_is))) == len(train_mask)
    assert len(train_is) == len(test_is) == K

    if block_size and train_mask.dtype is np.dtype(int):
        assert np.all([
            np.in1d(train_mask[block_size:], x) for x in train_is])
