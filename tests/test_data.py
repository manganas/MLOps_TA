from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
from mnist.data.make_dataset import CorruptMnist


def test_data():
    train_dataset = CorruptMnist(
        True, in_folder="data/raw", out_folder="data/processed"
    )
    test_dataset = CorruptMnist(
        False, in_folder="data/raw", out_folder="data/processed"
    )

    assert (
        len(train_dataset) == 45000
    ), "Training dataset does not have 45000 data points"

    assert len(test_dataset) == 5000, "Test dataset does not have 5000 data points"

    # assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    for elem in train_dataset:
        assert elem[0].shape == (
            1,
            28,
            28,
        ), "Training data point does not have shape (1,28,28)"

    for elem in test_dataset:
        assert elem[0].shape == (
            1,
            28,
            28,
        ), "Test data point does not have shape (1,28,28)"

    # assert that all labels are represented
    assert (
        len(set(train_dataset.targets.numpy())) == 10
    ), "Training dataset does not have all labels"

    assert (
        len(set(test_dataset.targets.numpy())) == 10
    ), "Test dataset does not have all labels"

    assert (
        min(set(train_dataset.targets.numpy())) == 0
        and max(set(train_dataset.targets.numpy())) == 9
    ), "Training dataset does not have all labels"

    assert (
        min(set(test_dataset.targets.numpy())) == 0
        and max(set(test_dataset.targets.numpy())) == 9
    ), "Test dataset does not have all labels"

    # assert that the data is normalized
    assert train_dataset.data.max() <= 1.0, "Training dataset is not normalized to 1.0"

    assert train_dataset.data.min() >= 0.0, "Training dataset is not normalized to 0.0"

    assert test_dataset.data.max() <= 1.0, "Test dataset is not normalized to 1.0"

    assert test_dataset.data.min() >= 0.0, "Test dataset is not normalized to 0.0"
