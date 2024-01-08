from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

import torch
from mnist.models.model import SimpleCNN
import pytest

# from tests.test_data import get_datasets


def test_model():
    model = SimpleCNN(28, 10)

    assert model(torch.randn(1, 1, 28, 28)).shape == (
        1,
        10,
    ), "Model output is not of shape (1,10)"


def test_model_raise():
    model = SimpleCNN(28, 10)

    x = torch.randn(1, 28, 28)

    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(x)
