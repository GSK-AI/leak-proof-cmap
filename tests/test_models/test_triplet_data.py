import random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from leakproofcmap.models.tripletloss.triplet_loss_datamodule import (
    CMAPPytorchDatasetTripletAcrossLines,
    CMAPPytorchDatasetTripletWithinLines,
    _BaseCMAPPytorchDataset,
)

TEST_RANDOM_SEED = 57


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "cell_id": [
                "cell1",
                "cell1",
                "cell1",
                "cell2",
                "cell2",
                "cell2",
                "cell3",
                "cell3",
                "cell3",
            ],
            "pert_iname": [
                "pert1",
                "pert1",
                "pert2",
                "pert2",
                "pert2",
                "pert3",
                "pert3",
                "pert3",
                "pert4",
            ],
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "feature2": [10, 11, 12, 13, 14, 15, 16, 17, 18],
        }
    )
    features = ["feature1", "feature2"]
    return df, features


def test_cmap_dataset_across_lines(sample_data):
    random.seed(TEST_RANDOM_SEED)
    dataset = CMAPPytorchDatasetTripletAcrossLines(
        df=sample_data[0], features=sample_data[1]
    )
    tensor1, tensor2, tensor3 = dataset[0]

    assert torch.allclose(tensor1, torch.tensor([1, 10]))
    assert torch.allclose(tensor2, torch.tensor([2, 11]))
    assert torch.allclose(tensor3, torch.tensor([5, 14]))


def test_cmap_dataset_within_lines(sample_data):
    random.seed(TEST_RANDOM_SEED)
    dataset = CMAPPytorchDatasetTripletWithinLines(
        df=sample_data[0], features=sample_data[1]
    )
    tensor1, tensor2, tensor3 = dataset[0]

    assert torch.allclose(tensor1, torch.tensor([1, 10]))
    assert torch.allclose(tensor2, torch.tensor([2, 11]))
    assert torch.allclose(tensor3, torch.tensor([3, 12]))
