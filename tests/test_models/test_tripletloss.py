from unittest.mock import MagicMock, patch

import pytest
import torch

from leakproofcmap.models.tripletloss import CMAPTripletLossLightningModule

TEST_RANDOM_SEED = 57


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (
            torch.tensor([1.0, 0.0]).unsqueeze(0),
            torch.tensor([-1.0, 0.0]).unsqueeze(0),
            2.0,
        ),  # opposite vectors
        (
            torch.tensor([1.0, 0.0]).unsqueeze(0),
            torch.tensor([0.0, 1.0]).unsqueeze(0),
            1.0,
        ),  # orthogonal vectors
        (
            torch.tensor([1.0, 0.0]).unsqueeze(0),
            torch.tensor([1.0, 0.0]).unsqueeze(0),
            0.0,
        ),  # identical vectors
    ],
)
def test_cosine_distance(x, y, expected):
    model = CMAPTripletLossLightningModule(
        network_shape=[10, 20, 30],
        lr=0.001,
        margin=1.0,
        dropout_rate=0.5,
        train_with_semi_hard_triplets_only=False,
        log_semi_hard_triplet_batch_fraction=False,
        log_val_on_step=False,
    )
    result = model.critereon_cosine_distance(x, y)
    assert torch.isclose(result, torch.tensor(expected), atol=1e-5)


def test_init():
    model = CMAPTripletLossLightningModule([10, 20], 0.01, 0.5, 0.1)
    assert model.lr == 0.01
    assert model.train_with_semi_hard_triplets_only == False
    assert model.log_semi_hard_triplet_batch_fraction == False
    assert model.log_val_on_step == False


@pytest.mark.parametrize(
    "batch_size, input_dim",
    [
        (3, 10),
        (5, 10),
        (2, 10),
    ],
)
def test_forward_shape(batch_size, input_dim):
    model = CMAPTripletLossLightningModule([10, 20], 0.01, 0.5, 0.1)
    x1, x2, x3 = (
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
    )
    out1, out2, out3 = model.forward(x1, x2, x3)
    assert out1.shape == torch.Size([batch_size, 20])
    assert out2.shape == torch.Size([batch_size, 20])
    assert out3.shape == torch.Size([batch_size, 20])


def test_forward():
    torch.manual_seed(TEST_RANDOM_SEED)
    model = CMAPTripletLossLightningModule([2, 5], 0.01, 0.5, 0.1)
    batch_size, input_dim = 2, 2

    tensor1 = torch.tensor(
        [
            [0.4172, 0.5788, 0.2695, 0.3752, -0.6927],
            [-0.4172, -0.5788, -0.2695, -0.3752, 0.6927],
        ]
    )

    tensor2 = torch.tensor(
        [
            [0.4178, 0.5808, 0.2706, 0.3751, -0.6943],
            [-0.4178, -0.5808, -0.2706, -0.3751, 0.6943],
        ]
    )

    tensor3 = torch.tensor(
        [
            [0.0261, -0.8272, -0.4883, 0.5458, 0.3937],
            [-0.0261, 0.8272, 0.4883, -0.5458, -0.3937],
        ]
    )

    ground_truth = (tensor1, tensor2, tensor3)

    x1, x2, x3 = (
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
    )
    with torch.no_grad():
        outputs = model.forward(x1, x2, x3)

    torch.testing.assert_close(outputs, ground_truth, atol=1e-3, rtol=1e-5)


@patch(
    "torch.nn.functional.triplet_margin_with_distance_loss",
    return_value=torch.tensor(0.5),
)
def test_training_step(mock_loss):
    model = CMAPTripletLossLightningModule([10, 20], 0.01, 0.5, 0.1)
    batch_size, input_dim = 10, 10
    train_batch = (
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
    )
    train_idx = 0
    loss = model.training_step(train_batch, train_idx)
    assert loss == 0.5


@patch(
    "torch.nn.functional.triplet_margin_with_distance_loss",
    return_value=torch.tensor(0.5),
)
def test_validation_step(mock_loss):
    model = CMAPTripletLossLightningModule([10, 20], 0.01, 0.5, 0.1)
    batch_size, input_dim = 10, 10
    val_batch = (
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
    )
    val_idx = 0
    model.validation_step(val_batch, val_idx)
    mock_loss.assert_called()


@patch(
    "torch.nn.functional.triplet_margin_with_distance_loss",
    return_value=torch.tensor(0.5),
)
def test_test_step(mock_loss):
    model = CMAPTripletLossLightningModule([10, 20], 0.01, 0.5, 0.1)
    batch_size, input_dim = 10, 10
    test_batch = (
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, input_dim),
    )
    test_idx = 0
    model.test_step(test_batch, test_idx)
    mock_loss.assert_called()
