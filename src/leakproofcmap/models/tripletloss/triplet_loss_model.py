# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Optional, Literal
import lightning as pl
import torch.nn


class CMAPTripletLossLightningModule(pl.LightningModule):
    def __init__(
        self,
        network_shape: list[int],
        lr: float,
        margin: float,
        dropout_rate: Union[None, float],
        train_with_semi_hard_triplets_only: bool = False,
        log_semi_hard_triplet_batch_fraction: bool = False,
        log_val_on_step: bool = False,
    ):
        """Pytorch TripletLoss-based metric module

        Parameters
        ----------
        network_shape : list[int]
            List of integers denoting the network shape
        lr : float
            Learning rate
        margin : float
            Margin to be applied to the scoring function
        dropout_rate : Union[None, float]
            Dropout rate
        train_with_semi_hard_triplets_only : bool, optional
            If True, then the model is trained with only semi-hard triplets, by default
            False
        log_semi_hard_triplet_batch_fraction : bool, optional
            If True, then the fraction of each batch deemed semi-hard is logged at each
            training step, by default False
        log_val_on_step : bool, optional
            If True, then the validation loss is logged at each training step, by
            default False
        """
        super().__init__()
        self.lr = lr
        self.train_with_semi_hard_triplets_only = train_with_semi_hard_triplets_only
        self.log_semi_hard_triplet_batch_fraction = log_semi_hard_triplet_batch_fraction
        self.log_val_on_step = log_val_on_step
        self.save_hyperparameters()

        # Distance ranges between 0 and 2
        self.critereon_cosine_distance = (
            lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y)
        )

        # Critereon ranges from -2 (best) to +2 (worst)
        self.critereon = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=self.critereon_cosine_distance, margin=margin
        )

        self.network = torch.nn.ModuleList()

        for nodes_i in range(1, len(network_shape)):
            is_last_layer = nodes_i == (len(network_shape) - 1)
            self.network.append(torch.nn.BatchNorm1d(network_shape[nodes_i - 1]))
            if dropout_rate is not None and not is_last_layer:
                self.network.append(torch.nn.Dropout(p=dropout_rate))
            self.network.append(
                torch.nn.Linear(
                    network_shape[nodes_i - 1], network_shape[nodes_i], bias=False
                )
            )
            if not is_last_layer:
                self.network.append(torch.nn.ReLU())

    def forward(self, x1, x2, x3):
        for layer in self.network:
            x1 = layer(x1)
            x2 = layer(x2)
            x3 = layer(x3)
        return x1, x2, x3

    def score_numpy(self, x1, x2):
        self.eval()
        with torch.no_grad():
            x1 = torch.tensor(x1)
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            x1 = x1.to(self.device)
            x2 = torch.tensor(x2)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            x2 = x2.to(self.device)
            for layer in self.network:
                x1 = layer(x1)
            for layer in self.network:
                x2 = layer(x2)
            result = self.critereon_cosine_distance(x1, x2).cpu().numpy()
            self.train()
            return result

    def get_embeddings(self, X):
        self.eval()
        with torch.no_grad():
            x1 = torch.tensor(X, device=self.device)
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            x1 = x1.to(self.device)
            for layer in self.network:
                x1 = layer(x1)
            return x1.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, train_idx):
        logits = self.forward(*train_batch)
        if (
            self.log_semi_hard_triplet_batch_fraction
            or self.train_with_semi_hard_triplets_only
        ):
            tdists = torch.nn.functional.triplet_margin_with_distance_loss(
                logits[0],
                logits[1],
                logits[2],
                distance_function=self.critereon_cosine_distance,
                margin=0.2,
                swap=False,
                reduction="none",
            )
            positive_d = self.critereon_cosine_distance(logits[0], logits[1])
            negative_d = self.critereon_cosine_distance(logits[0], logits[2])
            hard_indexes = (negative_d < positive_d).nonzero()
            semi_hard_indexes = ((negative_d > positive_d) & (tdists > 0)).nonzero()
            easy_indexes = ((negative_d > positive_d) & (tdists == 0.0)).nonzero()

            if self.log_semi_hard_triplet_batch_fraction:
                self.log(
                    "fraction_semi_hard_triplets",
                    len(semi_hard_indexes) / len(train_batch[0]),
                    on_step=True,
                )
                self.log(
                    "fraction_hard_triplets",
                    len(hard_indexes) / len(train_batch[0]),
                    on_step=True,
                )
                self.log(
                    "fraction_easy_triplets",
                    len(easy_indexes) / len(train_batch[0]),
                    on_step=True,
                )
        if self.train_with_semi_hard_triplets_only:
            loss = self.critereon(
                logits[0][semi_hard_indexes],
                logits[1][semi_hard_indexes],
                logits[2][semi_hard_indexes],
            )
        else:
            loss = self.critereon(*logits)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        logits = self.forward(*val_batch)
        val_loss = self.critereon(*logits)
        val_accuracy = (
            torch.sum(
                torch.vstack(
                    [
                        self.critereon_cosine_distance(logits[0], logits[1]),
                        self.critereon_cosine_distance(logits[0], logits[2]),
                    ]
                ).argmin(dim=0)
                == 0
            )
            / logits[0].shape[0]
        )
        self.log("val_loss", val_loss, on_step=self.log_val_on_step)
        self.log("val_accuracy", val_accuracy, on_step=self.log_val_on_step)

    def test_step(self, train_batch, train_idx):
        self.eval()
        with torch.no_grad():
            logits = self.forward(*train_batch)
            test_loss = self.critereon(*logits)
            test_accuracy = (
                torch.sum(
                    torch.vstack(
                        [
                            self.critereon_cosine_distance(logits[0], logits[1]),
                            self.critereon_cosine_distance(logits[0], logits[2]),
                        ]
                    ).argmin(dim=0)
                    == 0
                )
                / logits[0].shape[0]
            )
        self.log("test_loss", test_loss, on_epoch=True, on_step=True)
        self.log("test_accuracy", test_accuracy, on_epoch=True, on_step=True)

    @classmethod
    def instantiate_model_from_trial(
        cls,
        trial,
        n_features: int,
        margin: Union[float, Literal["suggest"]] = 0.2,
        embedding_size: Union[int, Literal["suggest"]] = "suggest",
        dropout_rate: Optional[Union[float, None, Literal["suggest"]]] = "suggest",
        train_with_semi_hard_triplets_only: bool = False,
        log_semi_hard_triplet_batch_fraction: bool = False,
    ):
        print("n_features: ", n_features)
        lr = trial.suggest_float("learning_rate", 1e-6, 0.1, log=True)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
        if margin == "suggest":
            margin = trial.suggest_float("margin", 0.0, 2.0)
        if embedding_size == "suggest":
            embedding_size = int(2 ** trial.suggest_int("embedding_size_po2", 5, 9))
        if dropout_rate == "suggest":
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        input_data_size = n_features
        network_layer_sizes = [input_data_size] + [
            trial.suggest_int(f"layer_{n}_size", input_data_size // 2, 4096)
            for n in range(num_hidden_layers)
        ]
        network_layer_sizes.append(embedding_size)
        print(f"Model: {network_layer_sizes=}, {margin=}, {lr=}, {dropout_rate=}")
        trial.set_user_attr("network_layer_sizes", network_layer_sizes)
        model_instantiation_args = {
            "network_shape": network_layer_sizes,
            "lr": lr,
            "margin": margin,
            "dropout_rate": dropout_rate,
            "train_with_semi_hard_triplets_only": train_with_semi_hard_triplets_only,
            "log_semi_hard_triplet_batch_fraction": log_semi_hard_triplet_batch_fraction,
        }
        trial.set_user_attr("model_instantiation_args", model_instantiation_args)
        return cls(**model_instantiation_args)
