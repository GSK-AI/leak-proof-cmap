# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import optuna.trial
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping


class OptunaPyTorchLightningPruningCallback(EarlyStopping):
    """PyTorch Lightning callback to prune unpromising trials.


    Reproduced from Optuna source repository as optuna version supporting
    newer pytorch lightning was not available as a package.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super(OptunaPyTorchLightningPruningCallback, self).__init__(monitor=monitor)
        self._trial = trial

    def _process(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

    # NOTE (crcrpar): This method is called <0.8.0
    def on_epoch_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._process(trainer, pl_module)

    # NOTE (crcrpar): This method is called >=0.8.0
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._process(trainer, pl_module)
