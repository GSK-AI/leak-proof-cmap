# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path
from lightning import LightningModule, LightningDataModule
from typing import Optional, Union, Type, Literal
import os
import json
from optuna.integration import PyTorchLightningPruningCallback
from time import sleep
from .phenonaut_functions import (
    get_cmap_phenonaut_object,
    get_phenonaut_object_containing_tests_from_split_object,
)
from .metrics import PhenotypicMetric
from .model_to_metric import get_phenonaut_metric_from_model_checkpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from multiprocessing import cpu_count
from hashlib import sha1
import pandas as pd
import numpy as np
import optuna
from phenonaut import Phenonaut
from .cmap_split import CMAPSplit
from .benchmarking import (
    auroc_performance,
    pertmutation_test_distinct_from_dmso,
)


class LeakProofCMAPLightningModelControler:
    def __init__(
        self,
        working_dir: Path,
        model_name: str,
        random_state: Union[None, int],
        uninstantiated_model: Type[LightningModule],
        uninstantiated_data_module: Type[LightningDataModule],
        phenonaut_packaged_dataset_dir: Path,
    ) -> None:
        self.random_state = random_state
        if self.random_state is not None:
            seed_everything(self.random_state, workers=True)
        self.working_dir = working_dir
        if not self.working_dir.exists():
            raise FileNotFoundError(f"working_dir ({self.working_dir}) must exist.")

        self.pickles_dir = (working_dir / "pickles").resolve()
        if not self.pickles_dir.exists():
            raise FileNotFoundError(f"pickles_dir ({self.pickles_dir}) must exist.")

        self.phenonaut_packaged_dataset_dir = phenonaut_packaged_dataset_dir

        self.model_name = model_name
        self.uninstantiated_model = uninstantiated_model
        self.uninstantiated_data_module = uninstantiated_data_module
        self.model = None

    def _check_class(self):
        """Check model class implements all required methods

        Unfortunately, we cannot use an Abstract Base Class to ensure uninstantiated model has the
        required functions, as LightningModule makes many calls to super which would be interfered
        with, therefore we check in this class that the leakproofcmap required functions are present.

        Raises
        ------
        AttributeError
            The class passed to LeakProofCMAPLightningModelControler must implement the following
            methods:
            - score_numpy
        """
        if self.model is None:
            return
        if not hasattr(self.mode, "score_numpy"):
            raise AttributeError(
                "Expected LightningModule derived model to have a score_numpy method"
            )

    def optuna_scan(
        self,
        n_workers: int,
        n_optuna_trials_per_job: int = 1000,
        instantiate_model_from_trial_kwargs: dict = {},
        db_filename_suffix: Optional[str] = None,
        max_epochs: int = 100,
        reseed: Union[None, int, Literal["suggest"]] = "suggest",
        use_median_pruner: bool = True,
    ):
        self._check_class()
        if isinstance(instantiate_model_from_trial_kwargs, str):
            instantiate_model_from_trial_kwargs = json.loads(
                instantiate_model_from_trial_kwargs
            )

        # Set up a short hash name for the optuna scan before we do any alteration of
        # the random_state.
        parameters_short_hash = sha1(
            str(
                instantiate_model_from_trial_kwargs
                | {"random_state": self.random_state, "model_name": self.model_name}
            ).encode("UTF-8")
        ).hexdigest()[:10]

        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if slurm_task_id is not None:
            slurm_task_id = int(slurm_task_id)
            print("SLURM_ARRAY_TASK_ID:", slurm_task_id)

        if isinstance(reseed, int):
            seed_everything(reseed, workers=True)
        elif reseed == "suggest":
            if slurm_task_id is None:
                raise ValueError(
                    "Could not suggest a value for reseed argument as slurm_task_id was None."
                    "  Manually set seed value to an int, or turn of suggestions by passing None"
                )
            self.random_state = self.random_state + slurm_task_id
            seed_everything(self.random_state, workers=True)

        # To avoid a race condition in checking if the target optuna database file exists if run
        # as a slurm job array split over nodes, check if the SLURM_ARRAY_TASK_ID is set, and if not
        # 0, then wait a bit, allowing task 0 to check for the existence of a .db file, and then
        # make it if it is not found.
        if isinstance(slurm_task_id, int):
            if slurm_task_id != 0:
                print("sleeping...")
                sleep(10 + (5 * slurm_task_id))
                print("awake")

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )

        # Set up the optuna hyperparameter scan

        optuna_db_path = (
            self.working_dir
            / Path(self.model_name)
            / f"optuna_scan_{parameters_short_hash}.db"
        )
        if not optuna_db_path.parent.exists():
            optuna_db_path.parent.mkdir(parents=True)

        if db_filename_suffix is not None:
            optuna_db_path = str(optuna_db_path)[:-3] + "_" + db_filename_suffix + ".db"

        n_optuna_trials: int = n_optuna_trials_per_job

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        optuna_study = optuna.create_study(
            sampler=sampler,
            study_name=self.model_name + parameters_short_hash,
            storage=f"sqlite:///{optuna_db_path}",
            load_if_exists=True,
            direction="minimize",
            pruner=(
                optuna.pruners.MedianPruner(n_warmup_steps=5)
                if use_median_pruner
                else None
            ),
        )
        optuna_study.set_user_attr("model_name", self.model_name)
        optuna_study.set_user_attr(
            "instantiate_model_from_trial_kwargs", instantiate_model_from_trial_kwargs
        )
        optuna_study.set_user_attr("random_state", self.random_state)

        # Define optuna objective function
        def _optuna_objective_func(trial):
            batch_size = trial.suggest_int("batch_size", 32, 4096)
            print(f"Training parameters: {batch_size=}")

            # All optuna scans are done with the first cell line and MOA split held out
            splits = CMAPSplit.load(
                self.working_dir
                / Path("split_data")
                / Path("cmap_split_cellidsplit1_moasplit1.json")
            )

            trainer = Trainer(
                logger=False,
                log_every_n_steps=5,
                accelerator="gpu",
                devices=1,
                max_epochs=max_epochs,
                enable_progress_bar=True,
                callbacks=(
                    [
                        EarlyStopping("val_loss", mode="min"),
                        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                    ]
                    if use_median_pruner
                    else [EarlyStopping("val_loss", mode="min")]
                ),
                enable_checkpointing=False,
                deterministic=True,
            )
            data_module = self.uninstantiated_data_module(
                phe.df,
                phe.ds.features,
                splits,
                batch_size=batch_size,
                num_workers=n_workers,
            )
            model = self.uninstantiated_model.instantiate_model_from_trial(
                trial,
                int(splits.pca.n_components_),
                **instantiate_model_from_trial_kwargs,
            )
            trainer.fit(
                model,
                data_module,
            )
            trial.set_user_attr("n_epochs", int(trainer.current_epoch))
            trial.set_user_attr("loss", trainer.callback_metrics["loss"].item())
            trial.set_user_attr("val_loss", trainer.callback_metrics["val_loss"].item())
            trial.set_user_attr(
                "val_accuracy",
                trainer.callback_metrics["val_accuracy"].item(),
            )
            return trainer.callback_metrics["val_loss"].item()

        optuna_study.optimize(
            _optuna_objective_func,
            n_trials=n_optuna_trials,
        )

    def cross_fold(
        self,
        optuna_db_file: Path,
        model_index: int,
        num_workers: Optional[int] = None,
        sort_key: str = "val_loss",
        sort_key_ascending=True,
        xval_dir_suffix: str = "",
    ):
        print(optuna_db_file)
        print(model_index)
        print(num_workers)
        if num_workers is None:
            num_workers = cpu_count()

        if xval_dir_suffix != "" and not xval_dir_suffix.startswith("_"):
            xval_dir_suffix = "_" + xval_dir_suffix

        optuna_db_file = Path(optuna_db_file)
        xval_dir = (
            self.working_dir
            / Path(self.model_name)
            / Path("xval")
            / (
                (str(optuna_db_file.stem).replace(".db", ""))
                + f"key_{sort_key}"
                + xval_dir_suffix
            )
        )
        xval_dir.mkdir(parents=True, exist_ok=True)

        cell_line_split_id = 1
        compound_split_id = 1

        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        splits = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path(
                f"cmap_split_cellidsplit{cell_line_split_id}_moasplit{compound_split_id}.json"
            )
        )

        study = optuna.load_study(
            storage=f"sqlite:///{optuna_db_file}",
            study_name=f"{self.model_name}{optuna_db_file.stem.split('_')[2]}",
        )
        print(f"{study.user_attrs=}")

        optuna_df = study.trials_dataframe()
        optuna_df = (
            optuna_df.rename(columns={"number": "trial", "value": "val_loss"})
            .set_index("trial")
            .sort_values(sort_key, ascending=sort_key_ascending)
        )

        optuna_df = optuna_df.query("state=='COMPLETE'")
        model_dict = optuna_df.iloc[model_index].to_dict()
        print(
            f"Found {optuna_df.shape[0]} trials in file {optuna_db_file}, crossfolding model {model_index} which has a model_dict of:"
        )
        print(model_dict)

        xval_df_dict = {
            "fold": [],
            "epochs": [],
            "loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        model_instantiation_args = model_dict["user_attrs_model_instantiation_args"]

        batch_size = int(model_dict["params_batch_size"])
        trial_number = optuna_df.iloc[model_index].name
        print(
            "Working on",
            self.model_name,
            f"{optuna_db_file.stem.split('optuna_scan_')[1].replace('.db','')}_{trial_number}",
        )
        log_dir = xval_dir / Path("lightning_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        for fold_number in range(1, 6):
            splits.set(
                use_train_splits=[fold_number],
                use_val_splits=[fold_number],
                use_test_splits=None,
            )
            logger = TensorBoardLogger(
                save_dir=log_dir, name=f"trial{trial_number}_fold{fold_number}"
            )
            model = self.uninstantiated_model(**model_instantiation_args)

            trainer = Trainer(
                deterministic=True,
                logger=logger,
                log_every_n_steps=1,
                accelerator="gpu",
                devices=1,
                max_epochs=300,
                enable_progress_bar=True,
                callbacks=[EarlyStopping("val_loss", mode="min")],
                enable_checkpointing=False,
            )

            trainer.fit(
                model,
                self.uninstantiated_data_module(
                    phe.df,
                    phe.ds.features,
                    splits,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ),
            )
            xval_df_dict["fold"].append(fold_number)
            xval_df_dict["epochs"].append(trainer.current_epoch + 1)
            xval_df_dict["loss"].append(trainer.callback_metrics["loss"].item())
            xval_df_dict["val_loss"].append(trainer.callback_metrics["val_loss"].item())
            xval_df_dict["val_accuracy"].append(
                trainer.callback_metrics["val_accuracy"].item()
            )

        xval_df = pd.DataFrame.from_dict(xval_df_dict)
        xval_df.to_csv(xval_dir / f"xval_scores_t{trial_number}.csv")
        print(xval_df.describe())

    def cross_fold_add_random_baseline(
        self,
        optuna_db_file: Path,
        num_workers: Optional[int] = None,
        sort_key: str = "val_loss",
        sort_key_ascending=True,
        xval_dir_suffix: str = "",
    ):
        np_rng = np.random.default_rng(self.random_state)
        print(optuna_db_file)
        print(num_workers)
        if num_workers is None:
            num_workers = cpu_count()

        if xval_dir_suffix != "" and not xval_dir_suffix.startswith("_"):
            xval_dir_suffix = "_" + xval_dir_suffix

        optuna_db_file = Path(optuna_db_file)
        xval_dir = (
            self.working_dir
            / Path(self.model_name)
            / Path("xval")
            / (
                (str(optuna_db_file.stem).replace(".db", ""))
                + f"key_{sort_key}"
                + xval_dir_suffix
            )
        )
        xval_dir.mkdir(parents=True, exist_ok=True)

        cell_line_split_id = 1
        compound_split_id = 1

        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        splits = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path(
                f"cmap_split_cellidsplit{cell_line_split_id}_moasplit{compound_split_id}.json"
            )
        )

        study = optuna.load_study(
            storage=f"sqlite:///{optuna_db_file}",
            study_name=f"{self.model_name}{optuna_db_file.stem.split('_')[2]}",
        )
        print(f"{study.user_attrs=}")

        optuna_df = study.trials_dataframe()
        optuna_df = (
            optuna_df.rename(columns={"number": "trial", "value": "val_loss"})
            .set_index("trial")
            .sort_values(sort_key, ascending=sort_key_ascending)
        )

        optuna_df = optuna_df.query("state=='COMPLETE'")

        xval_df_dict = {
            "fold": [],
            "epochs": [],
            "loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        print("Working on random baseline")
        for fold_number in range(1, 6):
            splits.set(
                use_train_splits=[fold_number],
                use_val_splits=[fold_number],
                use_test_splits=None,
            )
            val_dataloader = self.uninstantiated_data_module(
                phe.df,
                phe.ds.features,
                splits,
                batch_size=1000,
                num_workers=num_workers,
            ).val_dataloader()

            random_guesses = []
            for batch in val_dataloader:
                n_in_batch = len(batch[0])
                random_guesses.extend(np_rng.binomial(1, 0.5, n_in_batch))

            xval_df_dict["fold"].append(fold_number)
            xval_df_dict["epochs"].append(pd.NA)
            xval_df_dict["loss"].append(pd.NA)
            xval_df_dict["val_loss"].append(pd.NA)
            xval_df_dict["val_accuracy"].append(np.mean(random_guesses))

        xval_df = pd.DataFrame.from_dict(xval_df_dict)
        xval_df.to_csv(xval_dir / "xval_scores_random_chance.csv")
        print(xval_df.describe())

    def train(
        self,
        unique_model_name: str,
        n_workers: Union[int, None],
        n_epochs: int = 20,
        data_split_index: int = 11,
        model_init_kwargs: dict = {},
        merge_val_into_train: bool = True,
        compose_triplets_across_cell_lines: bool = True,
        compose_triplets_match_line_and_dose: bool = False,
    ):
        """Train a model

        Parameters
        ----------
        unique_model_name : str
            A unique name for the model - typically an Optuna trial number or similar
        n_workers : Union[int, None]
            Number of worker processes that the dataloader should use. If None, then the
            number is determined with a call to multiprocessing.cpu_count
        n_epochs : int
            Maximum number of epochs to train model for. When the trainer reaches epoch
            n_epochs+1, the training loop is exited.
        data_split_index : Union[None, int]
            A 2-digit integer with the first digit denoting the cell line split to use,
            and the second denoting the MOA split. By default 11
        model_init_kwargs : dict, optional
            Initialise model with additional arguments within this dictionary, by default {}

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        KeyError
            _description_
        """
        if n_workers is None:
            n_workers = cpu_count()

        if data_split_index is None:
            CELL_LINE_SPLIT = 1
            MOA_SPLIT = 1
        else:
            print(data_split_index, type(data_split_index))
            CELL_LINE_SPLIT = data_split_index // 10
            MOA_SPLIT = data_split_index % 10

        self._check_class()
        if isinstance(model_init_kwargs, str):
            model_init_kwargs = json.loads(model_init_kwargs)

        if "batch_size" not in model_init_kwargs:
            raise KeyError("batch_size must be present in model_init_kwargs")
        batch_size = model_init_kwargs["batch_size"]
        del model_init_kwargs["batch_size"]

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
        )
        checkpoint_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger = TensorBoardLogger(
            save_dir=output_dir / "lightning_logs",
            name=f"cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}",
        )

        splits = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path(f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json")
        )
        if merge_val_into_train:
            splits.merge_val_into_train(phe.df, phe.ds.features)
        model = self.uninstantiated_model(**model_init_kwargs)
        trainer = Trainer(
            logger=logger,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=1,
            max_epochs=n_epochs + 1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            deterministic=True,
        )
        trainer.fit(
            model,
            self.uninstantiated_data_module(
                phe.df,
                phe.ds.features,
                splits,
                batch_size=batch_size,
                num_workers=n_workers,
                compose_triplets_across_cell_lines=compose_triplets_across_cell_lines,
                compose_triplets_within_cell_lines_and_match_dose=compose_triplets_match_line_and_dose,
            ),
        )
        trainer.save_checkpoint(
            checkpoint_dir
            / f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
        )
        if merge_val_into_train:
            trainer.test(
                model=model,
                dataloaders=self.uninstantiated_data_module(
                    phe.df,
                    phe.ds.features,
                    splits,
                    batch_size=batch_size,
                    compose_triplets_across_cell_lines=compose_triplets_across_cell_lines,
                ).test_dataloader(),
            )
            test_accuracy_output_dir = (
                self.working_dir
                / Path("plot_data")
                / Path(f"{self.model_name}")
                / Path(f"{unique_model_name}")
                / Path("test_accuracies")
            )
            test_accuracy_output_dir.mkdir(exist_ok=True, parents=True)
            test_accuracy_output_file = test_accuracy_output_dir / Path(
                f"test_accuracy_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.csv"
            )
            pd.DataFrame(
                {
                    "cell line split": [CELL_LINE_SPLIT],
                    "MOA split": [MOA_SPLIT],
                    "seed": [self.random_state],
                    "test accuracy": [trainer.callback_metrics["test_accuracy"].item()],
                }
            ).to_csv(test_accuracy_output_file)
            return trainer.callback_metrics["test_accuracy"].item()
        else:
            trainer.validate(
                model=model,
                dataloaders=self.uninstantiated_data_module(
                    phe.df,
                    phe.ds.features,
                    splits,
                    batch_size=batch_size,
                    compose_triplets_across_cell_lines=compose_triplets_across_cell_lines,
                ).val_dataloader(),
            )
            return trainer.callback_metrics["val_accuracy"].item()

    def train_with_all_data(
        self,
        unique_model_name: str,
        n_workers: Union[int, None],
        n_epochs: int,
        model_init_kwargs: dict = {},
    ) -> None:
        """Train model with all data

        Parameters
        ----------
        unique_model_name : str
            A unique name for the model - typically an Optuna trial number or similar
        n_workers : Union[int, None]
            Number of worker processes that the dataloader should use. If None, then the number is
            determined with a call to multiprocessing.cpu_count.
        n_epochs : int
            Number of epochs to train model for
        data_split_index : Union[None, int]
            A 2-digit integer with the first digit denoting the cell line split to use, and the
            second denoting the MOA split. By default 11
        model_init_kwargs : dict, optional
            Initialise model with additional arguments within this dictionary, by default {}

        Raises
        ------
        KeyError
            _description_
        """
        if n_workers is None:
            n_workers = cpu_count()

        self._check_class()
        if isinstance(model_init_kwargs, str):
            model_init_kwargs = json.loads(model_init_kwargs)

        if "batch_size" not in model_init_kwargs:
            raise KeyError("batch_size must be present in model_init_kwargs")
        batch_size = model_init_kwargs["batch_size"]
        del model_init_kwargs["batch_size"]

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
        )
        checkpoint_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger = TensorBoardLogger(
            save_dir=output_dir / "lightning_logs",
            name=f"all_data_seed_{self.random_state}",
        )

        splits = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path("cmap_split_cellidsplit1_moasplit1.json")
        )
        splits.merge_all_data_into_train()

        model = self.uninstantiated_model(**model_init_kwargs)
        trainer = Trainer(
            logger=logger,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=1,
            max_epochs=n_epochs,
            enable_progress_bar=True,
            enable_checkpointing=False,
            deterministic=True,
        )
        trainer.fit(
            model,
            self.uninstantiated_data_module(
                phe.df,
                phe.ds.features,
                splits,
                batch_size=batch_size,
                num_workers=n_workers,
                compose_triplets_across_cell_lines=True,
                compose_triplets_within_cell_lines_and_match_dose=False,
            ),
        )
        trainer.save_checkpoint(
            checkpoint_dir / f"cmap_all_data_seed_{self.random_state}.ckpt"
        )

    def calc_compactness(
        self,
        unique_model_name: str,
        n_workers: int,
        data_split_index: int,
        pr_type=Union[
            Literal["pr"],
            Literal["pr_moas"],
            Literal["pr_across_lines"],
            Literal["pr_across_lines_moas"],
        ],
        calc_gpu_only_prs: bool = True,
        calc_cpu_only_prs: bool = True,
        jump_moa_compounds_only: bool = False,
        cuda_for_gpu_metrics: bool = True,
    ):
        allowed_pr_types = {
            "compactness": {
                "perturbation_column": "pert_iname",
                "replicate_criteria": ["pert_idose_uM", "cell_id"],
                "null_criteria": ["pert_idose_uM", "cell_id"],
            },
            "compactness_moas": {
                "perturbation_column": "moa",
                "replicate_criteria": ["cell_id"],
                "null_criteria": ["cell_id"],
            },
            "compactness_across_lines": {
                "perturbation_column": "pert_iname",
                "replicate_criteria_not": ["cell_id"],
                "null_criteria": ["cell_id"],
            },
            "compactness_across_lines_moas": {
                "perturbation_column": "moa",
                "replicate_criteria_not": ["cell_id"],
                "null_criteria": ["cell_id"],
            },
        }
        if pr_type not in allowed_pr_types:
            raise ValueError(
                f"The given value for pr_type ({pr_type} was not found in allowed_pr_types ({list(allowed_pr_types.keys())})"
            )

        CELL_LINE_SPLIT = data_split_index // 10
        MOA_SPLIT = data_split_index % 10

        self._check_class()

        if jump_moa_compounds_only:
            from .jump_moa_compounds import jump_moa_compound_pert_iname_list

            print("Running with jump moa compounds only")
            jump_moa_pert_inames = jump_moa_compound_pert_iname_list.copy()
            jump_moa_pert_inames.remove("DMSO")

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        checkpoint_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
            / "checkpoints"
        )
        output_dir = (
            self.working_dir
            / Path("plot_data")
            / Path("TripletLoss")
            / Path(f"{unique_model_name}")
            / Path(pr_type)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        from .benchmarking.percent_replicating import percent_replicating

        if calc_gpu_only_prs:
            if not cuda_for_gpu_metrics:
                print("Evaluating GPU metric without CUDA")
            metric = get_phenonaut_metric_from_model_checkpoint(
                "TripletLoss",
                self.uninstantiated_model,
                checkpoint_dir
                / Path(
                    f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
                ),
                higher_is_better=False,
                cuda=cuda_for_gpu_metrics,
            )

            new_phe = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=True,
                pca=True,
            )

            embeddings = metric.func.model.get_embeddings(new_phe.ds.data.values)
            new_features_for_embeddings = [
                f"TL_feat_{i}" for i in range(1, embeddings.shape[1] + 1)
            ]
            new_phe.ds.df = pd.concat(
                [
                    new_phe.ds.df,
                    pd.DataFrame(
                        embeddings,
                        columns=new_features_for_embeddings,
                        index=new_phe.ds.df.index,
                    ),
                ],
                axis=1,
            )
            old_features = new_phe.ds.features.copy()
            new_phe.ds.features = (
                new_features_for_embeddings,
                "Updated features to TripletLoss derived features",
            )
            new_phe.ds.drop_columns(
                old_features,
                "Removing features as they are superseeded by TripletLoss derived embeddings",
            )
            pr, pr_df = percent_replicating(
                new_phe,
                **allowed_pr_types[pr_type],
                restrict_evaluation_query=(
                    f"pert_iname in {jump_moa_pert_inames}"
                    if jump_moa_compounds_only
                    else None
                ),
                similarity_metric="cosine",
                similarity_metric_higher_is_better=False,
                parallel=False,
                return_full_performance_df=True,
                n_jobs=n_workers,
            )
            pr_df.to_csv(
                output_dir
                / f"pr_{metric.name}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            )
            print(f"PR {metric.name} = {pr}")

        if calc_cpu_only_prs:
            from .metrics import leak_proof_cmap_metrics

            new_phe = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=False,
                pca=False,
            )

            new_phe_scaled_pca = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=True,
                pca=True,
            )

            for metric_name, metric in leak_proof_cmap_metrics.items():
                print(
                    f"Working on {metric_name} for cls {CELL_LINE_SPLIT} and moas {MOA_SPLIT} for {pr_type}"
                )
                pr, pr_df = percent_replicating(
                    (
                        new_phe_scaled_pca
                        if (
                            (hasattr(metric, "apply_scaler") and metric.apply_scaler)
                            and (hasattr(metric, "apply_pca") and metric.apply_pca)
                        )
                        else new_phe
                    ),
                    **allowed_pr_types[pr_type],
                    restrict_evaluation_query=(
                        f"pert_iname in {jump_moa_pert_inames}"
                        if jump_moa_compounds_only
                        else None
                    ),
                    similarity_metric=metric,
                    parallel=False,
                    return_full_performance_df=True,
                    n_jobs=n_workers,
                )
                pr_df.to_csv(
                    output_dir
                    / f"pr_{metric_name}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
                )
                print(f"PR {metric_name} = {pr}")

    def calc_auroc_metrics(
        self,
        unique_model_name: str,
        data_split_index: int,
        eval_type: Union[
            Literal["auroc"],
            Literal["auroc_moas"],
            Literal["auroc_across_lines"],
            Literal["auroc_across_lines_moas"],
        ],
        calc_gpu_only_prs: bool = True,
        calc_cpu_only_prs: bool = True,
        jump_moa_compounds_only: bool = False,
        metrics_list: Optional[Union[list[str], str]] = None,
        parallel: bool = True,
        cuda_for_gpu_metrics: bool = True,
    ):
        if isinstance(metrics_list, str):
            metrics_list = [
                split_metric.strip()
                for split_metric in metrics_list.strip()
                .replace("'", "")
                .replace('"', "")
                .split(",")
            ]
        allowed_eval_types = {
            "auroc": {"groupby": ["cell_id", "pert_iname", "pert_idose_uM"]},
            "auroc_moas": {"groupby": ["cell_id", "moa"]},
            "auroc_across_lines": {"groupby": ["pert_iname"]},
            "auroc_across_lines_moas": {"groupby": ["moa"]},
        }
        if eval_type not in allowed_eval_types:
            raise ValueError(
                f"The given value for pr_type ({eval_type} was not found in allowed_pr_types ({list(allowed_eval_types.keys())})"
            )
        CELL_LINE_SPLIT = data_split_index // 10
        MOA_SPLIT = data_split_index % 10

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )

        if jump_moa_compounds_only:
            from .jump_moa_compounds import jump_moa_compound_pert_iname_list

            jump_moa_pert_inames = jump_moa_compound_pert_iname_list.copy()
            jump_moa_pert_inames.remove("DMSO")

        checkpoint_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
            / "checkpoints"
        )
        output_dir = (
            self.working_dir
            / Path("plot_data")
            / Path("TripletLoss")
            / Path(f"{unique_model_name}")
            / Path(eval_type)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if calc_gpu_only_prs:
            if not cuda_for_gpu_metrics:
                print("Evaluating GPU metric without CUDA")
            metric = get_phenonaut_metric_from_model_checkpoint(
                "TripletLoss",
                self.uninstantiated_model,
                checkpoint_dir
                / Path(
                    f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
                ),
                higher_is_better=False,
                cuda=cuda_for_gpu_metrics,
            )
            print("Getting TL embeddings...", end="")
            new_phe = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=True,
                pca=True,
            )

            embeddings = metric.func.model.get_embeddings(new_phe.ds.data.values)
            new_features_for_embeddings = [
                f"TL_feat_{i}" for i in range(1, embeddings.shape[1] + 1)
            ]
            new_phe.ds.df = pd.concat(
                [
                    new_phe.ds.df,
                    pd.DataFrame(
                        embeddings,
                        columns=new_features_for_embeddings,
                        index=new_phe.ds.df.index,
                    ),
                ],
                axis=1,
            )
            old_features = new_phe.ds.features.copy()
            new_phe.ds.features = (
                new_features_for_embeddings,
                "Updated features to TripletLoss derived features",
            )
            new_phe.ds.drop_columns(
                old_features,
                "Removing features as they are superseeded by TripletLoss derived embeddings",
            )
            print("...done")

            metric_df = auroc_performance(
                new_phe,
                **allowed_eval_types[eval_type],
                phenotypic_metric=PhenotypicMetric(
                    "TripletLoss", "cosine", (0, 2), higher_is_better=False
                ),
                parallel=False,
                allowed_pert_inames=(
                    jump_moa_pert_inames if jump_moa_compounds_only else None
                ),
            )
            metric_df.to_csv(
                output_dir
                / f"auroc_{metric.name}{'_jumpmoas' if jump_moa_compounds_only else ''}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}.csv"
            )

        if calc_cpu_only_prs:
            from .metrics import leak_proof_cmap_metrics

            if metrics_list is not None:
                print(f"Filtering metrics, attempting to include {metrics_list}")
                leak_proof_cmap_metrics = {
                    k: v
                    for k, v in leak_proof_cmap_metrics.items()
                    if k in metrics_list
                }
                print(f"Filtered metrics are {leak_proof_cmap_metrics.keys()}")
            new_phe = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=False,
                pca=False,
            )

            new_phe_scaled_pca = get_phenonaut_object_containing_tests_from_split_object(
                df=phe.df,
                features=phe.ds.features,
                split_object=self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                ),
                scale=True,
                pca=True,
            )

            for metric_name, metric in leak_proof_cmap_metrics.items():
                print(
                    f"Working on {metric_name} for cls {CELL_LINE_SPLIT} and moas {MOA_SPLIT} for {eval_type}"
                )
                metric_df = auroc_performance(
                    (
                        new_phe_scaled_pca
                        if (
                            (hasattr(metric, "apply_scaler") and metric.apply_scaler)
                            and (hasattr(metric, "apply_pca") and metric.apply_pca)
                        )
                        else new_phe
                    ),
                    **allowed_eval_types[eval_type],
                    phenotypic_metric=metric,
                    parallel=parallel,
                    allowed_pert_inames=(
                        jump_moa_pert_inames if jump_moa_compounds_only else None
                    ),
                )
                metric_df.to_csv(
                    output_dir
                    / f"auroc_{metric_name}{'_jumpmoas' if jump_moa_compounds_only else ''}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}.csv"
                )

    def write_learntmetric_vs_metric_scatter_metrics(
        self,
        unique_model_name: str,
        data_split_index: int,
        eval_type: Union[
            Literal["metricvmetric"],
            Literal["metricvmetric_moas"],
            Literal["metricvmetric_across_lines"],
            Literal["metricvmetric_across_lines_moas"],
        ],
        jump_moa_compounds_only: bool = False,
        comparison_metric_name: str = "Zhang",
        comparison_metric_stdscale: bool = False,
        comparison_metric_pca: bool = False,
        cuda_for_gpu_metrics: bool = True,
    ):
        from .benchmarking import trained_model_vs_metric
        from .metrics import get_leak_proof_cmap_standard_L1000_metrics

        # This serves to hold the assinged groupbys
        allowed_eval_types = {
            "metricvmetric": {"groupby": ["pert_iname", "cell_id", "pert_idose_uM"]},
            "metricvmetric_moas": {"groupby": ["moa", "cell_id"]},
            "metricvmetric_across_lines": {"groupby": ["pert_iname"]},
            "metricvmetric_across_lines_moas": {"groupby": ["moa"]},
        }
        if eval_type not in allowed_eval_types:
            raise ValueError(
                f"The given value for pr_type ({eval_type} was not found in allowed_pr_types ({list(allowed_eval_types.keys())})"
            )
        CELL_LINE_SPLIT = data_split_index // 10
        MOA_SPLIT = data_split_index % 10

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )

        checkpoint_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
            / "checkpoints"
        )
        output_dir = (
            self.working_dir
            / Path("plot_data")
            / Path("TripletLoss")
            / Path(f"{unique_model_name}")
            / Path(eval_type)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if jump_moa_compounds_only:
            from .jump_moa_compounds import jump_moa_compound_pert_iname_list

            phe.ds.df = phe.ds.df.query(
                "pert_iname in @jump_moa_compound_pert_iname_list"
            )
        # Here we will extract the split object and manually add DMSO so that it may
        # be included and compared with test split treatments
        split_object = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path(f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json")
        )
        split_object.test[0]["moas"].append("control vehicle")

        phe_scaled_pca = get_phenonaut_object_containing_tests_from_split_object(
            df=phe.df,
            features=phe.ds.features,
            split_object=split_object,
            scale=True,
            pca=True,
        )
        if not cuda_for_gpu_metrics:
            print("Evaluating GPU metric without CUDA")
        metric = get_phenonaut_metric_from_model_checkpoint(
            "TripletLoss",
            self.uninstantiated_model,
            checkpoint_dir
            / Path(
                f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
            ),
            higher_is_better=False,
            cuda=cuda_for_gpu_metrics,
        )

        embeddings = metric.func.model.get_embeddings(phe_scaled_pca.ds.data.values)
        new_features_for_embeddings = [
            f"TL_feat_{i}" for i in range(1, embeddings.shape[1] + 1)
        ]
        phe_embeddings = Phenonaut(phe_scaled_pca)
        phe_embeddings.ds.df = pd.concat(
            [
                phe_scaled_pca.ds.df,
                pd.DataFrame(
                    embeddings,
                    columns=new_features_for_embeddings,
                    index=phe_scaled_pca.ds.df.index,
                ),
            ],
            axis=1,
        )
        old_features = phe_embeddings.ds.features.copy()
        phe_embeddings.ds.features = (
            new_features_for_embeddings,
            "Updated features to TripletLoss derived features",
        )
        phe_embeddings.ds.drop_columns(
            old_features,
            "Removing features as they are superseeded by TripletLoss derived embeddings",
        )
        metric.func = "cosine"

        phe_comparison_metric = get_phenonaut_object_containing_tests_from_split_object(
            df=phe.df,
            features=phe.ds.features,
            split_object=split_object,
            scale=comparison_metric_stdscale,
            pca=comparison_metric_pca,
        )
        comparison_metric = get_leak_proof_cmap_standard_L1000_metrics().get(
            comparison_metric_name, None
        )
        if comparison_metric is None:
            print(
                f"Could not find metric ({comparison_metric_name}) in available metrics ({get_leak_proof_cmap_standard_L1000_metrics().keys()})"
            )

        # Get DF for TL
        res_df = trained_model_vs_metric(
            output_dir=output_dir,
            phe_embeddings=phe_embeddings,
            phe_comparison_metric=phe_comparison_metric,
            learnt_metric=metric,
            comparison_metric=comparison_metric,
            groupby=allowed_eval_types[eval_type]["groupby"],
            write_images=True,
            random_state=self.random_state,
        )

        # Get DF for Zhang

    def calc_treatment_permutation_score_metrics(
        self,
        unique_model_name: str,
        data_split_index: int,
        eval_type: Union[
            Literal["permutationtest"],
            Literal["permutationtest_moas"],
            Literal["permutationtest_across_lines"],
            Literal["permutationtest_across_lines_moas"],
        ],
        calc_gpu_metrics: bool = True,
        calc_cpu_metrics: bool = True,
        jump_moa_compounds_only: bool = False,
        metrics_list: Optional[Union[list[str], str]] = None,
        cuda_for_gpu_metrics: bool = True,
    ):
        if isinstance(metrics_list, str):
            metrics_list = [
                split_metric.strip()
                for split_metric in metrics_list.strip()
                .replace("'", "")
                .replace('"', "")
                .split(",")
            ]
        # This serves to hold the assinged groupbys
        allowed_eval_types = {
            "permutationtest": {"groupby": ["cell_id", "pert_iname", "pert_idose_uM"]},
            "permutationtest_moas": {"groupby": ["cell_id", "moa"]},
            "permutationtest_across_lines": {"groupby": ["pert_iname"]},
            "permutationtest_across_lines_moas": {"groupby": ["moa"]},
        }
        if eval_type not in allowed_eval_types:
            raise ValueError(
                f"The given value for eval_type ({eval_type} was not found in allowed_eval_types ({list(allowed_eval_types.keys())})"
            )
        CELL_LINE_SPLIT = data_split_index // 10
        MOA_SPLIT = data_split_index % 10

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )

        checkpoint_dir = (
            self.working_dir
            / Path(f"{self.model_name}")
            / Path("trained")
            / Path(f"{unique_model_name}")
            / "checkpoints"
        )
        output_dir = (
            self.working_dir
            / Path("plot_data")
            / Path("TripletLoss")
            / Path(f"{unique_model_name}")
            / Path(eval_type)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if jump_moa_compounds_only:
            from .jump_moa_compounds import jump_moa_compound_pert_iname_list

            phe.ds.df = phe.ds.df.query(
                "pert_iname in @jump_moa_compound_pert_iname_list"
            )
            print(
                f"Filtered for JUMP MOA compounds, now {len(phe.df.pert_iname.unique())}"
            )
            print(
                f"Filtered for JUMP MOA compounds, now {phe.df.pert_iname.value_counts()}"
            )
        # Here we will extract the split object and manually add DMSO so that it may
        # be included and compared with test split treatments
        split_object = CMAPSplit.load(
            self.working_dir
            / Path("split_data")
            / Path(f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json")
        )
        split_object.test[0]["moas"].append("control vehicle")

        phe = get_phenonaut_object_containing_tests_from_split_object(
            df=phe.df,
            features=phe.ds.features,
            split_object=split_object,
            scale=False,
            pca=False,
        )

        phe_scaled_pca = get_phenonaut_object_containing_tests_from_split_object(
            df=phe.df,
            features=phe.ds.features,
            split_object=split_object,
            scale=True,
            pca=True,
        )

        if calc_gpu_metrics:
            if not cuda_for_gpu_metrics:
                print("Evaluating GPU metric without CUDA")
            print("Getting TL embeddings...", end="")

            metric = get_phenonaut_metric_from_model_checkpoint(
                "TripletLoss",
                self.uninstantiated_model,
                checkpoint_dir
                / Path(
                    f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
                ),
                higher_is_better=False,
                cuda=cuda_for_gpu_metrics,
            )

            embeddings = metric.func.model.get_embeddings(phe_scaled_pca.ds.data.values)
            new_features_for_embeddings = [
                f"TL_feat_{i}" for i in range(1, embeddings.shape[1] + 1)
            ]
            phe_embeddings = Phenonaut(phe_scaled_pca)
            phe_embeddings.ds.df = pd.concat(
                [
                    phe_scaled_pca.ds.df,
                    pd.DataFrame(
                        embeddings,
                        columns=new_features_for_embeddings,
                        index=phe_scaled_pca.ds.df.index,
                    ),
                ],
                axis=1,
            )
            old_features = phe_embeddings.ds.features.copy()
            phe_embeddings.ds.features = (
                new_features_for_embeddings,
                "Updated features to TripletLoss derived features",
            )
            phe_embeddings.ds.drop_columns(
                old_features,
                "Removing features as they are superseeded by TripletLoss derived embeddings",
            )

            groupby = allowed_eval_types[eval_type]["groupby"]
            metric.func = "cosine"
            results_df = pertmutation_test_distinct_from_dmso(
                phe_embeddings,
                groupby,
                metric,
                random_state=self.random_state,
            )
            print(f"Writing to {output_dir}")
            results_df.to_csv(
                output_dir
                / f"pt_{metric.name}{'_jumpmoas' if jump_moa_compounds_only else ''}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}.csv"
            )

        if calc_cpu_metrics:
            from .metrics import leak_proof_cmap_metrics

            # Here we will extract the split object and manually add DMSO so that it may
            # be included and compared with test split treatments
            split_object = CMAPSplit.load(
                self.working_dir
                / Path("split_data")
                / Path(
                    f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json"
                )
            )
            split_object.test[0]["moas"].append("control vehicle")

            groupby = allowed_eval_types[eval_type]["groupby"]
            if metrics_list is not None:
                print(f"Filtering metrics, attempting to include {metrics_list}")
                leak_proof_cmap_metrics = {
                    k: v
                    for k, v in leak_proof_cmap_metrics.items()
                    if k in metrics_list
                }
                print(f"Filtered metrics are {leak_proof_cmap_metrics.keys()}")

            for metric_name, metric in leak_proof_cmap_metrics.items():
                print(
                    f"Working on {metric_name} for cls {CELL_LINE_SPLIT} and moas {MOA_SPLIT} for {eval_type}"
                )

                results_df = pd.DataFrame()

                phe_to_use = (
                    phe_scaled_pca
                    if (
                        (hasattr(metric, "apply_scaler") and metric.apply_scaler)
                        and (hasattr(metric, "apply_pca") and metric.apply_pca)
                    )
                    else phe
                )

                results_df = pertmutation_test_distinct_from_dmso(
                    phe_to_use,
                    groupby,
                    metric,
                    random_state=self.random_state,
                )
                print(f"Writing to {output_dir}")
                results_df.to_csv(
                    output_dir
                    / f"pt_{metric.name}{'_jumpmoas' if jump_moa_compounds_only else ''}_cls{CELL_LINE_SPLIT}_ps{MOA_SPLIT}_seed_{self.random_state}.csv"
                )

    def test_split_to_fitted_umap(
        self,
        model_identifier: str,
        data_split_index: int,
        umap_constructor_kwargs: dict = {
            "min_dist": 0.2,
            "metric": "cosine",
            "n_neighbors": 50,
            "output_metric": "haversine",
        },
    ) -> None:
        """Perform UMAP on embeddings of samples within a test split

        Parameters
        ----------
        model_identifier : str
            Model identifier - typically optuna run number
        data_split_index : int
            The two digit split identifier with the first digit denoting the cell line
            split and the second digit denoting the MoA split
        umap_constructor_kwargs : dict, optional
            Additional kw_args which can be passed to umap. The internal random_state of
            leakproofcmap will overwrite any random state added to this argument
            dictionary, by default {"metric": "cosine"}
        """
        import umap
        import pickle
        import lzma

        if isinstance(umap_constructor_kwargs, str):
            umap_constructor_kwargs = json.loads(umap_constructor_kwargs)
        model_identifier = str(model_identifier)
        split_dir = self.working_dir / "split_data"
        checkpoint_dir = (
            self.working_dir
            / Path("TripletLoss/trained/")
            / Path(model_identifier)
            / Path("checkpoints")
        )
        if data_split_index is None:
            CELL_LINE_SPLIT = 1
            MOA_SPLIT = 1
        else:
            print(data_split_index, type(data_split_index))
            CELL_LINE_SPLIT = data_split_index // 10
            MOA_SPLIT = data_split_index % 10

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path("plot_data/TripletLoss/")
            / Path(model_identifier)
            / Path("UMAP")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        split_object = CMAPSplit.load(
            split_dir
            / Path(f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json")
        )
        df, features = split_object.get_df(
            df=phe.ds.df,
            features=phe.ds.features,
            scale=True,
            pca=False,
            tvt_type="test",
            fold_number=1,
        )
        print(
            f"Got {len(df)} samples with {len(features)} features from split {data_split_index}"
        )
        model = self.uninstantiated_model.load_from_checkpoint(
            checkpoint_dir
            / f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
        )
        embeddings = model.get_embeddings(df[features].values)
        print(
            f"{embeddings.shape[0]} embeddings of length {embeddings.shape[1]} generated, running UMAP..."
        )
        umap_constructor_kwargs.update({"random_state": self.random_state})
        reducer = umap.UMAP(**umap_constructor_kwargs)
        umap_embedding = reducer.fit_transform(embeddings)
        out_df = pd.concat(
            [
                df[[c for c in df.columns if c not in features]],
                pd.DataFrame(
                    umap_embedding, index=df.index, columns=["UMAP_1", "UMAP_2"]
                ),
            ],
            axis=1,
        )
        csv_output_path = (
            output_dir
            / f"umap_embeddings_cls{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}_seed_{self.random_state}.csv"
        )
        umap_lzma_pickle_path = (
            output_dir
            / f"umap_reducer_cls{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}_seed_{self.random_state}.pkl.lzma"
        )
        out_df.to_csv(csv_output_path)
        with lzma.open(
            umap_lzma_pickle_path,
            "wb",
        ) as f:
            pickle.dump(
                reducer,
                f,
            )
        print(
            f"UMAP embeddings ({out_df.shape[0]} x {out_df.shape[1]}) generated and written to {csv_output_path}"
        )
        print(f"Fitted UMAP object written to {umap_lzma_pickle_path}")

    def test_split_to_tsne_embedding(
        self,
        model_identifier: str,
        data_split_index: int,
        tsne_constructor_kwargs: dict = {"metric": "cosine", "perplexity": 50},
    ) -> None:
        """Perform UMAP on embeddings of samples within a test split

        Parameters
        ----------
        model_identifier : str
            Model identifier - typically optuna run number
        data_split_index : int
            The two digit split identifier with the first digit denoting the cell line
            split and the second digit denoting the MoA split
        umap_constructor_kwargs : dict, optional
            Additional kw_args which can be passed to umap. The internal random_state of
            leakproofcmap will overwrite any random state added to this argument
            dictionary, by default {"metric": "cosine"}
        """
        from sklearn.manifold import TSNE

        if isinstance(tsne_constructor_kwargs, str):
            tsne_constructor_kwargs = json.loads(tsne_constructor_kwargs)
        model_identifier = str(model_identifier)
        split_dir = self.working_dir / "split_data"
        checkpoint_dir = (
            self.working_dir
            / Path("TripletLoss/trained/")
            / Path(model_identifier)
            / Path("checkpoints")
        )
        if data_split_index is None:
            CELL_LINE_SPLIT = 1
            MOA_SPLIT = 1
        else:
            print(data_split_index, type(data_split_index))
            CELL_LINE_SPLIT = data_split_index // 10
            MOA_SPLIT = data_split_index % 10

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path("plot_data/TripletLoss/")
            / Path(model_identifier)
            / Path("tSNE")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        split_object = CMAPSplit.load(
            split_dir
            / Path(f"cmap_split_cellidsplit{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}.json")
        )
        df, features = split_object.get_df(
            df=phe.ds.df,
            features=phe.ds.features,
            scale=True,
            pca=False,
            tvt_type="test",
            fold_number=1,
        )
        print(
            f"Got {len(df)} samples with {len(features)} features from split {data_split_index}"
        )
        model = self.uninstantiated_model.load_from_checkpoint(
            checkpoint_dir
            / f"cmap_split_cls_{CELL_LINE_SPLIT}_moas_{MOA_SPLIT}_seed_{self.random_state}.ckpt"
        )
        embeddings = model.get_embeddings(df[features].values)
        print(
            f"{embeddings.shape[0]} embeddings of length {embeddings.shape[1]} generated, running tSNE..."
        )
        tsne_constructor_kwargs.update({"random_state": self.random_state})
        reducer = TSNE(**tsne_constructor_kwargs)
        umap_embedding = reducer.fit_transform(embeddings)
        out_df = pd.concat(
            [
                df[[c for c in df.columns if c not in features]],
                pd.DataFrame(
                    umap_embedding, index=df.index, columns=["tSNE_1", "tSNE_2"]
                ),
            ],
            axis=1,
        )
        csv_output_path = (
            output_dir
            / f"tSNE_embeddings_cls{CELL_LINE_SPLIT}_moasplit{MOA_SPLIT}_seed_{self.random_state}.csv"
        )
        out_df.to_csv(csv_output_path)
        print(
            f"tSNE embeddings ({out_df.shape[0]} x {out_df.shape[1]}) generated and written to {csv_output_path}"
        )

    def full_model_to_fitted_umap(
        self,
        model_identifier: str,
        umap_constructor_kwargs: dict = {"min_dist": 0.2, "metric": "cosine"},
    ) -> None:
        """Perform UMAP on embeddings of samples within a test split

        Parameters
        ----------
        model_identifier : str
            Model identifier - typically optuna run number
        data_split_index : int
            The two digit split identifier with the first digit denoting the cell line
            split and the second digit denoting the MoA split
        umap_constructor_kwargs : dict, optional
            Additional kw_args which can be passed to umap. The internal random_state of
            leakproofcmap will overwrite any random state added to this argument
            dictionary, by default {"metric": "cosine"}
        """
        import umap
        import pickle
        import lzma

        if isinstance(umap_constructor_kwargs, str):
            umap_constructor_kwargs = json.loads(umap_constructor_kwargs)
        model_identifier = str(model_identifier)
        split_dir = self.working_dir / "split_data"
        checkpoint_dir = (
            self.working_dir
            / Path("TripletLoss/trained/")
            / Path(model_identifier)
            / Path("checkpoints")
        )

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path("plot_data/TripletLoss/")
            / Path(model_identifier)
            / Path("UMAP")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        split_object = CMAPSplit.load(
            split_dir / Path("cmap_split_cellidsplit1_moasplit1.json")
        )
        split_object.merge_all_data_into_train()

        df, features = split_object.get_df(
            df=phe.ds.df,
            features=phe.ds.features,
            scale=True,
            pca=False,
            tvt_type="train",
            fold_number=1,
        )
        print(f"Got {len(df)} samples with {len(features)} features")
        model = self.uninstantiated_model.load_from_checkpoint(
            checkpoint_dir / f"cmap_all_data_seed_{self.random_state}.ckpt"
        )
        embeddings = model.get_embeddings(df[features].values)
        print(
            f"{embeddings.shape[0]} embeddings of length {embeddings.shape[1]} generated, running UMAP..."
        )
        umap_constructor_kwargs.update({"random_state": self.random_state})
        reducer = umap.UMAP(**umap_constructor_kwargs)
        umap_embedding = reducer.fit_transform(embeddings)
        out_df = pd.concat(
            [
                df[[c for c in df.columns if c not in features]],
                pd.DataFrame(
                    umap_embedding, index=df.index, columns=["UMAP_1", "UMAP_2"]
                ),
            ],
            axis=1,
        )
        csv_output_path = (
            output_dir / f"umap_embeddings_all_data_seed_{self.random_state}.csv"
        )
        umap_lzma_pickle_path = (
            output_dir / f"umap_reducer_all_data_seed_{self.random_state}.pkl.lzma"
        )
        out_df.to_csv(csv_output_path)
        with lzma.open(
            umap_lzma_pickle_path,
            "wb",
        ) as f:
            pickle.dump(
                reducer,
                f,
            )
        print(
            f"UMAP embeddings ({out_df.shape[0]} x {out_df.shape[1]}) generated and written to {csv_output_path}"
        )
        print(f"Fitted UMAP object written to {umap_lzma_pickle_path}")

    def full_model_to_tsne_embedding(
        self,
        model_identifier: str,
        tsne_constructor_kwargs: dict = {"metric": "cosine"},
    ) -> None:
        """Perform tSNE on embeddings of samples within a test split

        Parameters
        ----------
        model_identifier : str
            Model identifier - typically optuna run number
        data_split_index : int
            The two digit split identifier with the first digit denoting the cell line
            split and the second digit denoting the MoA split
        tsne_constructor_kwargs : dict, optional
            Additional kw_args which can be passed to tSNE. The internal random_state of
            leakproofcmap will overwrite any random state added to this argument
            dictionary, by default {"metric": "cosine"}
        """
        from sklearn.manifold import TSNE

        if isinstance(tsne_constructor_kwargs, str):
            tsne_constructor_kwargs = json.loads(tsne_constructor_kwargs)
        model_identifier = str(model_identifier)
        split_dir = self.working_dir / "split_data"
        checkpoint_dir = (
            self.working_dir
            / Path("TripletLoss/trained/")
            / Path(model_identifier)
            / Path("checkpoints")
        )

        self._check_class()

        # Load all CMAP data into a phenonaut object
        phe = get_cmap_phenonaut_object(
            self.phenonaut_packaged_dataset_dir,
            working_dir=self.working_dir,
            pickle_dir=self.pickles_dir,
        )
        output_dir = (
            self.working_dir
            / Path("plot_data/TripletLoss/")
            / Path(model_identifier)
            / Path("tSNE")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        split_object = CMAPSplit.load(
            split_dir / Path("cmap_split_cellidsplit1_moasplit1.json")
        )
        split_object.merge_all_data_into_train()

        df, features = split_object.get_df(
            df=phe.ds.df,
            features=phe.ds.features,
            scale=True,
            pca=False,
            tvt_type="train",
            fold_number=1,
        )
        print(f"Got {len(df)} samples with {len(features)} features")
        model = self.uninstantiated_model.load_from_checkpoint(
            checkpoint_dir / f"cmap_all_data_seed_{self.random_state}.ckpt"
        )
        embeddings = model.get_embeddings(df[features].values)
        print(
            f"{embeddings.shape[0]} embeddings of length {embeddings.shape[1]} generated, running tSNE..."
        )
        tsne_constructor_kwargs.update({"random_state": self.random_state})
        reducer = TSNE(**tsne_constructor_kwargs)
        tsne_embedding = reducer.fit_transform(embeddings)
        out_df = pd.concat(
            [
                df[[c for c in df.columns if c not in features]],
                pd.DataFrame(
                    tsne_embedding, index=df.index, columns=["tSNE_1", "tSNE_2"]
                ),
            ],
            axis=1,
        )
        csv_output_path = (
            output_dir / f"tsne_embeddings_all_data_seed_{self.random_state}.csv"
        )

        out_df.to_csv(csv_output_path)
        print(
            f"tSNE embeddings ({out_df.shape[0]} x {out_df.shape[1]}) generated and written to {csv_output_path}"
        )
