# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import fire
from typing import Optional, Union
from pathlib import Path
import os


class Pipeline:
    def __init__(
        self,
        random_state: Union[int, None],
        working_dir: str = "working_dir",
        phenonaut_packaged_dataset_dir: Optional[str] = None,
    ):
        """The leakproofcmap entry point control pipeline

        Parameters
        ----------
        random_state : Union[int, None]
            Random state seed, used to ensure reproducibility
        working_dir : str, optional
            Working directory into which intermediate and analysis files should be
            written, by default "working_dir"
        phenonaut_packaged_dataset_dir : Optional[str], optional
            Phenonaut packaged dataset dir, required if extracting/generating CMap
            information for the first time and no usable intermediate pickles are found.
            If None, then the environmental variable PHENONAUT_PACKAGED_DATASET_DIR is
            checked.  If found and set, then the directory pointed to by the variable is
            used, by default None

        Raises
        ------
        ValueError
            phenonaut_packaged_dataset_dir was not set (None), and the environmental
            variable PHENONAUT_PACKAGED_DATASET_DIR was not set. Set at least one
        """
        self.random_state = random_state
        self.phenonaut_packaged_dataset_dir = phenonaut_packaged_dataset_dir
        if self.phenonaut_packaged_dataset_dir is None:
            self.phenonaut_packaged_dataset_dir = os.environ.get(
                "PHENONAUT_PACKAGED_DATASET_DIR", None
            )
            if self.phenonaut_packaged_dataset_dir is None:
                raise ValueError(
                    "phenonaut_packaged_dataset_dir was not set (None), and the environmental variable PHENONAUT_PACKAGED_DATASET_DIR was not set. Set at least one"
                )
        self.phenonaut_packaged_dataset_dir = Path(self.phenonaut_packaged_dataset_dir)

        self.working_dir = Path(working_dir).resolve()
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

    def tripletloss(self):
        from .models.tripletloss import (
            CMAPTripletLossLightningModule,
            CMAPTripletDataModule,
        )
        from .model_control import LeakProofCMAPLightningModelControler

        model_controler = LeakProofCMAPLightningModelControler(
            working_dir=self.working_dir,
            model_name="TripletLoss",
            random_state=self.random_state,
            uninstantiated_model=CMAPTripletLossLightningModule,
            uninstantiated_data_module=CMAPTripletDataModule,
            phenonaut_packaged_dataset_dir=self.phenonaut_packaged_dataset_dir,
        )
        return model_controler


def run():
    """Main entrypoint for the leakproofcmap package"""
    fire.Fire(Pipeline)


if __name__ == "__main__":
    fire.Fire(Pipeline())
