# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from random import choice
import lightning as pl
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from ...cmap_split import CMAPSplit


class _BaseCMAPPytorchDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame, features: list[str]):
        """Base class for CMap Pytorch datasets

        Parameters
        ----------
        df : pd.DataFrame
            All CMap data
        features : list[str]
            Feature column names for CMap data
        """
        self.pertiname_list = df.pert_iname.values.tolist()
        # Sets are not guaranteed deterministic, so cast to list and sort for
        # deterministic behaviour
        self.unique_pert_inames = sorted(list(set(self.pertiname_list)))
        self.pert_inames_to_indexes = {
            pert_iname: (df.pert_iname.values == pert_iname).nonzero()[0]
            for pert_iname in self.unique_pert_inames
        }
        self.data = torch.tensor(df[features].values)


class CMAPPytorchDatasetTripletAcrossLines(_BaseCMAPPytorchDataset):
    def __init__(self, df: pd.DataFrame, features: list[str]):
        """Pytorch dataset supplying triplets matches across cell lines

        Parameters
        ----------
        df : pd.DataFrame
            All CMap data
        features : list[str]
            Feature column names for CMap data
        """
        super().__init__(df, features)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pert_iname = self.pertiname_list[idx]

        non_matching_pert_iname = choice(
            [
                nm_pert_iname
                for nm_pert_iname in self.unique_pert_inames
                if nm_pert_iname != pert_iname
            ]
        )

        return (
            self.data[idx],
            self.data[
                choice(
                    [
                        pid
                        for pid in self.pert_inames_to_indexes[self.pertiname_list[idx]]
                        if pid != idx
                    ]
                )
            ],
            self.data[
                choice(
                    [
                        pid
                        for pid in self.pert_inames_to_indexes[non_matching_pert_iname]
                    ]
                )
            ],
        )


class CMAPPytorchDatasetTripletWithinLines(_BaseCMAPPytorchDataset):
    def __init__(self, df: pd.DataFrame, features: list[str]):
        """Pytorch dataset supplying triplets matches within cell lines

        Parameters
        ----------
        df : pd.DataFrame
            All CMap data
        features : list[str]
            Feature column names for CMap data
        """
        super().__init__(df, features)
        self.np_rng = np.random.default_rng()
        df = df[df.columns.difference(features)].copy().reset_index()
        self.cell_id_list = df.cell_id.values.tolist()
        self.cell_line_pert_iname_to_index_dict = {
            cl: {} for cl in df.cell_id.unique().tolist()
        }
        for row_idx, row in tqdm(
            df.iterrows(), desc="Building cell_line perturbation index lookup"
        ):
            cell_id = row["cell_id"]
            pert_iname = row["pert_iname"]

            if pert_iname not in self.cell_line_pert_iname_to_index_dict[cell_id]:
                self.cell_line_pert_iname_to_index_dict[cell_id][pert_iname] = []
            self.cell_line_pert_iname_to_index_dict[cell_id][pert_iname].append(row_idx)

        print(f"Using data within lines, {len(df)=}")
        for cl in self.cell_line_pert_iname_to_index_dict:
            for ptb in self.cell_line_pert_iname_to_index_dict[cl]:
                if len(self.cell_line_pert_iname_to_index_dict[cl][ptb]) == 1:
                    print(f"{cl=}, {ptb=} = 1")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        anchor_pert_iname = self.pertiname_list[idx]

        anchor_cell_line = self.cell_id_list[idx]
        positive_idx_choices = [
            pos_index
            for pos_index in self.cell_line_pert_iname_to_index_dict[anchor_cell_line][
                anchor_pert_iname
            ]
            if pos_index != idx
        ]
        if len(positive_idx_choices) == 0:
            positive_idx = idx
        else:
            positive_idx = self.np_rng.choice(positive_idx_choices)
        negative_pert_iname = choice(
            [
                neg_pert_iname
                for neg_pert_iname in self.cell_line_pert_iname_to_index_dict[
                    anchor_cell_line
                ]
                if neg_pert_iname != anchor_pert_iname
            ]
        )
        negative_idx = choice(
            [
                neg_idx
                for neg_idx in self.cell_line_pert_iname_to_index_dict[
                    anchor_cell_line
                ][negative_pert_iname]
            ]
        )
        return (
            # Anchor
            self.data[idx],
            # Positive
            self.data[positive_idx],
            # Negative
            self.data[negative_idx],
        )


class CMAPPytorchDatasetTripletWithinLinesSameConcentrations(_BaseCMAPPytorchDataset):
    def __init__(self, df: pd.DataFrame, features: list[str]):
        """Pytorch dataset supplying triplets matches with matching concentrations

        Parameters
        ----------
        df : pd.DataFrame
            All CMap data
        features : list[str]
            Feature column names for CMap data
        """
        super().__init__(df, features)
        self.np_rng = np.random.default_rng()
        self.df = df[df.columns.difference(features)].copy().reset_index()
        print(f"Using data within lines, {len(df)=}, matching dose")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        anchor = self.df.iloc[idx]
        pert_id = anchor["pert_id"]
        cell_id = anchor["cell_id"]
        pert_idose_uM = anchor["pert_idose_uM"]
        positive_idx_choices = self.df.query(
            "cell_id == @cell_id and pert_idose_uM==@pert_idose_uM and pert_id == @pert_id"
        )
        if len(positive_idx_choices) == 0:
            positive_idx = idx
        else:
            positive_idx = positive_idx_choices.sample(
                1, random_state=self.np_rng
            ).index.tolist()[0]
        negative_idx_choices = self.df.query(
            "cell_id == @cell_id and pert_idose_uM==@pert_idose_uM and pert_id != @pert_id"
        )
        if len(negative_idx_choices) == 0:
            negative_idx_choices = self.df.query(
                "cell_id == @cell_id and pert_id != @pert_id"
            )
        negative_idx = negative_idx_choices.sample(
            1, random_state=self.np_rng
        ).index.tolist()[0]
        return (
            # Anchor
            self.data[idx],
            # Positive
            self.data[positive_idx],
            # Negative
            self.data[negative_idx],
        )


class CMAPTripletDataModule(pl.LightningDataModule):
    def __init__(
        self,
        big_df: pd.DataFrame,
        features: list[str],
        split_data: CMAPSplit,
        split_number: int = 1,
        batch_size: int = 1024,
        num_workers: int = 16,
        compose_triplets_across_cell_lines: bool = True,
        compose_triplets_within_cell_lines_and_match_dose: bool = False,
    ):
        """CMap pytorch datamodule

        Parameters
        ----------
        big_df : pd.DataFrame
            All CMap data
        features : list[str]
            CMap feature column names
        split_data : CMAPSplit
            CMap split object directing the data module which data to return
        split_number : int, optional
            Which fold number should be used within the split object, by default 1
        batch_size : int, optional
            Batch size, by default 1024
        num_workers : int, optional
            Number of CPU worker threads to spawn, by default 16
        compose_triplets_across_cell_lines : bool, optional
            If True, then triplet anchors and positives can be from different cell
            lines, by default True
        compose_triplets_within_cell_lines_and_match_dose : bool, optional
            If True, the triplet anchor and positives must match dose, by default False
        """
        super().__init__()
        self.big_df = big_df
        self.features = features
        self.split_data = split_data
        self.split_number = split_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        if compose_triplets_across_cell_lines:
            self.CMDSTClass = CMAPPytorchDatasetTripletAcrossLines
        else:
            self.CMDSTClass = CMAPPytorchDatasetTripletWithinLines

        if compose_triplets_within_cell_lines_and_match_dose:
            print("Matching dose and concentration")
            self.CMDSTClass = CMAPPytorchDatasetTripletWithinLinesSameConcentrations

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.CMDSTClass(
                *self.split_data.get_df(
                    df=self.big_df,
                    features=self.features,
                    scale=True,
                    pca=True,
                    tvt_type="train",
                    fold_number=self.split_number,
                )
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.split_data.val is None or self.split_data.val == []:
            return None
        return torch.utils.data.DataLoader(
            CMAPPytorchDatasetTripletAcrossLines(
                *self.split_data.get_df(
                    df=self.big_df,
                    features=self.features,
                    scale=True,
                    pca=True,
                    tvt_type="val",
                    fold_number=self.split_number,
                )
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, batch_size=None):
        return torch.utils.data.DataLoader(
            CMAPPytorchDatasetTripletAcrossLines(
                *self.split_data.get_df(
                    df=self.big_df,
                    features=self.features,
                    scale=True,
                    pca=True,
                    tvt_type="test",
                    fold_number=self.split_number,
                )
            ),
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
