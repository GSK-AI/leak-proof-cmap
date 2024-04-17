# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

import json
import pickle
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class CMAPSplit:
    @classmethod
    def load(cls, json_path: Union[Path, str]):
        """Create a split object by loading a split json file

        Parameters
        ----------
        json_path : Union[Path, str]
            Path to split json file

        Returns
        -------
        CMAPSplit
            Split object

        Raises
        ------
        FileNotFoundError
            Error raised if given split information file is not found
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError("Couldnt find specified file", json_path)
        new_split = cls()
        new_split.__dict__ = json.load(open(json_path))
        if (json_path.parent / "stdscaler_and_pca.pkl").exists():
            print(
                f"Loading fitted standard scaler and PCA found in split json dir ({json_path.parent/'stdscaler_and_pca.pkl'})"
            )
            new_split.scaler, new_split.pca = pickle.load(
                open(json_path.parent / "stdscaler_and_pca.pkl", "rb")
            )
        return new_split

    def save(self, json_path: Union[str, Path]):
        """Save split to json file

        Parameters
        ----------
        json_path : Union[str, Path]
            Target path for the json file containing split information
        """
        json_path = Path(json_path)
        if not json_path.parent.exists():
            json_path.parent.mkdir(parents=True)
        if hasattr(self, "scaler"):
            temp_scaler = self.scaler
            temp_pca = self.pca
            del self.scaler, self.pca
            json.dump(self.__dict__, open(json_path, "w"), sort_keys=True, indent=4)
            self.scaler = temp_scaler
            self.pca = temp_pca
        else:
            json.dump(self.__dict__, open(json_path, "w"), sort_keys=True, indent=4)

    @classmethod
    def from_splits_tsv(
        cls,
        cell_id_split_file: Path,
        moa_split_file: Path,
        cell_line_split_number: int,
        moa_split_number: int,
        df: pd.DataFrame,
        features: list[str],
        name: str = "Unnamed CMAP split",
        np_rng_seed=7,
        is_scalar_and_pca_target: bool = False,
    ):
        """Generate a split from a TSV file

        Parameters
        ----------
        cell_id_split_file : Path
            Cell line split information TSV file path
        moa_split_file : Path
            MOA split information TSV file path
        cell_line_split_number : int
            Cell line split number
        moa_split_number : int
            MOA split number
        df : pd.DataFrame
            DataFrame containing all CMap data
        features : list[str]
            Features columns of CMap data
        name : str, optional
            Split name, by default "Unnamed CMAP split"
        np_rng_seed : int, optional
            RNG seed may be specified for reproducibility, by default 7
        is_scalar_and_pca_target : bool, optional
            If True, then standard scaler and PCA objects are fitted to first fold data,
            by default False

        Returns
        -------
        CMAPSplit
            CMAPSplit object
        """
        if is_scalar_and_pca_target:
            scaler_and_pca_output_dir = Path(cell_id_split_file.parent)
        else:
            scaler_and_pca_output_dir = None

        np_random_rng = np.random.default_rng(np_rng_seed)

        csplit = cls(name)

        # Nans will be present and have to be removed, simply because the input tables are not square, and filled with Nans.
        cell_id_splits_df = pd.read_csv(
            Path(cell_id_split_file), sep="\t", index_col=[0]
        )

        test_cell_ids = cell_id_splits_df.loc[f"split_{cell_line_split_number}"].values
        test_cell_ids = test_cell_ids[~pd.isnull(test_cell_ids)]
        test_cell_ids = test_cell_ids.tolist()
        np_random_rng.shuffle(test_cell_ids)

        trainval_cell_ids = np.array(
            list(set(cell_id_splits_df.values.flatten().tolist()) - set(test_cell_ids))
        )
        trainval_cell_ids = trainval_cell_ids[~pd.isnull(trainval_cell_ids)]
        trainval_cell_ids = trainval_cell_ids.tolist()
        np_random_rng.shuffle(trainval_cell_ids)

        moa_splits_df = pd.read_csv(Path(moa_split_file), sep="\t", index_col=[0])

        test_moas = moa_splits_df.loc[f"split_{moa_split_number}"].values
        test_moas = test_moas[~pd.isnull(test_moas)]
        test_moas = test_moas.tolist()
        np_random_rng.shuffle(test_moas)

        trainval_moas = np.array(
            list(set(moa_splits_df.values.flatten().tolist()) - set(test_moas))
        )
        trainval_moas = trainval_moas[~pd.isnull(trainval_moas)]
        trainval_moas = trainval_moas.tolist()
        np_random_rng.shuffle(trainval_moas)

        csplit.add_train_val_splits_and_a_test(
            trainval_cell_ids,
            test_cell_ids,
            trainval_moas,
            test_moas,
            df,
            features,
            n_splits=5,
            kfold_seed=np_rng_seed,
            scaler_and_pca_output_dir=scaler_and_pca_output_dir,
        )

        return csplit

    def __init__(self, name: str = "Unnamed CMAP split"):
        """CMAPSplit objects capturing split information

        Parameters
        ----------
        name : str, optional
            Name for the Split object, by default "Unnamed CMAP split"
        """
        self.name = name
        self.train = []
        self.val = []
        self.test = []
        self.allowed_tvt_types = ["train", "val", "test"]

    def add_train_val_splits_and_a_test(
        self,
        cell_lines_train_val: Union[list[str], str],
        cell_lines_test: Union[list[str], str],
        pert_moas_train_val: Union[list[str], str],
        pert_moas_test: Union[list[str], str],
        df: pd.DataFrame,
        features: list[str],
        n_splits: int = 5,
        kfold_seed: int = 42,
        scaler_and_pca_output_dir: Optional[Path] = None,
    ):
        """Add data to object train val and test

        Parameters
        ----------
        cell_lines_train_val : Union[list[str], str]
            List of cell lines within training and validation sets
        cell_lines_test : Union[list[str], str]
            List of cell lines within test set
        pert_moas_train_val : Union[list[str], str]
            List of MOAs in training and validation sets
        pert_moas_test : Union[list[str], str]
            List of MOAs in test set
        df : pd.DataFrame
            DataFrame containing all CMap information
        features : list[str]
            Feature columns for CMap information
        n_splits : int, optional
            Number of splits, by default 5
        kfold_seed : int, optional
            Seed for use in K-fold splitting, by default 42
        scaler_and_pca_output_dir : Optional[Path], optional
            Output directory for fitted standard scaler and PCA objects. If None, then
            these objects are not written out, by default None
        """
        assert len(cell_lines_train_val) == len(
            set(cell_lines_train_val)
        ), f"cell_lines_train_val has repeats {cell_lines_train_val, set(cell_lines_train_val)}"
        assert len(cell_lines_test) == len(
            set(cell_lines_test)
        ), f"cell_lines_test has repeats {cell_lines_test}"
        assert len(pert_moas_train_val) == len(
            set(pert_moas_train_val)
        ), f"pert_inames_train_val has repeats {pert_moas_train_val}"
        assert len(pert_moas_test) == len(
            set(pert_moas_test)
        ), f"pert_inames_test has repeats {pert_moas_test}"

        self.train = [{} for _ in range(n_splits)]
        self.val = [{} for _ in range(n_splits)]
        self.test = [{"cell_lines": cell_lines_test, "moas": pert_moas_test}]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)
        for split_i, (train_indexes, val_indexes) in enumerate(
            kf.split(cell_lines_train_val)
        ):
            self.train[split_i]["cell_lines"] = np.array(cell_lines_train_val)[
                train_indexes
            ].tolist()
            self.val[split_i]["cell_lines"] = np.array(cell_lines_train_val)[
                val_indexes
            ].tolist()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)
        for split_i, (train_indexes, val_indexes) in enumerate(
            kf.split(pert_moas_train_val)
        ):
            self.train[split_i]["moas"] = np.array(pert_moas_train_val)[
                train_indexes
            ].tolist() + ["control vehicle"]
            self.val[split_i]["moas"] = np.array(pert_moas_train_val)[
                val_indexes
            ].tolist()

        if scaler_and_pca_output_dir is not None:
            retrieved_df, retrieved_features = self.get_df(
                df=df,
                features=features,
                scale=False,
                pca=False,
                tvt_type="train",
                fold_number=1,
            )
            self.scaler = StandardScaler()
            print("Fitting master scaler and PCA for fold 1")
            sc_transformed_data = self.scaler.fit_transform(
                retrieved_df[retrieved_features]
            )
            self.pca = PCA(n_components=0.995, whiten=True).fit(sc_transformed_data)
            pickle.dump(
                [self.scaler, self.pca],
                open(Path(scaler_and_pca_output_dir) / "stdscaler_and_pca.pkl", "wb"),
            )
            print("Done fitting")

    def merge_val_into_train(self, df: pd.DataFrame, features: list[str]):
        """Merge validation into training sets

        Parameters
        ----------
        df : pd.DataFrame
            CMap data
        features : list[str]
            CMap data feature columns names

        Raises
        ------
        RuntimeError
            Raised if the number of folds in training are not the same as in validation
        """
        if len(self.val) != len(self.train):
            raise RuntimeError(
                f"Length of train folds is not the same as the lengthof val folds ({len(self.train)}!={len(self.val)})"
            )
        self.train[0]["cell_lines"] = (
            self.train[0]["cell_lines"] + self.val[0]["cell_lines"]
        )
        self.train[0]["moas"] = self.train[0]["moas"] + self.val[0]["moas"]
        self.train = self.train[:1]
        self.val = []

    def merge_all_data_into_train(self):
        """Merge data from training, validation and test into train

        Raises
        ------
        RuntimeError
            Raised if the number of folds in training are not the same as in validation
        """
        if len(self.val) != len(self.train):
            raise RuntimeError(
                f"Length of train folds is not the same as the lengthof val folds ({len(self.train)}!={len(self.val)})"
            )
        self.train[0]["cell_lines"] = (
            self.train[0]["cell_lines"]
            + self.val[0]["cell_lines"]
            + self.test[0]["cell_lines"]
        )
        self.train[0]["moas"] = (
            self.train[0]["moas"] + self.val[0]["moas"] + self.test[0]["moas"]
        )
        self.train = self.train[:1]
        self.val = []
        self.test = []

    def get_df(
        self,
        df: pd.DataFrame,
        features: list[str],
        scale: bool,
        pca: bool,
        tvt_type: str = "train",
        fold_number: int = 1,
        silent=True,
    ) -> pd.DataFrame:
        """Get DataFrame for train/val/test split

        Parameters
        ----------
        df : pd.Data
            Large dataframe containing all CMAP data
        features : list[str]
            List of features within the dataframe
        scale : bool
            If True, then the returned data is transformed using the standard scaler
            fitted to the training set of split 1,1, fold 1.
        pca : bool
            If True, then the returned data is transformed using PCA fitted to the
            training set of split 1,1, fold 1.with whitening.
        tvt_type : str, optional
            Train/val/test string, indicating which split is to be returned.
            By default "train".
        fold_number : int, optional
            By default, 5 folds are made with indexes/numbers 1,2,3,4,5 (not zero
            indexed). With a value of 1 here, fold 1 is returned. By default 1.
        Returns
        -------
        Tuple(pd.DataFrame, list[str,...])
            Tuple, with the first element being a Pandas DataFrame containing the
            requsted split, and the second being the features of that split as the
            features will change if a PCA transform is requested.
        """
        if fold_number < 1:
            raise ValueError(
                f"split_number should be between 1 and num_splits ({self.n_k_fold_splits})"
            )
        d = getattr(self, tvt_type)

        if d is None:
            raise ValueError(
                f"CMAP_Split.add_pert called with {tvt_type}, valid train/val/test arguments to this function are: {self.allowed_tvt_types}"
            )

        cell_ids = d[fold_number - 1]["cell_lines"]
        moas = d[fold_number - 1]["moas"]

        if not scale:
            if not silent:
                print(f"Getting '{tvt_type}' set")
            if not pca:
                return df.query("cell_id in @cell_ids and moa in @moas"), features
            else:
                raise NotImplementedError(
                    "pca was True, but scale was False, handling of this is not yet implemented.  Please rerun with pca=True and scale=True to perform PCA"
                )
        else:
            qdf = df.query("cell_id in @cell_ids and moa in @moas")
            if not silent:
                print(f"Getting scaled '{tvt_type}' set")
            transformed_data = self.scaler.transform(qdf[features])
            if pca:
                transformed_data = self.pca.transform(transformed_data)
                pca_features_list = [
                    f"PC_{n}" for n in range(1, transformed_data.shape[1] + 1)
                ]

            return (
                pd.concat(
                    [
                        qdf.drop(features, axis=1),
                        pd.DataFrame(
                            transformed_data,
                            index=qdf.index,
                            columns=features if not pca else pca_features_list,
                            dtype=np.float32,
                        ),
                    ],
                    axis=1,
                ),
                pca_features_list if pca else features,
            )

    def set(
        self,
        use_train_splits: Union[None, list[int]] = [1],
        use_val_splits: Union[None, list[int]] = [1],
        use_test_splits: Union[None, list[int]] = [1],
    ):
        """Set the state of a split object to return certain folds

        Parameters
        ----------
        use_train_splits : Union[None, list[int]], optional
            List of fold numbers to use when supplying training data, by default [1]
        use_val_splits : Union[None, list[int]], optional
            List of fold numbers to use when supplying validation data, by default [1]
        use_test_splits : Union[None, list[int]], optional
            List of test fold numbers to use when supplying test data. Test should only
            really ever have 1 'fold', but included here for completeness/possible
            future expansion, by default [1]
        """
        self.use_train_splits = use_train_splits
        self.use_val_splits = use_val_splits
        self.use_test_splits = use_test_splits
