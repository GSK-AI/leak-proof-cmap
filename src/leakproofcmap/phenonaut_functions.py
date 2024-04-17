# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path
from typing import Optional, Union
from io import StringIO
import importlib.resources
import json
import pandas as pd

import phenonaut

from .cmap_split import CMAPSplit


def get_cmap_phenonaut_object(
    phenonaut_packaged_dataset_dir: Union[str, Path],
    working_dir: Union[str, Path],
    pickle_dir: Union[str, Path],
    phe_cmap_l4_path="phe_ob_CMAP_L4.pkl",
    phe_cmap_l4_clue_moas_downsampled="phe_ob_CMAP_CLUE_MOAs_downsampled.pkl",
) -> phenonaut.Phenonaut:
    """Get a LPCMap filtered Phenonaut object

    Get a Phenonaut object containing the LPCMap filtered CMap samples with MOA
    annotations. Uses the CMap level 4 loader from Phenonaut to download CMap and
    extract LPCMap filtered profiles. Intermediate pickles are stored so as not to
    require re-download/extractio of CMap upon every call.

    Parameters
    ----------
    phenonaut_packaged_dataset_dir : Union[str, Path]
        A directory with lots of space within which Phenonaut may download/extract
        CMap and store intermediate pickles
    working_dir : Union[str, Path]
        Working directory for leakproofcmap. This is usually a 'working_dir' folder in
        the current directory
    pickle_dir : Union[str, Path]
        A directory into which filtered CMap may be written as a pickle file. Usually
        'pickles'. If the path given is relative, then it is placed under the directory
        defined in the 'working_dir' argument. If absolute, then this absolute path is
        used
    phe_cmap_l4_path : str, optional
        Pickle filename for the Phenonaut downloaded CMap level 4 data, by default
        "phe_ob_CMAP_L4.pkl"
    phe_cmap_l4_clue_moas_downsampled : str, optional
        Pickle filename for downsampled CMap level 4 data used by LPCMap, by default
        "phe_ob_CMAP_CLUE_MOAs_downsampled.pkl"

    Returns
    -------
    phenonaut.Phenonaut
        Phenonaut object containing the LPCMap L4 filtered CMap data
    """

    phenonaut_packaged_dataset_dir = Path(phenonaut_packaged_dataset_dir)

    working_dir = Path(working_dir)
    if not working_dir.is_absolute():
        working_dir = working_dir.resolve()

    pickle_dir = Path(pickle_dir)
    if not pickle_dir.is_absolute():
        pickle_dir = working_dir / pickle_dir
    if not pickle_dir.exists():
        pickle_dir.mkdir(parents=True)

    phe_cmap_l4_path = Path(phe_cmap_l4_path)
    if not phe_cmap_l4_path.is_absolute():
        phe_cmap_l4_path = pickle_dir / phe_cmap_l4_path

    phe_cmap_l4_clue_moas_downsampled = Path(phe_cmap_l4_clue_moas_downsampled)
    if not phe_cmap_l4_clue_moas_downsampled.is_absolute():
        phe_cmap_l4_clue_moas_downsampled = (
            pickle_dir / phe_cmap_l4_clue_moas_downsampled
        )

    if not phe_cmap_l4_clue_moas_downsampled.exists():
        print(f"Could not find {phe_cmap_l4_clue_moas_downsampled}, regenerating")
        if not phe_cmap_l4_path.exists():
            phe = phenonaut.Phenonaut(
                dataset=phenonaut.packaged_datasets.CMAP_Level4(
                    root=phenonaut_packaged_dataset_dir / "cmap/", download=True
                )
            )
            phe.save(phe_cmap_l4_path)
        else:
            phe = phenonaut.Phenonaut.load(phe_cmap_l4_path)

        clue_moa_file = (
            phenonaut_packaged_dataset_dir / "repurposing_drugs_20200324.txt"
        )
        if not clue_moa_file.exists():
            import urllib.request

            urllib.request.urlretrieve(
                "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt",
                clue_moa_file,
            )
        moa_df = pd.read_csv(clue_moa_file, sep="\t", skiprows=9).dropna(subset=["moa"])
        moa_df = moa_df[~moa_df["moa"].str.contains(r"\|")]

        phe.new_dataset_from_query(
            "cmap jump moa compounds",
            f"pert_iname in {moa_df.pert_iname.unique().tolist()}",
        )
        del phe[0]
        phe.ds.df = phe.df.merge(
            moa_df[["pert_iname", "moa"]], on="pert_iname", how="left"
        )

        lpcmap_cmap_filtered_profiles_df = None
        try:
            lpcmap_cmap_filtered_profiles_df = pd.read_csv(
                StringIO(
                    importlib.resources.read_text(
                        "leakproofcmap.resources", "lpcmap_cmap_filtered_profiles.tsv"
                    )
                ),
                sep="\t",
                index_col=None,
            )
            phe.ds.df = phe.ds.df.merge(
                lpcmap_cmap_filtered_profiles_df,
                how="right",
                left_on=["plate_id", "well"],
                right_on=["plate_id", "well"],
                suffixes=(None, "_y"),
            )[phe.ds.df.columns]
            phe[0].name = "cmap broad moa compounds"
            phe.save(phe_cmap_l4_clue_moas_downsampled)
            phe.df.to_csv(
                phe_cmap_l4_clue_moas_downsampled.with_suffix(".tsv"), sep="\t"
            )

        except FileNotFoundError as e:
            print(
                "WARNING, leak proof CMAP filtered profiles file which should be in the source package directory under 'leakproofcmap/resources/lpcmap_cmap_filtered_profiles.tsv' not found. We will now generate a new downsampled CMAP. This will highly likely be inconsistent with the published splits and train, validation, and test sets of the Leak Proof CMap paper."
            )
            # We use the clue.io 3/24/2020 "drug information" data https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt to annotate MOAs

            # If overrepresented, downsample.
            downsample_to = 200
            for pert_iname, num_repeats in (
                phe.df.pert_iname.value_counts().to_dict().items()
            ):
                if pert_iname == "DMSO":
                    continue
                if num_repeats > downsample_to:
                    high_rep_cpd_df = phe.df.query("pert_iname==@pert_iname")
                    phe.ds.df = pd.concat(
                        [
                            phe.df.loc[~phe.df.index.isin(high_rep_cpd_df.index)],
                            high_rep_cpd_df.sample(downsample_to),
                        ]
                    )
            dmso_by_cell_line_df_list = []
            dmso_df = phe.df.query("pert_id=='DMSO'")
            phe.ds.df = phe.df.loc[~phe.df.index.isin(dmso_df.index)]

            for _, dmso_gb_df in dmso_df.groupby("cell_id"):
                if len(dmso_gb_df) > downsample_to:
                    dmso_by_cell_line_df_list.append(dmso_gb_df.sample(downsample_to))
                else:
                    dmso_by_cell_line_df_list.append(dmso_gb_df)
            phe.ds.df = pd.concat([phe.df] + dmso_by_cell_line_df_list)

            phe[0].name = f"cmap broad moa compounds {downsample_to} max"
            phe.save(phe_cmap_l4_clue_moas_downsampled)
            phe.df.to_csv(
                phe_cmap_l4_clue_moas_downsampled.with_suffix(".tsv"), sep="\t"
            )
            lpcmap_cmap_filtered_profiles_df = phe.ds.df[
                ["cell_id", "moa", "plate_id", "well"]
            ]
            lpcmap_cmap_filtered_profiles_df.to_csv(
                Path(".") / "lpcmap_cmap_filtered_profiles.tsv", sep="\t", index=None
            )
            print(
                "lpcmap_cmap_filtered_profiles.tsv written to the current working directory, this should be included in the source distribution under leakproofcmap/resources/lpcmap_cmap_filtered_profiles.tsv"
            )
    else:
        phe = phenonaut.Phenonaut.load(phe_cmap_l4_clue_moas_downsampled)
    return phe


def get_phenonaut_object_containing_tests_from_split_object(
    df: pd.DataFrame,
    features: list[str],
    scale: bool,
    pca: bool,
    split_object: Union[CMAPSplit, str, Path],
    dataset_name: Optional[str] = None,
    phenonaut_object_name: Optional[str] = None,
) -> phenonaut.Phenonaut:
    """Get Phenonaut object containing test data from split

    Parameters
    ----------
    df : pd.DataFrame
        CMap DataFrame
    features : list[str]
        List of feature columns within the DataFrame
    scale : bool
        Apply standard scaler to features
    pca : bool
        Apply PCA to features
    split_object : Union[CMAPSplit, str, Path]
        SplitObject, or path as string or Path object to the SplitObject defining the
        split
    dataset_name : Optional[str], optional
        Name to be given to the dataset within the Phenonaut object, by default None
    phenonaut_object_name : Optional[str], optional
        Name of the Phenonaut object, by default None

    Returns
    -------
    phenonaut.Phenonaut
        Phenonaut object with a single dataset from split test data
    """
    if isinstance(split_object, str):
        split_object = Path(split_object)
    if isinstance(split_object, Path):
        split_object = CMAPSplit.load(split_object)

    if dataset_name is None:
        dataset_name = f"Test dataset from {split_object.name}"
    if phenonaut_object_name is None:
        phenonaut_object_name = "Phenonaut test data"

    retrieved_df, retrieved_features = split_object.get_df(
        df=df, features=features, scale=scale, pca=pca, tvt_type="test", fold_number=1
    )
    return phenonaut.Phenonaut(
        retrieved_df,
        name=phenonaut_object_name,
        dataframe_name=dataset_name,
        features=retrieved_features,
    )


def get_split_train_validation_test_dfs_from_cmap(
    cell_line_split_num: int,
    moa_split_num: int,
    working_dir: Union[str, Path] = "working_dir",
    pickle_dir: Union[str, Path] = "pickles",
    phe_cmap_l4_clue_moas_downsampled="phe_ob_CMAP_CLUE_MOAs_downsampled.pkl",
) -> ([pd.DataFrame], [pd.DataFrame], pd.DataFrame):
    """Get train, validation and test dataframes from a split

    Returns dataframes representative of the data in a split. Returns a tuple containing
    a list of training data DataFrames (of length K - where K is the number of folds),
    followed by a list of validation DataFrames (of length K), and finally a test
    DataFrame

    Parameters
    ----------
    cell_line_split_num : int
        Cell line split number (standard splits range from 1 to 5 inclusive)
    moa_split_num : int
        MOA split number (standard splits range from 1 to 5 inclusive)
    working_dir : Union[str, Path]
        Working directory for leakproofcmap. This is usually a 'working_dir' folder in
        the current directory
    pickle_dir : Union[str, Path]
        A directory into which filtered CMap may be written as a pickle file. Usually
        'pickles'. If the path given is relative, then it is placed under the directory
        defined in the 'working_dir' argument. If absolute, then this absolute path is
        used
    phe_cmap_l4_clue_moas_downsampled : str, optional
        Pickle filename for downsampled CMap level 4 data used by LPCMap, by default
        "phe_ob_CMAP_CLUE_MOAs_downsampled.pkl"

    Returns
    -------
    ([pd.DataFrame], [pd.DataFrame], pd.DataFrame)
        Tuple containing three elements, the first being a list of training set data, of
        length K, where K is the number of folds used in cross fold validation (by
        default 5). The second elenment is a list of validation data data frames of
        length K (again, representative of each fold). The final element is a DataFrame
        capturing test data of the split.
    Raises
    ------
    FileNotFoundError
        Raises an error if standard package data resources are not found
    """

    working_dir = Path(working_dir)
    if not working_dir.is_absolute():
        working_dir = working_dir.resolve()

    pickle_dir = Path(pickle_dir)
    if not pickle_dir.is_absolute():
        pickle_dir = working_dir / pickle_dir
    if not pickle_dir.exists():
        pickle_dir.mkdir(parents=True)

    phe_cmap_l4_clue_moas_downsampled = Path(phe_cmap_l4_clue_moas_downsampled)
    if not phe_cmap_l4_clue_moas_downsampled.is_absolute():
        phe_cmap_l4_clue_moas_downsampled = (
            pickle_dir / phe_cmap_l4_clue_moas_downsampled
        )

    if not phe_cmap_l4_clue_moas_downsampled.exists():
        raise FileNotFoundError(
            f"Could not find {phe_cmap_l4_clue_moas_downsampled}, run get_cmap_phenonaut_object to generate Leak Proof CMap downsampled CMap data first"
        )

    phe = phenonaut.Phenonaut.load(phe_cmap_l4_clue_moas_downsampled)
    split = json.loads(
        importlib.resources.read_text(
            "leakproofcmap.resources",
            f"cmap_split_cellidsplit{cell_line_split_num}_moasplit{moa_split_num}.json",
        )
    )
    print(split)

    train_dfs = []
    for fold in split["train"]:
        fold_cell_lines = fold["cell_lines"]
        fold_moas = fold["moas"]
        train_dfs.append(
            phe.df.query("cell_id in @fold_cell_lines and moa in @fold_moas")
        )

    validation_dfs = []
    for fold in split["val"]:
        fold_cell_lines = fold["cell_lines"]
        fold_moas = fold["moas"]
        validation_dfs.append(
            phe.df.query("cell_id in @fold_cell_lines and moa in @fold_moas")
        )

    fold_cell_lines = split["test"][0]["cell_lines"]
    fold_moas = split["test"][0]["moas"]
    test_df = phe.df.query("cell_id in @fold_cell_lines and moa in @fold_moas")

    return train_dfs, validation_dfs, test_df
