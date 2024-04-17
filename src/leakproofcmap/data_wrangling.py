# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.stats import percentileofscore
from tqdm import tqdm


class _ModelMetric:
    def __init__(self, name, higher_is_better) -> None:
        self.name = name
        self.higher_is_better = higher_is_better


def postprocess_pr_files_unmerged(
    seed: int = 7,
    model_name: str = "TripletLoss",
    model_identifier: str = "155",
    working_dir: Union[str, Path] = "working_dir",
    restrict_pr_type: Optional[str] = None,
    jump_moa_compounds_only: bool = False,
):
    """Postprocess unmerged PR files into plot data of percent replicating vs cutoff

    Parameters
    ----------
    seed : int, optional
        The random number seed used in training of models and calculation of percent
        replicating. This is used for identification of output PR files, by default 7
    model_name : str, optional
        Name of the model. This is used for identification of output PR files, by
        default "TripletLoss"
    model_identifier : str, optional
        Model identifier. This is used for identification of output PR files, by default
        "155"
    working_dir : Union[str, Path]
        Working directory for leakproofcmap. This is usually a 'working_dir' folder in
        the current directory
    restrict_pr_type : Optional[str], optional
        Optionally restrict the analysis to a specific PR type (any of 'compactness',
        'compactness_across_lines', 'compactness_moas', or
        'compactness_across_lines_moas'). If None, then each PR type is calculated/
         derived from files, by default None
    jump_moa_compounds_only : bool, optional
        If True, then only compounds in the JUMP MOA dataset are included in the
        analysis, by default False
    """

    working_dir = Path(working_dir)

    unmerged_pr_output_dir = (
        working_dir
        / Path("plot_data")
        / Path(model_name)
        / Path(model_identifier)
        / Path("compactnessunmerged")
    )
    if not unmerged_pr_output_dir.exists():
        unmerged_pr_output_dir.mkdir(parents=True, exist_ok=True)
    from .metrics import leak_proof_cmap_metrics

    pr_visualisations = [
        {
            "pr_type": "compactness",
            "raw_plot_data": unmerged_pr_output_dir
            / Path(
                f"plot_compactnessunmerged_cutoffs_cmap_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
        },
        {
            "pr_type": "compactness_across_lines",
            "raw_plot_data": unmerged_pr_output_dir
            / Path(
                f"plot_compactnessunmerged_cutoffs_cmap_across_lines_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
        },
        {
            "pr_type": "compactness_moas",
            "raw_plot_data": unmerged_pr_output_dir
            / Path(
                f"plot_compactnessunmerged_cutoffs_cmap_moas_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
        },
        {
            "pr_type": "compactness_across_lines_moas",
            "raw_plot_data": unmerged_pr_output_dir
            / Path(
                f"plot_compactnessunmerged_cutoffs_cmap_across_lines_moas_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
        },
    ]

    metrics_for_which_a_warning_has_been_issued = []
    for pr_visualisation in pr_visualisations:
        if (
            restrict_pr_type is not None
            and pr_visualisation["pr_type"] != restrict_pr_type
        ):
            continue

        # Process raw PR files to a pr curve, reevaluating percent replicating for percentiles 0-100.
        split_pr_files_dir = (
            working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path(pr_visualisation["pr_type"])
        )

        metric_names = set(
            [
                f.stem.split("_")[1]
                for f in split_pr_files_dir.glob(
                    f"pr_*cls*ps*_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
                )
            ]
        )
        plot_df = pd.DataFrame()

        print(f"Working on metrics: {metric_names}")
        for metric_name in metric_names:
            if metric_name not in leak_proof_cmap_metrics:
                higher_is_better = False
                if (
                    metric_name not in metrics_for_which_a_warning_has_been_issued
                    and metric_name != "TripletLoss"
                ):
                    f"Warning, {metric_name} not found in default metrics list, it will be assumed that it is a distance metric and lower is better"
                    metrics_for_which_a_warning_has_been_issued.append(metric_name)
            else:
                higher_is_better = leak_proof_cmap_metrics[metric_name].higher_is_better

            split_pr_cutoff_df = pd.DataFrame()
            for split_pr_file in tqdm(
                list(
                    (split_pr_files_dir).glob(
                        f"pr_{metric_name}_*cls*ps*_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
                    )
                ),
                desc=f"Processing {metric_name} for unmerged {pr_visualisation['pr_type']}",
            ):
                split_identifier = ",".join(
                    str(split_pr_file.stem).split("_")[2:4]
                ).replace(",", "_")
                split_pr_df = pd.read_csv(split_pr_file)

                split_pr_df = _get_pr_over_cutoffs(split_pr_df, higher_is_better)
                split_pr_cutoff_df[["cutoff percentile", "percent replicating"]] = [
                    (
                        i,
                        np.nansum(split_pr_df["median_replicate_percentile"] > i)
                        / split_pr_df.shape[0]
                        * 100.0,
                    )
                    for i in range(0, 101)
                ]
                split_pr_cutoff_df["similarity metric"] = metric_name
                split_pr_cutoff_df["split"] = split_identifier
                plot_df = pd.concat([plot_df, split_pr_cutoff_df], ignore_index=True)
        plot_df.to_csv(pr_visualisation["raw_plot_data"])


def postprocess_pr_files(
    seed: int = 7,
    model_name: str = "TripletLoss",
    model_identifier: str = "155",
    working_dir: Union[str, Path] = "working_dir",
    restrict_pr_type: Optional[str] = None,
    jump_moa_compounds_only: bool = False,
):
    """Postprocess unmerged PR files into plot data of percent replicating vs cutoff

    Parameters
    ----------
    seed : int, optional
        The random number seed used in training of models and calculation of percent
        replicating. This is used for identification of output PR files, by default 7
    model_name : str, optional
        Name of the model. This is used for identification of output PR files, by
        default "TripletLoss"
    model_identifier : str, optional
        Model identifier. This is used for identification of output PR files, by default
        "155"
    working_dir : Union[str, Path]
        Working directory for leakproofcmap. This is usually a 'working_dir' folder in
        the current directory
    restrict_pr_type : Optional[str], optional
        Optionally restrict the analysis to a specific PR type (any of 'compactness',
        'compactness_across_lines', 'compactness_moas', or
        'compactness_across_lines_moas'). If None, then each PR type is calculated/
         derived from files, by default None
    jump_moa_compounds_only : bool, optional
        If True, then only compounds in the JUMP MOA dataset are included in the
        analysis, by default False
    """
    from .metrics import leak_proof_cmap_metrics

    pr_visualisations = [
        {
            "pr_type": "compactness_across_lines",
            "raw_plot_data": working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path("compactness_across_lines")
            / Path(
                f"plot_compactness_cutoffs_cmap_across_lines_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
            "plot_title": f"CMAP {'JUMP MOA compounds ' if jump_moa_compounds_only else ''}Percent replicating treatments across lines vs percentile cutoffs,\nreplicate criteria='pert_iname', replicate_criteria_not='cell_id', null_criteria='cell_id'",
        },
        {
            "pr_type": "compactness",
            "raw_plot_data": working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path("compactness")
            / Path(
                f"plot_compactness_cutoffs_cmap_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
            "plot_title": f"CMAP {'JUMP MOA compounds ' if jump_moa_compounds_only else ''}Percent replicating treatments vs percentile cutoffs,\nreplicate_criteria=['pert_iname', 'pert_idose_uM', 'cell_id'], null_criteria=['pert_iname', 'pert_idose_uM', 'cell_id']",
        },
        {
            "pr_type": "compactness_moas",
            "raw_plot_data": working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path("compactness_moas")
            / Path(
                f"plot_compactness_cutoffs_cmap_moas_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
            "plot_title": f"CMAP {'JUMP MOA compounds ' if jump_moa_compounds_only else ''}Percent replicating MOAs vs percentile cutoffs,\nreplicate_criteria=['moa', 'cell_id'], null_criteria='cell_id'",
        },
        {
            "pr_type": "compactness_across_lines_moas",
            "raw_plot_data": working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path("compactness_across_lines_moas")
            / Path(
                f"plot_compactness_cutoffs_cmap_across_lines_moas_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
            ),
            "plot_title": f"CMAP {'JUMP MOA compounds ' if jump_moa_compounds_only else ''}Percent replicating MOAs across lines vs percentile cutoffs,\nreplicate_criteria='moa', replicate_criteria_not='cell_id', null_criteria='cell_id'",
        },
    ]

    metrics_for_which_a_warning_has_been_issued = []
    for pr_visualisation in pr_visualisations:
        if (
            restrict_pr_type is not None
            and pr_visualisation["pr_type"] != restrict_pr_type
        ):
            continue
        split_pr_files_dir = (
            working_dir
            / Path("plot_data")
            / Path(model_name)
            / Path(model_identifier)
            / Path(pr_visualisation["pr_type"])
        )

        metric_names = set(
            [
                f.stem.split("_")[1]
                for f in split_pr_files_dir.glob(
                    f"pr_*cls*ps*_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
                )
            ]
        )
        plot_df = pd.DataFrame()

        print(f"Working on metrics: {metric_names}")
        for metric_name in metric_names:
            print(f"Processing {metric_name} for {pr_visualisation['pr_type']}")
            merged_pr_file = pd.concat(
                [
                    pd.read_csv(f)
                    for f in (split_pr_files_dir).glob(
                        f"pr_{metric_name}_*cls*ps*_seed_{seed}{'_jumpmoa' if jump_moa_compounds_only else ''}.csv"
                    )
                ]
            )
            chunk_of_plot_df = pd.DataFrame()

            if metric_name not in leak_proof_cmap_metrics:
                higher_is_better = False
                if (
                    metric_name not in metrics_for_which_a_warning_has_been_issued
                    and metric_name != "TripletLoss"
                ):
                    f"Warning, {metric_name} not found in default metrics list, it will be assumed that it is a distance metric and lower is better"
                    metrics_for_which_a_warning_has_been_issued.append(metric_name)
            else:
                higher_is_better = leak_proof_cmap_metrics[metric_name].higher_is_better

            merged_pr_file = _get_pr_over_cutoffs(merged_pr_file, higher_is_better)
            chunk_of_plot_df[["cutoff percentile", "percent replicating"]] = [
                (
                    i,
                    np.nansum(merged_pr_file["median_replicate_percentile"] > i)
                    / merged_pr_file.shape[0]
                    * 100.0,
                )
                for i in tqdm(range(0, 101))
            ]
            chunk_of_plot_df["similarity metric"] = metric_name
            plot_df = pd.concat([plot_df, chunk_of_plot_df], ignore_index=True)
        plot_df.to_csv(pr_visualisation["raw_plot_data"])


def _get_pr_over_cutoffs(merged_pr_file: pd.DataFrame, higher_is_better: bool):
    """Calculate PR values over a range of cutoffs

    Parameters
    ----------
    merged_pr_file : pd.DataFrame
        Merged PR DataFrame from files
    higher_is_better : bool
        If True, then it is assumed that higher is better and replicates passing
        replicating criteria must have median all to all distances higher than each
        explored percentile. If False, then they must have values lower than each
        explored percentile to be deemed replicating

    Returns
    -------
    pd.DataFrame
        DataFrame containing percent replicaitng values over a range of percentiles
    """
    if higher_is_better:
        merged_pr_file["median_replicate_percentile"] = merged_pr_file.apply(
            lambda x: percentileofscore(
                x[[f"null_{i}" for i in range(1, 1001)]],
                x["median_replicate_score"],
                kind="strict",
                nan_policy="omit",
            ),
            axis=1,
        )
    else:
        merged_pr_file["median_replicate_percentile"] = merged_pr_file.apply(
            lambda x: percentileofscore(
                np.negative(x[[f"null_{i}" for i in range(1, 1001)]]),
                np.negative(x["median_replicate_score"]),
                kind="strict",
                nan_policy="omit",
            ),
            axis=1,
        )
    return merged_pr_file
