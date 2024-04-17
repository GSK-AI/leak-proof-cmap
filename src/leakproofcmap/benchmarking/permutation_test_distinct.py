# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Optional, Callable
from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.special import binom


def _choose_k_bits_from_n_exhaustive_generator(n: int, k: int):
    for combo in map(list, combinations(range(n), k)):
        yield np.array(combo)
    yield None


def _choose_k_bits_from_n_random_generator(n: int, k: int, np_rng: np.random.Generator):
    range_list = range(n)
    while True:
        yield np_rng.choice(range_list, size=k, replace=False)


def _permuation_test_2_dfs(
    trt: np.ndarray,
    veh: np.ndarray,
    dist_func: Callable[[list[float], list[float]], float],
    n_iters=10000,
    random_state: Union[int, np.random.Generator] = 42,
    quiet: bool = False,
):
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    len_trt = len(trt)
    len_veh = len(veh)
    stacked_data = np.vstack([trt, veh])

    n_combinations = binom(len_trt + len_veh, len_trt)
    orig_d = dist_func(np.mean(trt, axis=0), np.mean(veh, axis=0))
    n_gt_or_equal = 0
    full_range = np.array(range(len_trt + len_veh), dtype=int)
    cur_iter = 1
    if n_combinations > n_iters:
        # Sample label permutations
        n_gt_or_equal = 1
        generator = _choose_k_bits_from_n_random_generator(
            len_trt + len_veh, len_trt, random_state
        )
        pbar = tqdm(total=n_iters, disable=quiet)
        while cur_iter < n_iters:
            generated_indexes = next(generator)
            if np.all(np.array(generated_indexes) < len(trt)):
                continue
            cur_iter += 1
            pbar.update(1)
            d = dist_func(
                np.mean(stacked_data[generated_indexes], axis=0),
                np.mean(
                    stacked_data[np.setxor1d(generated_indexes, full_range)], axis=0
                ),
            )
            if d >= orig_d:
                n_gt_or_equal += 1
    else:
        # Exhaustively enumerate label permutations
        n_iters = n_combinations
        n_gt_or_equal = 1
        generator = _choose_k_bits_from_n_exhaustive_generator(
            len_trt + len_veh, len_trt
        )
        generated_indexes = next(generator)
        pbar = tqdm(total=n_iters, disable=quiet)
        pbar.update(1)
        while generated_indexes is not None:
            if np.all(generated_indexes < len(trt)):
                generated_indexes = next(generator)
                continue
            pbar.update(1)
            d = dist_func(
                np.mean(stacked_data[generated_indexes], axis=0),
                np.mean(
                    stacked_data[np.setxor1d(generated_indexes, full_range)], axis=0
                ),
            )
            if d >= orig_d:
                n_gt_or_equal += 1
            generated_indexes = next(generator)

    return n_gt_or_equal / n_iters


def pertmutation_test_distinct_from_dmso(
    ds: Union[Phenonaut, Dataset, list[pd.DataFrame]],
    groupby: Union[str, list[str], None],
    phenotypic_metric: PhenotypicMetric,
    vehicle_treatment_groupby_match: Optional[Union[list[str], str]] = None,
    vehicle_query: str = "pert_iname=='DMSO'",
    n_iters=10000,
    random_state: Union[int, np.random.Generator] = 42,
    max_samples_in_a_group=50,
    quiet: bool = False,
):
    if vehicle_treatment_groupby_match is not None:
        if isinstance(vehicle_treatment_groupby_match, str):
            vehicle_treatment_groupby_match = [vehicle_treatment_groupby_match]
    # Manage RNG
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    if isinstance(phenotypic_metric.func, str):
        if phenotypic_metric.func == "spearman":
            dist_func = (
                lambda x, y: 1
                - np.corrcoef(np.argsort(np.argsort(np.vstack([x, y]))))[0, 1]
            )
        else:
            if phenotypic_metric.higher_is_better:
                dist_func = (
                    lambda x, y: 1
                    - squareform(pdist(np.vstack(x, y), metric=phenotypic_metric.func))[
                        0, 1
                    ]
                )
            else:
                dist_func = lambda x, y: squareform(
                    pdist(np.vstack([x, y]), metric=phenotypic_metric.func)
                )[0, 1]
    else:
        dist_func = phenotypic_metric.distance

    res_df = pd.DataFrame()

    # If lists supplied, perform calc
    if isinstance(ds, list):
        if len(ds) != 2:
            raise ValueError(
                "If supplying a list as the ds argument, then it must be of length 2, with the perturbation dataframe first, followed by the vehicle"
            )
        if not isinstance(ds[0], pd.DataFrame) or not isinstance(ds[1], pd.DataFrame):
            raise ValueError(
                f"If supplying a list as the ds argument, then it must contain the perturbation dataframe first, followed by the vehicle dataframe, supplied types were {type(ds[0])}, and {type(ds[1])}"
            )
        res_df.loc["trt", "pvalue"] = _permuation_test_2_dfs(
            trt=ds[0],
            veh=ds[1],
            dist_func=dist_func,
            n_iters=n_iters,
            random_state=random_state,
            quiet=quiet,
        )
        return res_df

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    if not isinstance(ds, Dataset):
        raise ValueError(
            f"ds was found to be of type '{type(ds)}', should be phenonaut.Phenonaut, phenonaut.Dataset, or list[pd.DataFrame] (of length 2)"
        )

    df = ds.df
    veh_df = df.query(vehicle_query)[ds.features]
    veh_ilocs = [df.index.get_loc(veh_df.iloc[i].name) for i in range(len(veh_df))]

    if groupby is not None:
        grouped_df = df.groupby(groupby)
        grp_id_ilocs = grouped_df.indices.items()
    else:
        grp_id_ilocs = (("all", np.setxor1d(range(len(df)), veh_ilocs)),)

    if veh_df.shape[0] > max_samples_in_a_group:
        veh_df = veh_df.sample(
            n=max_samples_in_a_group, replace=False, random_state=random_state
        )
    for grp_idx, grp_ilocs in tqdm(grp_id_ilocs, disable=quiet):
        if groupby is not None and vehicle_treatment_groupby_match is not None:
            veh_query = " and ".join(
                [
                    (
                        f"{gbmatch}=='{grp_idx[groupby.index(gbmatch)]}'"
                        if isinstance(grp_idx[groupby.index(gbmatch)], str)
                        else f"{gbmatch}=={grp_idx[groupby.index(gbmatch)]}"
                    )
                    for gbmatch in vehicle_treatment_groupby_match
                ]
            )
            veh_df = df.query(f"{vehicle_query} and {veh_query}")[ds.features]
            if veh_df.shape[0] > max_samples_in_a_group:
                veh_df = veh_df.sample(
                    n=max_samples_in_a_group, replace=False, random_state=random_state
                )
        trt_df = df.iloc[grp_ilocs][ds.features]
        if trt_df.shape[0] > max_samples_in_a_group:
            trt_df = trt_df.sample(
                n=max_samples_in_a_group, replace=False, random_state=random_state
            )
        res_df.loc[str(grp_idx), "pvalue"] = _permuation_test_2_dfs(
            trt=trt_df,
            veh=veh_df,
            dist_func=dist_func,
            n_iters=n_iters,
            random_state=random_state,
            quiet=True,
        )
    return res_df
