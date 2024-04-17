# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Optional
from phenonaut import Phenonaut
from phenonaut.data import Dataset
from phenonaut.metrics.non_ds_phenotypic_metrics import PhenotypicMetric
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import os

_d = None
_metric = None


def _auroc_parallel_eval_phe_worker_init(big_data, metric):
    global _d, _metric
    _d = big_data[:]
    _metric = deepcopy(metric)
    print(os.sched_getaffinity(0))
    os.sched_setaffinity(0, range(mp.cpu_count()))


def _auroc_get_metrics_results_dict(
    y_true,
    scores,
):
    all_treatment_roc_scores = [roc_auc_score(y_true, s) for s in scores]
    return {
        "n_samples": len(scores),
        "trt_meanauroc": np.mean(all_treatment_roc_scores),
        "trt_aurocs": [all_treatment_roc_scores],
    }


def _auroc_parallel_eval_work(grp_key_indices_tuple):
    global _d
    global _metric
    d = _d[:]
    metric = deepcopy(_metric)
    scores = np.empty((len(grp_key_indices_tuple[1]), d.shape[0]))
    for i, q in enumerate(d[grp_key_indices_tuple[1]]):
        scores[i] = -metric.distance(q, d)
    y_true = np.zeros(d.shape[0], dtype=int)
    y_true[grp_key_indices_tuple[1]] = 1
    all_treatment_roc_scores = [roc_auc_score(y_true, s) for s in scores]
    results_dict = {
        "n_samples": len(scores),
        "trt_meanauroc": np.mean(all_treatment_roc_scores),
        "trt_aurocs": [all_treatment_roc_scores],
    }
    return grp_key_indices_tuple[0], results_dict


def auroc_performance(
    ds: Union[Phenonaut, Dataset],
    groupby: Union[str, list[str]],
    phenotypic_metric: PhenotypicMetric,
    parallel: bool = True,
    allowed_pert_inames: Optional[list[str]] = None,
):
    print("Affinity: ", os.sched_getaffinity(0))
    os.sched_setaffinity(0, range(1024))

    if isinstance(ds, Phenonaut):
        ds = ds[-1]
    grouped_df = ds.df.groupby(groupby)

    grp_id_ilocs = grouped_df.indices.items()
    if allowed_pert_inames is not None:
        allowed_ilocs = np.argwhere(
            [v in allowed_pert_inames for v in ds.df.pert_iname.values]
        )
        grp_id_ilocs = [
            (gid, [il for il in grouped_df.indices[gid] if il in allowed_ilocs])
            for gid in grouped_df.indices
        ]
        grp_id_ilocs = [(gid, ils) for gid, ils in grp_id_ilocs if len(ils) > 0]
    group_label_list = list(grouped_df.indices.keys())
    y = np.empty(ds.df.shape[0], dtype=int)
    for group_name, indexes in grouped_df.indices.items():
        y[indexes] = group_label_list.index(group_name)

    res_df = pd.DataFrame()
    if parallel:
        with mp.Pool(
            processes=None,
            initializer=_auroc_parallel_eval_phe_worker_init,
            initargs=(
                ds.data.values,
                phenotypic_metric,
            ),
        ) as pool:
            for grp_name, res in tqdm(
                pool.imap_unordered(
                    _auroc_parallel_eval_work, grp_id_ilocs, chunksize=1
                ),
                total=len(grp_id_ilocs),
            ):
                res_df = pd.concat(
                    [res_df, pd.DataFrame(res, index=[grp_name])], axis=0
                )
    else:
        for grp_key_indices_tuple in tqdm(grp_id_ilocs):
            scores = np.empty((len(grp_key_indices_tuple[1]), ds.df.shape[0]))
            for i, q in enumerate(ds.data.values[grp_key_indices_tuple[1]]):
                scores[i] = -phenotypic_metric.distance(q, ds.data.values).flatten()
            y_true = np.zeros(ds.df.shape[0], dtype=int)
            y_true[grp_key_indices_tuple[1]] = 1
            all_treatment_roc_scores = [roc_auc_score(y_true, s) for s in scores]
            results_dict = {
                "n_samples": len(scores),
                "trt_meanauroc": np.mean(all_treatment_roc_scores),
                "trt_aurocs": [all_treatment_roc_scores],
            }
            res_df = pd.concat(
                [res_df, pd.DataFrame(results_dict, index=[grp_key_indices_tuple[0]])],
                axis=0,
            )
    return res_df
