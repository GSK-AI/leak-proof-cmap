# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import numpy as np
import pandas as pd
from leakproofcmap.benchmarking import pertmutation_test_distinct_from_dmso
import pytest
from phenonaut import Phenonaut
from leakproofcmap.metrics import get_leak_proof_cmap_standard_L1000_metrics


def test_distinctness(synthetic_screening_dataset_1):
    euclidean_metric = get_leak_proof_cmap_standard_L1000_metrics()["Euclidean"]
    phe = Phenonaut(
        synthetic_screening_dataset_1,
        metadata={"features": ["feat_1", "feat_2", "feat_3"]},
    )
    print(phe.ds.df)
    permtest_results = pertmutation_test_distinct_from_dmso(
        phe, "pert_iname", euclidean_metric
    )
    print(permtest_results.sum())
    assert permtest_results.sum().item() == pytest.approx(0.8181999999999999)
