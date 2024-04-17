# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import numpy as np
import pandas as pd
from leakproofcmap.benchmarking import auroc_performance
import pytest
from phenonaut import Phenonaut
from leakproofcmap.metrics import get_leak_proof_cmap_standard_L1000_metrics


def test_uniqueness(synthetic_screening_dataset_1):
    euclidean_metric = get_leak_proof_cmap_standard_L1000_metrics()["Euclidean"]
    phe = Phenonaut(
        synthetic_screening_dataset_1,
        metadata={"features": ["feat_1", "feat_2", "feat_3"]},
    )
    auroc_results = auroc_performance(
        phe, "pert_iname", phenotypic_metric=euclidean_metric
    )
    assert np.mean(
        [np.mean(vs) for vs in auroc_results["trt_aurocs"].values]
    ) == pytest.approx(0.9873697916666666)
