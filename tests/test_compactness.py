# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import numpy as np
import pandas as pd
from leakproofcmap.benchmarking import percent_replicating
import pytest


def test_compactness(synthetic_screening_dataset_1):
    pr_results = percent_replicating(
        synthetic_screening_dataset_1,
        "pert_iname",
        "pert_iname!='DMSO'",
        features=["feat_1", "feat_2", "feat_3"],
    )
    assert pr_results == pytest.approx(80.0)
