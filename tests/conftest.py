# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_screening_dataset_1():
    """Return simulated screening dataset #1

    sklearn generated regression datasets, 100 rows, 100 features (named feat_n where n
    is 1-100), and one target column which is a regression target.

    Generated using the following python code:
        from sklearn.datasets import make_regression
        X,y=make_regression(random_state=42)
        df = pd.DataFrame(np.hstack([X,y.reshape(-1,1)]), columns=[f"feat_{i+1}" for i in range(X.shape[1])]+["target"])
        df.to_csv("test/generated_regression_dataset.csv")

    """
    n_dims = 3
    n_dmso_wells = 64
    n_treatments = 5
    n_replicates = 4
    random_state = np.random.default_rng(7)
    dataframes = []

    # Append the DMSO DF
    dataframes.append(
        pd.concat(
            [
                pd.DataFrame(
                    random_state.normal([0] * n_dims, 1, size=(n_dmso_wells, n_dims)),
                    columns=[f"feat_{i+1}" for i in range(n_dims)],
                ),
                pd.Series(["DMSO"] * n_dmso_wells, name="pert_iname"),
            ],
            axis=1,
        )
    )

    # Append treatment DFs
    for replicate_i in range(n_treatments):
        dataframes.append(
            pd.concat(
                [
                    pd.DataFrame(
                        random_state.normal(
                            random_state.uniform(-10, 10, n_dims),
                            random_state.uniform(0.5, 3),
                            size=(n_replicates, n_dims),
                        ),
                        columns=[f"feat_{i+1}" for i in range(n_dims)],
                    ),
                    pd.Series(
                        [f"Trt_{replicate_i+1}"] * n_replicates, name="pert_iname"
                    ),
                ],
                axis=1,
            )
        )

    return pd.concat(dataframes)
