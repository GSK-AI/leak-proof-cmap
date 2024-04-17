# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from typing import Union, Callable, Tuple
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import cdist


def calc_zhang_scores(anchor: np.array, queries: np.array) -> Union[float, np.ndarray]:
    """Calculate Zhang scores between two np.ndarrays

    Implementation of the Zhang method for comparing L1000/CMAP signatures.
    Zhang, Shu-Dong, and Timothy W. Gant. "A simple and robust method for
    connecting small-molecule drugs using gene-expression signatures." BMC
    bioinformatics 9.1 (2008): 1-10.
    Implemented by Steven Shave, following above paper as a reference
    https://doi.org/10.1186/1471-2105-9-258

    Parameters
    ----------
    anchor : np.array
        Anchor profiles/features. Can be a MxN matrix, allowing M sequences to
        be queried against queries (using N features).
    queries : np.array
        Candidate profiles/features. Can be a MxN matrix, allowing M candidate
        sequences to be evaluated against anchor sequences.

    Returns
    -------
    Union[float, np.ndarray]
        If anchor and candidate array ndims are both 1, then a single float
        representing the Zhang score is returned. If one input array has ndims
        of  2 (and the other has ndims of 1), then a 1-D np.ndarray is
        returned. If both inputs are 2-D, then a 2D MxN array is returned, where
        M is the


    """
    if queries is None:
        queries = anchor
    if anchor.ndim != 1:
        if anchor.ndim == 2:
            multi_anchor_results = np.full((anchor.shape[0], queries.shape[0]), np.nan)
            for i in range(anchor.shape[0]):
                multi_anchor_results[i, :] = calc_zhang_scores(anchor[i], queries)
            return multi_anchor_results
        else:
            raise ValueError(
                f"Anchor should be a 1D array, it had shape: {anchor.shape}"
            )
    anchor_profile = rankdata(np.abs(anchor), axis=-1) * np.sign(anchor)
    if queries.ndim == 1:
        queries.reshape(1, -1)
    if queries.shape[-1] != anchor.shape[-1]:
        raise ValueError(
            f"Different number of features found in anchor ({anchor.shape[-1]}) and queries ({queries.shape[-1]})"
        )
    query_profiles = rankdata(np.abs(queries), axis=-1) * np.sign(queries)
    return np.sum(anchor_profile * query_profiles, axis=-1) / np.sum(
        anchor_profile**2, axis=-1
    )


def calc_zhang_scores_all_v_all(anchor: np.array):
    anchor = np.array(anchor)
    if anchor.ndim == 2:
        multi_anchor_results = np.full((anchor.shape[0], anchor.shape[0]), np.nan)
        query_profiles = rankdata(np.abs(anchor), axis=-1) * np.sign(anchor)
        for i in range(anchor.shape[0]):
            anchor_profile = rankdata(np.abs(anchor[i]), axis=-1) * np.sign(anchor[i])
            multi_anchor_results[i, :] = np.sum(
                anchor_profile * query_profiles, axis=-1
            ) / np.sum(anchor_profile**2, axis=-1)
        return multi_anchor_results
    else:
        raise ValueError(f"Anchor should be a 2D array, it had shape: {anchor.shape}")


def calc_spearmansrank_scores(
    anchor: np.array, queries: np.array
) -> Union[float, np.ndarray]:
    """Calculate spearman's rank correlations

    Parameters
    ----------
    anchor : np.array
        Anchor (or candidate) data
    queries : np.array
        Query data

    Returns
    -------
    Union[float, np.ndarray]
        Single spearman's rank correlation coefficient if queries is 1D, or a  Numpy
        array if the query is 2D

    Raises
    ------
    ValueError
        Error is raised if queries is higher than 2D
    """
    # Standardise inputs to np arrays
    anchor = np.array(anchor)
    if queries is None:
        queries = np.array(anchor)
    else:
        queries = np.array(queries)

    if anchor.ndim == 1:
        if queries.ndim == 1:
            return spearmanr(anchor, queries).correlation
        else:
            return np.array([spearmanr(anchor, q).correlation for q in queries])
    else:
        if anchor.ndim != 2:
            raise ValueError("Anchor should be 1D or 2D, it was {anchor.ndim}D")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        results_array = np.empty((anchor.shape[0], queries.shape[0]))

        for i in range(anchor.shape[0]):
            results_array[i] = np.array(
                [spearmanr(anchor[i], q).correlation for q in queries]
            )
        return results_array


class PhenotypicMetric:
    """Metrics evaluate one profile/feature vector against another

    SciPy and other libraries traditionally supply distance metrics, like
    Manhattan, Euclidean etc. These are typically unbound in their max value,
    but not always, for example; cosine distance with a maximum dissimilarity of
    1.  Scientific literature is also full of similarity metrics, where a high
    value indicates most similarity - the opposite of a similarity metric. This
    dataclass coerces metrics into a standard form, with .similarity and
    .distance functions to turn any metric into a similarity or distance metric.

    This allows the definition of something like the Zhang similarity metric,
    which ranges from -1 to 1, indicating most dissimilarity and most similarity
    respectively. Calling the metric defined by this Zhang function will return
    the traditional Zhang metric value - ranging from -1 to 1.

    The methods similarity and distance will also be added

    Calling distance will return a value between 0 and 1, with 0 being most
    similar and 1 being most dissimilar.

    Calling similarity will return a value between 0 and 1, with 1 being the
    most similar and 0 being the most different.

    """

    def __init__(
        self,
        name: str,
        method: Union[Callable, str],
        range: Tuple[float, float],
        higher_is_better: bool = True,
    ):
        """Constructor for PhenotypicMetric object

        Parameters
        ----------
        name : str
            Name of the metric
        method : Union[Callable, str]
            Callable method taking two arguments (nd.arrays) and returning a similarity
            or distance score between the two datasets
        range : Tuple[float, float]
            The lower and upper bounds of values returnable by the method. Elements
            may be None, meaning unbounded
        higher_is_better : bool, optional
            If True, then higher values are deemed better, by default False
        """
        self.name = name
        self.func = method
        self.range = range
        self.higher_is_better = higher_is_better
        if isinstance(self.func, str):
            self.is_magic_string = True
        else:
            self.is_magic_string = False
        if not any(np.isinf(self.range)):
            self.scalable = True
        else:
            self.scalable = False

    def __repr__(self) -> str:
        """Return the name of the phenotypic similarity method

        Returns
        -------
        str
            Name of the phenotypic similarity method
        """
        return self.name

    def __str__(self) -> str:
        """Return the name of the phenotypic similarity method

        Returns
        -------
        str
            Name of the phenotypic similarity method
        """
        return self.name

    def __call__(self, anchor, query):
        """Apply the similarity method to two passed arrays

        Parameters
        ----------
        anchor : np.array
            Anchor (or candidate) data
        queries : np.array
            Query data

        Returns
        -------
        float
            Phenotypic similarity method result

        Raises
        ------
        ValueError
            Anchor was not one dimensional
        """
        anchor = np.array(anchor)
        query = np.array(query)
        if anchor.ndim != 1:
            raise ValueError(
                f"Expected anchor to have 1 dimension, it had {anchor.ndim}"
            )

        if query.ndim == 2:
            return np.array([self.__call__(anchor, row) for row in query]).flatten()
        if isinstance(self.func, str):
            if self.func == "spearman":
                d1 = np.atleast_2d(anchor)
                d2 = np.atleast_2d(query)
                return ((d1 * d2).mean(axis=1) - d1.mean(axis=1) * d2.mean(axis=1)) / (
                    d1.std(axis=1) * d2.std(axis=1)
                ).flatten()
            else:
                return cdist(
                    anchor.reshape(1, -1), query.reshape(1, -1), metric=self.func
                ).flatten()
        else:
            return self.func(anchor, query)

    def scale(self, score):
        return (score - self.range[0]) / (self.range[1] - self.range[0])

    def distance(self, anchor, query):
        """Calculate distance from similarity method

        Parameters
        ----------
        anchor : np.array
            Anchor (or candidate) data
        queries : np.array
            Query data

        Returns
        -------
        float
            Phenotypic distance
        """
        score = self.__call__(anchor, query)

        if self.scalable:
            score = self.scale(score)
            if self.higher_is_better:
                return 1.0 - score
            return score
        else:
            if self.higher_is_better:
                return 1.0 - score
            return score

    def similarity(self, anchor, query):
        """Calculate similarity from similarity method

        Parameters
        ----------
        anchor : np.array
            Anchor (or candidate) data
        queries : np.array
            Query data

        Returns
        -------
        float
            Phenotypic similarity method result
        """
        score = self.__call__(anchor, query)
        if self.scalable:
            score = self.scale(score)
            if self.higher_is_better:
                return score
            else:
                return 1 - score
        else:
            if self.higher_is_better:
                return score
            else:
                if isinstance(score, float):
                    if score == 0:
                        return 1
                    return 1.0 / score
                return 1.0 / np.clip(score, np.finfo(float).eps, None)


# Metrics with magic values - magic values referrs to the fast methods which  scipy's pdist/cidist use
metrics_with_magic_values = {
    "Rank": PhenotypicMetric("Rank", "spearman", (-1, 1)),
    "Euclidean": PhenotypicMetric(
        "Euclidean", "euclidean", (0, np.inf), higher_is_better=False
    ),
    "Manhattan": PhenotypicMetric(
        "Manhattan", "cityblock", (0, np.inf), higher_is_better=False
    ),
    "Cosine": PhenotypicMetric("Cosine", "cosine", (0, 2), higher_is_better=False),
    "Zhang": PhenotypicMetric("Zhang", calc_zhang_scores, (-1, 1)),
}


def get_leak_proof_cmap_standard_L1000_metrics():
    """Get performance evaluation metrics

    Returns
    -------
    dict
        A dictionary with keys as the name of the metric and values being the Phenonaut
        wrapped metric
    """
    from copy import deepcopy

    cpu_metrics = metrics_with_magic_values.copy()
    del cpu_metrics["Manhattan"]
    del cpu_metrics["Euclidean"]
    cpu_metrics["EuclideanPCA"] = deepcopy(metrics_with_magic_values["Euclidean"])
    cpu_metrics["EuclideanPCA"].name = "EuclideanPCA"
    cpu_metrics["EuclideanPCA"].apply_scaler = True
    cpu_metrics["EuclideanPCA"].apply_pca = True
    cpu_metrics["Cosine"].apply_scaler = True
    cpu_metrics["Cosine"].apply_pca = True
    cpu_metrics["Euclidean"] = deepcopy(metrics_with_magic_values["Euclidean"])
    cpu_metrics["Euclidean"].name = "Euclidean"
    return cpu_metrics


leak_proof_cmap_metrics = get_leak_proof_cmap_standard_L1000_metrics()
