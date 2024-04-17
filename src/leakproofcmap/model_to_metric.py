# Copyright © The University of Edinburgh, 2024.
# Development has been supported by GSK.

from .metrics import PhenotypicMetric


class NNModuleToMetric:
    """Convenience objects to help turn NN modules into LPCMap usable metrics"""

    def __init__(self, nn_module_class, checkpoint, cuda=True):
        self.model = nn_module_class.load_from_checkpoint(checkpoint)
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

    def __call__(self, x1, x2):
        return self.model.score_numpy(x1, x2)


def get_phenonaut_metric_from_model_checkpoint(
    name: str,
    nn_module_class,
    checkpoint,
    higher_is_better: bool = False,
    cuda: bool = True,
) -> PhenotypicMetric:
    """Get a phenotypic metric from a model checkpoint

    Parameters
    ----------
    name : str
        Name for the metric
    nn_module_class : Object
        Uninstantiated neural network module from which to derive the metric
    checkpoint : Object
        Checkpoint file containing the trained model weights and biases to be used in
        creation of the metric
    higher_is_better : bool, optional
        If True, then higher values are deemed better, by default False
    cuda : bool, optional
        Use CUDA devices to run the NN module and perform metric calculations. If False,
        then only the CPU is used for calculation, by default True

    Returns
    -------
    PhenotypicMetric
        LPCMap PhenotypicMetric object derived from a trained neural network
    """
    nn_scorer = NNModuleToMetric(
        nn_module_class=nn_module_class, checkpoint=checkpoint, cuda=cuda
    )
    new_phenotypic_metric = PhenotypicMetric(
        name, nn_scorer, (0, 2), higher_is_better=higher_is_better
    )
    new_phenotypic_metric.model = nn_scorer.model
    return new_phenotypic_metric
