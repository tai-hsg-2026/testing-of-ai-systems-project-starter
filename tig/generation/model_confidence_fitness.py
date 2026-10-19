from torch import nn

from tig.generation.fitness import Fitness
from tig.representation.individual import Individual


class ModelConfidenceFitness(Fitness):

    def __init__(self, model: nn.Module, reference_class: int) -> None:
        super().__init__(model=model, reference_class=reference_class)

    def get_fitness(self, individual: Individual) -> float:
        """
        Execute the model on the individual and compute its fitness from the
        confidence of the model. See Fitness.get_fitness.
        Args:
            individual (Individual): Individual to evaluate.
        Returns:
            float: Fitness value.
        """
        raise NotImplementedError()
