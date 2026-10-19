from abc import ABC, abstractmethod

from torch import nn

from tig.representation.individual import Individual


class Fitness(ABC):

    def __init__(self, model: nn.Module, reference_class: int) -> None:
        self.model = model
        self.reference_class = reference_class

    @abstractmethod
    def get_fitness(self, individual: Individual) -> float:
        """
        Execute the model on the individual and compute its fitness.
        A lower fitness value is closer to a misclassification (i.e., the generators
        minimize the fitness). The method also sets the fitness value and the
        predicted label of the individual (see Individual.set_fitness and Individual.set_label).
        One call is one fitness evaluation, i.e., one execution of the model.
        Args:
            individual (Individual): Individual to evaluate.
        Returns:
            float: Fitness value.
        """
        raise NotImplementedError("Not implemented")
