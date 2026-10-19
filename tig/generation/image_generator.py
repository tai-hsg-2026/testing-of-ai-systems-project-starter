from abc import ABC, abstractmethod

import torch

from tig.generation.fitness import Fitness
from tig.representation.individual import Individual
from tig.utils.randomness import RandomGenerator


class ImageGenerator(ABC):

    def __init__(
        self,
        fitness: Fitness,
        tensor_image_seeds: torch.Tensor,
        num_seeds_to_select,
    ) -> None:
        self.random_generator = RandomGenerator.get_instance()
        self.tensor_image_seeds = tensor_image_seeds
        # Set before select_seeds, so that select_seeds can use the fitness
        self.fitness = fitness
        assert num_seeds_to_select <= len(
            tensor_image_seeds
        ), f"Number of seeds to select {num_seeds_to_select} cannot be greater than the number of available seeds {len(tensor_image_seeds)}."
        self.num_seeds_to_select = num_seeds_to_select
        # indices (in tensor_image_seeds) of the seeds the generator can start from
        self.selected_seed_indices: list[int] = self.select_seeds()

    def select_seeds(self) -> list[int]:
        """
        Select a random subset of self.num_seeds_to_select seeds.
        Returns:
            list[int]: Indices (in self.tensor_image_seeds) of the selected seeds.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate(self, num_fitness_evaluations: int) -> list[Individual]:
        """
        Generate inputs with num_fitness_evaluations fitness evaluations (model executions).
        Args:
            num_fitness_evaluations (int): Number of fitness evaluations (model executions).
        Returns:
            list[Individual]: Misclassified individuals.
        """
        raise NotImplementedError("Not implemented")
