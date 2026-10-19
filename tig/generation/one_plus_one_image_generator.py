import torch

from tig.generation.fitness import Fitness
from tig.generation.image_generator import ImageGenerator
from tig.representation.individual import Individual


class OnePlusOneImageGenerator(ImageGenerator):

    def __init__(
        self,
        tensor_image_seeds: torch.Tensor,
        fitness: Fitness,
        num_seeds_to_select: int,
        max_num_of_mutations_per_seed: int,
        num_neighbors: int,
    ) -> None:
        super().__init__(
            fitness=fitness,
            num_seeds_to_select=num_seeds_to_select,
            tensor_image_seeds=tensor_image_seeds,
        )
        self.max_num_of_mutations_per_seed = max_num_of_mutations_per_seed
        self.num_neighbors = num_neighbors

    def generate(self, num_fitness_evaluations: int) -> list[Individual]:
        """
        Generate misclassifications with hill climbing from the selected seeds.
        Args:
            num_fitness_evaluations (int): Number of fitness evaluations (model executions).
        Returns:
            list[Individual]: Misclassified individuals.
        """
        raise NotImplementedError()
