import torch

from tig.representation.individual import Individual


class FeatureMap:

    def __init__(self, tensor_images: torch.Tensor) -> None:
        """
        Make an empty feature map with two features: boldness and orientation.
        The ranges and the granularity of the features are computed automatically 
        from a set of images.
        Args:
            tensor_images (torch.Tensor): Images (normalized) with shape (N, 1, 1, 28, 28), as in the seed pool.
                They set the ranges and the granularity.
        """
        raise NotImplementedError()

    def fill_map(self, misclassified_individuals: list[Individual]) -> None:
        """
        Put each individual in the cell of its feature values.
        Args:
            misclassified_individuals (list[Individual]): Individuals to put in the map.
        """
        raise NotImplementedError()

    def get_shape(self) -> tuple[int, int]:
        """
        Get the shape of the feature map.
        Returns:
            tuple[int, int]: Number of cells for the boldness and number of cells for the orientation.
        """
        raise NotImplementedError()

    def get_cell(self, individual: Individual) -> tuple[int, int]:
        """
        Get the cell of an individual, i.e., the cell of its feature values.
        The individual does not go in the map (see fill_map).
        Args:
            individual (Individual): Individual.
        Returns:
            tuple[int, int]: Cell for the boldness and cell for the orientation.
        """
        raise NotImplementedError()

    def get_num_individuals(self) -> int:
        """
        Get the number of individuals in the map.
        Returns:
            int: Number of individuals that fill_map put in the map.
        """
        raise NotImplementedError()

    def get_coverage(self) -> float:
        """
        Get the coverage of the feature map.
        Returns:
            float: Number of cells with at least one individual, divided by the number of cells.
        """
        raise NotImplementedError()
