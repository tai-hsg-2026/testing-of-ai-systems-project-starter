import matplotlib.pyplot as plt
import numpy as np
from svgelements import SVG
from torch import Tensor

from tig.representation.rasterization import rasterize
from tig.representation.vectorization import vectorize
from tig.utils.randomness import RandomGenerator
from tig.utils.torch_utils import tensor_image_to_array_image


class _IdCounter:
    __instance: "_IdCounter" = None

    @staticmethod
    def get_instance(start_count: int = 0) -> "_IdCounter":
        if _IdCounter.__instance is None:
            _IdCounter(start_count=start_count)
        return _IdCounter.__instance

    def __init__(self, start_count: int = 0):
        if _IdCounter.__instance is not None:
            raise RuntimeError("This class is a singleton!")

        self.id = start_count
        _IdCounter.__instance = self

    def get_next_id(self) -> int:
        current_id = self.id + 1
        self.id = current_id
        return current_id


class Individual:

    def __init__(
        self,
        tensor_image: Tensor,
        svg_object: SVG | None = None,
        start_count_id: int = 0,
        seed_idx: int = -1,
    ) -> None:
        if svg_object is None:
            self.svg_object = vectorize(tensor_image=tensor_image)
        else:
            self.svg_object = svg_object
        self.tensor_image = tensor_image
        self.random_generator = RandomGenerator.get_instance()
        self.seed_idx = seed_idx
        self.id = _IdCounter.get_instance(start_count=start_count_id).get_next_id()
        # The fitness sets the fitness value and the label (see Fitness.get_fitness)
        self.fitness_value: float | None = None
        self.label: int | None = None

    # docs for the library: https://github.com/meerk40t/svgelements/
    # svg editor: https://yqnn.github.io/svg-path-editor/
    def mutate(self) -> "Individual":
        """
        Mutate the svg representation of the individual.
        Returns:
            Individual: Mutated individual.
        """
        raise NotImplementedError()

    def get_boldness_feature_value(self, threshold: float) -> int:
        """
        Get the boldness feature value, i.e., the number of pixels of the
        current image (svg representation) that are greater than or equal to the threshold.
        Args:
            threshold (float): The threshold to consider a pixel as "bold".
        Returns:
            int: The boldness feature value.
        """
        raise NotImplementedError()

    def get_orientation_feature_value(self, threshold: float) -> int:
        """
        Get the orientation feature value, i.e., the angle of the digit in the
        current image (svg representation) from the vertical, in degrees.
        Args:
            threshold (float): The threshold to consider a pixel as part of the digit.
        Returns:
            int: The orientation feature value.
        """
        raise NotImplementedError()

    def get_fitness(self) -> float:
        assert self.fitness_value is not None, f"Individual {self.id} is not evaluated"
        return self.fitness_value

    def set_fitness(self, fitness_value: float) -> None:
        self.fitness_value = fitness_value

    def get_label(self) -> int:
        assert self.label is not None, f"Individual {self.id} is not evaluated"
        return self.label

    def set_label(self, label: int) -> None:
        self.label = label

    def get_str_representation(self) -> str:
        return self.svg_object.string_xml()

    def get_image_array_and_tensor_representation(self) -> tuple[np.ndarray, Tensor]:
        array_image, tensor_image = rasterize(svg_str=self.get_str_representation())
        return array_image, tensor_image

    def plot_svg_representation(self, filename: str) -> None:
        with open(f"{filename}.svg", "w") as f:
            f.write(self.get_str_representation())

    def plot_current_image(self, filename: str) -> None:
        array_image, _ = self.get_image_array_and_tensor_representation()
        self._plot_image(array_image=array_image, filename=filename)

    @staticmethod
    def _plot_image(array_image: np.ndarray, filename: str) -> None:
        plt.imshow(array_image, cmap="grey")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.close()

    def plot_original_image(self, filename: str) -> None:
        array_image = tensor_image_to_array_image(tensor_image=self.tensor_image)
        self._plot_image(array_image=array_image, filename=filename)
