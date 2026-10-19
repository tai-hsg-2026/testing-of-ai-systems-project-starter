import torch
from torch import nn
from torch.utils.data import Dataset


def get_all_correctly_classified_images_of_a_class(
    model: nn.Module, dataset: Dataset, reference_class: int
) -> torch.Tensor:
    """
    Get all images of a specific class that are correctly classified by the model,
    also after the conversion png-svg-png.
    Args:
        model (nn.Module): Trained model.
        dataset (Dataset): Dataset of (image, label) pairs. Each image has shape (1, 28, 28).
        reference_class (int): Class label to filter images.
    Returns:
        torch.Tensor: Images with shape (N, 1, 1, 28, 28).
    """
    raise NotImplementedError()
