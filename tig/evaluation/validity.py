import numpy as np
import torch

from networks.vae import ConvVAE
from tig.representation.individual import Individual


class Validator:
    def __init__(self, vae_model: ConvVAE, reconstruction_threshold: float) -> None:
        """
        Validity oracle with a VAE.
        An individual is valid if its reconstruction error is less than a threshold.
        Args:
            vae_model (ConvVAE): Trained VAE.
            reconstruction_threshold (float): Threshold on the reconstruction error.
        """
        assert (
            reconstruction_threshold > 0
        ), f"Reconstruction threshold must be greater than 0, found {reconstruction_threshold}"
        self.vae_model = vae_model
        self.vae_model.eval()
        self.reconstruction_threshold = reconstruction_threshold
        self.device = next(vae_model.parameters()).device

    def get_reconstruction_error(self, individual: Individual) -> float:
        """
        Get the reconstruction error of the current image (svg representation) of an individual.
        Args:
            individual (Individual): Individual to evaluate.
        Returns:
            float: Reconstruction error of the image of the individual.
        """
        raise NotImplementedError()

    def is_valid(self, individual: Individual) -> bool:
        """
        Check if an individual is valid.
        Args:
            individual (Individual): Individual to evaluate.
        Returns:
            bool: True if the reconstruction error is less than the threshold.
        """
        raise NotImplementedError()


def compute_reconstruction_errors(
    vae_model: ConvVAE, tensor_images: torch.Tensor, batch_size: int = 1024
) -> np.ndarray:
    """
    Compute the reconstruction error of each image with a VAE.
    Args:
        vae_model (ConvVAE): Trained VAE.
        tensor_images (torch.Tensor): Images with shape (N, 1, 28, 28) and pixel values
            in [0, 1] (i.e., not normalized).
        batch_size (int): Number of images that the VAE processes at the same time.
    Returns:
        np.ndarray: Reconstruction error of each image, with shape (N,).
    """
    raise NotImplementedError()


def get_validator(
    vae_model: ConvVAE, tensor_images: torch.Tensor, validity_confidence: float
) -> Validator:
    """
    Compute the threshold on the reconstruction error and make the validity oracle.
    The threshold comes from a gamma distribution fitted to the reconstruction
    errors of the images.
    Args:
        vae_model (ConvVAE): Trained VAE.
        tensor_images (torch.Tensor): Images that set the threshold, with shape (N, 1, 28, 28)
            and pixel values in [0, 1].
        validity_confidence (float): Confidence that sets the threshold.
    Returns:
        Validator: Validity oracle.
    """
    raise NotImplementedError()

