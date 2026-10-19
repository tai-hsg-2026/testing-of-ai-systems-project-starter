
import os

import torch
import torch.nn.functional as F
from torch import nn


def vae_loss(
    tensor_image: torch.Tensor,
    reconstructed_tensor_image: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss as the sum of reconstruction loss and KL divergence.
    The reconstruction loss is the binary cross entropy. Thus, the pixel values
    of the images must be in [0, 1] (i.e., the images are not normalized).
    Args:
        tensor_image: Original input tensor image.
        reconstructed_tensor_image: Reconstructed tensor image from the VAE.
        mu: Mean from the encoder's latent space.
        logvar: Log variance from the encoder's latent space.
    Returns:
        total_loss: Total VAE loss.
    """
    recon = F.binary_cross_entropy(
        input=reconstructed_tensor_image, target=tensor_image, reduction="sum"
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon + kl
    return total_loss, recon, kl


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14 → 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14 → 28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    Args:
        mu: Mean from the encoder's latent space.
        log_var: Log variance from the encoder's latent space.
    Returns:
        Sampled latent vector.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def load_vae(model_path: str, latent_dim: int = 20) -> ConvVAE:
    """
    Load a trained VAE.
    Args:
        model_path (str): Path of the trained VAE (e.g., models/vae_mnist.pt).
        latent_dim (int): Latent dimension of the trained VAE.
    Returns:
        ConvVAE: Trained VAE, on the accelerator if it is available.
    """
    assert os.path.exists(model_path), f"{model_path} does not exist"
    device = (
        torch.accelerator.current_accelerator()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    vae_model = ConvVAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    return vae_model
