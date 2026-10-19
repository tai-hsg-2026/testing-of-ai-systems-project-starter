import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.classifier import Net, load_model
from networks.vae import ConvVAE, load_vae
from tig.representation.individual import Individual
from tig.utils.randomness import set_random_seed
from tig.utils.torch_utils import (
    array_image_to_tensor_image,
    get_test_dataset,
    tensor_image_to_array_image,
)


def get_image_of_a_class(digit: int, index: int) -> torch.Tensor:
    """
    Get an image of the MNIST test set with a specific label.
    Args:
        digit (int): Label of the image (from 0 to 9).
        index (int): Index of the image among the images with this label (0 is the first image).
    Returns:
        torch.Tensor: Image (normalized) with shape (1, 1, 28, 28).
    """
    # The classifier uses normalized images
    test_dataset = get_test_dataset(dataset_name="mnist", normalize=True)
    num_found = 0
    for tensor_image, label in test_dataset:
        if label != digit:
            continue
        if num_found == index:
            # Add the batch dimension: from (1, 28, 28) to (1, 1, 28, 28)
            return tensor_image.unsqueeze(0)
        num_found += 1
    raise ValueError(f"The test set has only {num_found} images with label {digit}")


def predict(model: Net, tensor_image: torch.Tensor) -> int:
    """
    Get the prediction of the classifier for one image.
    Args:
        model (Net): Trained classifier.
        tensor_image (torch.Tensor): Image (normalized) with shape (1, 1, 28, 28).
    Returns:
        int: Predicted label.
    """
    # eval mode
    model.eval()
    with torch.no_grad():
        # The output has shape (1, 10): one log-probability (log_softmax) for each class
        output = model(tensor_image)
    return int(output.argmax(dim=1).item())


def reconstruct(vae_model: ConvVAE, array_image: np.ndarray) -> np.ndarray:
    """
    Reconstruct one image with the VAE.
    The VAE samples the latent vector, thus the reconstruction is random.
    Args:
        vae_model (ConvVAE): Trained VAE.
        array_image (np.ndarray): Image with shape (28, 28) and pixel values in [0, 1].
    Returns:
        np.ndarray: Reconstructed image with shape (28, 28) and pixel values in [0, 1].
    """
    # eval mode
    vae_model.eval()
    device = next(vae_model.parameters()).device
    # The VAE uses images that are not normalized (pixel values in [0, 1])
    tensor_image = array_image_to_tensor_image(array_image=array_image, normalize=False)
    with torch.no_grad():
        # The VAE also returns the mean and the log variance of the latent distribution
        reconstructed_tensor_image, _, _ = vae_model(tensor_image.to(device))
    return reconstructed_tensor_image.cpu().numpy().squeeze()


def main():
    parser = argparse.ArgumentParser(
        description="Show the conversion png-svg-png, the prediction of the classifier, and the reconstruction of the VAE for one MNIST image"
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=5,
        help="Label of the MNIST test image (default: 5)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Index of the image among the test images with this label (default: 1)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "mnist.pt"),
        help="Path to the trained classifier (default: models/mnist.pt)",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=os.path.join("models", "vae_mnist.pt"),
        help="Path to the trained VAE (default: models/vae_mnist.pt)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed (default: 0). The VAE uses random values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "demo"),
        help="Directory of the output files (default: results/demo)",
    )
    args = parser.parse_args()

    digit = args.digit
    assert digit in range(10), "Digit must be between 0 and 9"
    index = args.index
    assert index >= 0, "Index must be greater than or equal to 0"
    assert os.path.exists(args.model_path), f"{args.model_path} does not exist"
    assert os.path.exists(args.vae_path), f"{args.vae_path} does not exist"

    set_random_seed(seed=args.random_seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = os.path.join(output_dir, f"digit_{digit}_index_{index}")

    # 1. Conversion png-svg-png
    tensor_image = get_image_of_a_class(digit=digit, index=index)
    # The constructor converts the image to an svg representation (png -> svg)
    individual = Individual(tensor_image=tensor_image)
    individual.plot_original_image(filename=f"{filename_prefix}_original")
    print(f"Original image saved in {filename_prefix}_original.png")
    individual.plot_svg_representation(filename=filename_prefix)
    print(f"Svg representation saved in {filename_prefix}.svg")
    # The svg representation is rasterized (svg -> png)
    individual.plot_current_image(filename=f"{filename_prefix}_rasterized")
    print(f"Rasterized image saved in {filename_prefix}_rasterized.png")
    # The array image has pixel values in [0, 1]. The tensor image is normalized.
    converted_array_image, converted_tensor_image = (
        individual.get_image_array_and_tensor_representation()
    )
    # The original image has pixel values in [0, 1] after this conversion
    original_array_image = tensor_image_to_array_image(tensor_image=tensor_image)

    # 2. Prediction of the classifier
    model = load_model(model_path=args.model_path)
    original_prediction = predict(model=model, tensor_image=tensor_image)
    converted_prediction = predict(model=model, tensor_image=converted_tensor_image)
    print(f"Label: {digit}")
    print(f"Prediction for the original image: {original_prediction}")
    print(f"Prediction for the converted image: {converted_prediction}")

    # 3. Reconstruction of the VAE
    vae_model = load_vae(model_path=args.vae_path)
    original_reconstruction = reconstruct(
        vae_model=vae_model, array_image=original_array_image
    )
    converted_reconstruction = reconstruct(
        vae_model=vae_model, array_image=converted_array_image
    )

    _, axes = plt.subplots(2, 2, figsize=(5, 5))
    plots = [
        (original_array_image, "Original png"),
        (original_reconstruction, "VAE reconstruction"),
        (converted_array_image, "Rasterized png"),
        (converted_reconstruction, "VAE reconstruction"),
    ]
    for axis, (array_image, title) in zip(axes.flatten(), plots):
        axis.imshow(array_image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    plt.suptitle(f"VAE reconstruction: test image with label {digit} (index {index})")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_vae.png", dpi=150)
    plt.close()
    print(f"VAE reconstructions saved in {filename_prefix}_vae.png")


if __name__ == "__main__":
    main()
