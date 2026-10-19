# 1. Setup

## 1.1 Install Docker

Install [Docker](https://docs.docker.com/engine/install/) for your system.

## 1.2 Get the image

The image `dockercontainervm/model-based-tig:2026` is on Docker Hub.
It is available for `linux/amd64` (Intel/AMD) and for `linux/arm64` (e.g., Apple Silicon).
The image contains all the dependencies of the project; do not install any new dependency on it.
The Devcontainer extension pulls the image automatically (see section 1.3).
To pull the image manually, use this command:

```bash
docker pull dockercontainervm/model-based-tig:2026
```

## 1.3 Use VSCode Devcontainer

1. Download [VSCode](https://code.visualstudio.com/Download) for your platform.
2. Install the Dev Containers extension.
3. Open the Command Palette in VSCode (`Ctrl+Shift+P`, or `Cmd+Shift+P` on macOS).
4. Run the command "Dev Containers: Open Folder in Container...".
5. Select the project folder.

The extension uses the `dockercontainervm/model-based-tig:2026` image to make a container, and you will develop inside it.
The container runs as the non-root user `tai_user`.
The Python interpreter is `/opt/venv/bin/python`, which VSCode selects by default.

Run all the commands of this README in a terminal of the container, from the project folder.

# 2. Trained models

The project contains two trained models:

- `models/mnist.pt`: the MNIST classifier (the model under test). Its accuracy on the test set is 99%.
- `models/vae_mnist.pt`: a Variational Autoencoder (VAE) trained on MNIST. Use it for the validity evaluation.

The code that defines the models is in the directory `networks`:

- `networks/classifier.py` defines the classifier (`Net`).
- `networks/vae.py` defines the VAE (`ConvVAE`) and the function `load_vae`.

The classifier uses normalized images (mean `0.1307`, standard deviation `0.3081`).
The VAE uses images with pixel values in `[0, 1]` (not normalized).
The functions in `tig/utils/torch_utils.py` convert the images between the two formats.

# 3. Run the demo

*Replace this part with instructions on how to run the code. Structure the sections according to the macro-requirements.*

The script `main_demo.py` shows some examples on how to use the code.
It does these steps for one image of the MNIST test set:

1. It converts the image to an svg representation (png-svg), and then converts the svg representation to an image again (svg-png).
2. It gets the prediction of the classifier `models/mnist.pt` for the original image and for the converted image.
3. It uses the VAE `models/vae_mnist.pt` to reconstruct the original image and the converted image.

Run this command:

```bash
python main_demo.py
```

The command uses the second test image with label `5` (index `1`).
If the MNIST dataset is not in the directory `data`, the command downloads it.
The command shows the label and the two predictions.
It saves these files in the directory `results/demo`:

- `digit_5_index_1_original.png`: the original image.
- `digit_5_index_1.svg`: the svg representation of the image (png-svg).
- `digit_5_index_1_rasterized.png`: the rasterized svg representation, i.e., the converted image (svg-png).
- `digit_5_index_1_vae.png`: four images. 
  The top row shows the original image and its VAE reconstruction. 
  The bottom row shows the converted image and its VAE reconstruction.

To select a different image, use `--digit` and `--index`.
For example, this command uses the fourth test image with label `7`:

```bash
python main_demo.py --digit 7 --index 3
```

The VAE samples the latent vector. 
Thus, the reconstruction is random. 
The command sets the random seed to `0`. 
Thus, two executions with the same arguments give the same reconstruction. 
To change the random seed, use `--random-seed`.
To change the directory of the files, use `--output-dir`.
To see all the options, use `python main_demo.py --help`.

# 4. Project structure

The code of the project is in the directory `tig`.
Each subdirectory agrees with a part of the requirements:

| Directory | Contents |
|---|---|
| `tig/representation` | Model-based representation of an input: conversion png-svg-png and mutation (`Individual`). |
| `tig/seeds` | Seed pool: inputs of the selected class that the model classifies correctly, also after the conversion. |
| `tig/generation` | Fitness functions and image generators (random and hill climbing). |
| `tig/evaluation` | Diversity (feature map) and validity (VAE) of the misclassifications. |
| `tig/utils` | Conversion of tensors and arrays, datasets, and random generators. |

The functions and the methods that you must implement raise `NotImplementedError`.
Their docstrings tell what they should do.
Do not change the names, the parameters, nor the content of the functions, methods, and classes that exist.
You can add new functions, methods, classes, and files.

A "seed" is an input image that a generator starts from.
A "random seed" is the value that initializes the random generators.
