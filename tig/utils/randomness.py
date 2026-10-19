import random

import numpy as np
import torch
from numpy.random import RandomState


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators (Python, NumPy, PyTorch, and RandomGenerator).
    Each call resets the random generators. Thus, two calls with the same seed
    give the same random values, also in the same process.
    Args:
        seed (int): Random seed.
    Returns:
        None
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # get_instance uses the seed only when it creates the instance.
    # Thus, seed the random state of the instance also if it exists.
    RandomGenerator.get_instance(seed=seed).rnd_state.seed(seed)


class RandomGenerator:
    __instance: "RandomGenerator" = None

    @staticmethod
    def get_instance(
        seed: int = 0,
    ) -> "RandomGenerator":
        if RandomGenerator.__instance is None:
            RandomGenerator(seed=seed)
        return RandomGenerator.__instance

    def __init__(self, seed: int):
        if RandomGenerator.__instance is not None:
            raise RuntimeError("This class is a singleton!")

        self.rnd_state: RandomState = np.random.RandomState(seed=seed)
        RandomGenerator.__instance = self
