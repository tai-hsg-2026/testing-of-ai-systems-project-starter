import io

import cairosvg
import numpy as np
from PIL import Image
from torch import Tensor

from tig.utils.torch_utils import array_image_to_tensor_image


def rasterize(svg_str: str) -> tuple[np.ndarray, Tensor]:
    png_bytes = cairosvg.svg2png(bytestring=svg_str)
    image = Image.open(io.BytesIO(png_bytes))

    array_image = np.array(image, dtype=np.float32)
    # take only the last channel (the conversion with PIL results in an array with 3 channels)
    array_image = array_image[:, :, 3]
    # normalize
    array_image /= 255.0

    tensor_image = array_image_to_tensor_image(array_image=array_image)

    return array_image, tensor_image
