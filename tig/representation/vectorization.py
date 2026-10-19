import io
import xml.etree.ElementTree as ET

import numpy as np
import potrace
from svgelements import SVG
from torch import Tensor

from tig.utils.torch_utils import tensor_image_to_array_image


def _preprocess(image: np.ndarray) -> np.ndarray:
    bw = np.asarray(image).copy()
    bw[bw < 0.5] = 0  # Black
    bw[bw >= 0.5] = 1  # White
    return bw


def _createSVGpath(path) -> str:
    path_desc = ""
    # Iterate over path curves
    for curve in path:
        path_desc = (
            path_desc
            + " M "
            + str(curve.start_point[0])
            + ","
            + str(curve.start_point[1])
        )
        for segment in curve:
            if segment.is_corner:
                path_desc = (
                    path_desc
                    + " L "
                    + str(segment.c[0])
                    + ","
                    + str(segment.c[1])
                    + " L "
                    + str(segment.end_point[0])
                    + ","
                    + str(segment.end_point[1])
                )
            else:
                path_desc = (
                    path_desc
                    + " C "
                    + str(segment.c1[0])
                    + ","
                    + str(segment.c1[1])
                    + " "
                    + str(segment.c2[0])
                    + ","
                    + str(segment.c2[1])
                    + " "
                    + str(segment.end_point[0])
                    + ","
                    + str(segment.end_point[1])
                )
    return path_desc + " Z"


def _create_svg_xml(desc: str) -> str:
    root = ET.Element("svg")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(28))
    root.set("width", str(28))
    path = ET.SubElement(root, "path")
    path.set("d", desc)
    tree = ET.ElementTree(root)
    tree = tree.getroot()
    xml_str = ET.tostring(tree, encoding="unicode", method="xml")
    return xml_str


def vectorize(tensor_image: Tensor) -> str:

    array_image = tensor_image_to_array_image(tensor_image=tensor_image)

    array = _preprocess(image=array_image)
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    desc = _createSVGpath(path)

    svg_str = _create_svg_xml(desc)
    svg_object = SVG.parse(io.StringIO(svg_str))
    return svg_object
