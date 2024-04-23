"""A pytorch-first transform library for ND data, such as multi-channel 3D volumes"""

import sys
from importlib import metadata as importlib_metadata

from .ops import Cast, ConvertDtype, To, ToDevice
from .rotation import RandomRotate, RandomRotate90
from .shape import CenterCrop, RandomCrop, RandomFlip, RandomPadding, RandomResize, Resize
from .structure import Compose, RandomApply
from .value import AdditiveBetaNoise, GaussianNoise, Normalize, SaltAndPepperNoise, UniformNoise

__all__ = sorted(
    [
        "RandomRotate90",
        "Compose",
        "RandomApply",
        "UniformNoise",
        "Normalize",
        "Cast",
        "ConvertDtype",
        "ToDevice",
        "To",
        "SaltAndPepperNoise",
        "AdditiveBetaNoise",
        "GaussianNoise",
        "CenterCrop",
        "RandomCrop",
        "RandomFlip",
        "RandomRotate",
        "Resize",
        "RandomResize",
        "RandomPadding",
    ]
)


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
