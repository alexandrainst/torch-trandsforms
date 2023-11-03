"""Structural classes like Compose, Random Apply, etc"""

import torch

__all__ = ["Compose"]


class Compose:
    """
    Composes several transforms together.
    Please note that this class only automates the running of a transform pipeline.
        The dimensionality and operating keys is left to the user, as they are rarely
        assumed to be equal for all transform objects.

    Args:
        transforms (list of `transform` objects): list of transforms to compose.

    Example:
        >>> torch_trandsforms.Compose([
        >>>    torch_trandsforms.RandomRotate90(p=0.66, nd=3),
        >>>    torch_trandsforms.UniformNoise(p=1., low=0, hi=0.1, keys=["data", "extra_data"])
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **inputs):
        for transform in self.transforms:
            inputs = transform(**inputs)
        return inputs

    def __repr__(self) -> str:
        """stolen from torchvision"""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            try:
                format_string += f"    {t}"
            except AttributeError:
                format_string += f"    {t.__class__.__name__}"
        format_string += "\n)"
        return format_string
