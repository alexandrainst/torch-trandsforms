"""Transforms for shape and size"""

import torch
import torchvision

from . import _functional as F
from ._utils import get_tensor_sequence
from .base import KeyedNdTransform


class Crop(KeyedNdTransform):
    """
    Base class for cropping. Raises NotImplementedError when called.

    Args:
        size (int or array-like): int | long (sequence) of crop size (if sequence, must be length nd)
        padding (str, float, or array-like): Padding argument. See `torch_trandsforms._functional.pad` for information on the argument
    """

    def __init__(self, size, padding=None, p=1, nd=3, keys="*"):
        super().__init__(p, nd, keys)
        self.size = get_tensor_sequence(size, nd, acceptable_types=(torch.int, torch.long))
        self.padding = padding


class CenterCrop(Crop):
    """
    Crops the input at the center. Returns a slice of the named inputs of the given size.
    If padding is None, and the input is smaller than the crop size, raises an error.
    Inputs are expected to be the same shape at the trailing `nd` dimensions

    Args:
        size (int or array-like): int | long (sequence) of crop size (if sequence, must be length nd)
        padding (str, float, or array-like): Padding argument. See `torch_trandsforms._functional.pad` for information on the argument
    """

    def __init__(self, size, padding=None, p=1, nd=3, keys="*"):
        super().__init__(size, padding, p, nd, keys)

    def get_parameters(self, **inputs):
        shapes = [i.shape[-self.nd :] for i in inputs.values()]
        assert all([shapes[0] == shape for shape in shapes[1:]]), "All input shapes at the trailing `nd` dimensions must be the same"

        shape_t = torch.tensor(shapes[0])
        pos = (shape_t - self.size) // 2

        return {"pos": pos}

    def apply(self, input, **params):
        pos = params["pos"]
        return F.crop(input, pos, self.size, padding=self.padding)


class RandomCrop(Crop):
    """
    Randomly extracts a crop from the named inputs. Returns a slice of the named inputs of the given size.
    If padding is None, and the input is smaller than the crop size, raises an error.
    Inputs are expected to be the same shape at the trailing `nd` dimensions

    Args:
        size (int or array-like): int | long (sequence) of crop size (if sequence, must be length nd)
        padding (str, float, or array-like): Padding argument. See `torch_trandsforms._functional.pad` for information on the argument
    """

    def __init__(self, size, padding=None, p=1, nd=3, keys="*"):
        super().__init__(size, padding, p, nd, keys)

    def get_parameters(self, **inputs):
        shapes = [i.shape[-self.nd :] for i in inputs.values()]
        assert all([shapes[0] == shape for shape in shapes[1:]]), "All input shapes at the trailing `nd` dimensions must be the same"

        shape_t = torch.tensor(shapes[0])
        max_pos = shape_t - self.size

        pos = tuple(torch.randint(min(p // 2, 0), max(p // 2, p) + 1, size=(1,)).item() for p in max_pos)

        return {"pos": pos}

    def apply(self, input, **params):
        pos = params["pos"]
        return F.crop(input, pos, self.size, padding=self.padding)
