"""Transforms for shape and size"""

from typing import Any, List, Optional

import numpy
import torch
import torchvision
from torch.nn import functional as F_t

from . import _functional as F
from ._utils import extract_min_max, get_tensor_sequence
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


class RandomFlip(KeyedNdTransform):
    """
    Randomly flips a dimension (like Horizontal- and VerticalFlip)
    Applies the same flip to all inputs found in keys
    """

    def __init__(self, p=0.5, nd=3, keys="*"):
        super().__init__(p, nd, keys)

    def get_parameters(self, **inputs):
        return {"dim": torch.randint(0, self.nd, size=(1,)).item()}

    def apply(self, input, **params):
        if input.ndim < self.nd:
            raise ValueError(f"Input dimensionality {input.ndim} must be greater than or equal to self.nd {self.nd}")

        return input.flip(params["dim"])


class Resize(KeyedNdTransform):
    """
    Apply a given scale factor or size, one of which must be provided
    Scales all named inputs the same way

    Args:
        scale_factor (Optional[sequence or number]): Optional argument to scale by a given factor regardless of input size
        size (Optional[sequence or number]): Optional argument to always scale to a given size
        interpolation_mode (str): algorithm used for upsampling. For nd <= 3, current available modes are: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        align_corners (bool): see torch.nn.functional.interpolate for an explanation
    """

    def __init__(self, scale_factor=None, size=None, interpolation_mode="area", align_corners=None, p=1, nd=3, keys="*"):
        super().__init__(p, nd, keys)
        if self.nd > 3:
            raise NotImplementedError("Scaling for nd > 3 is not yet implemented")

        if (scale_factor is None and size is None) or (scale_factor is not None and size is not None):
            raise ValueError("Expected one of scale_factor or size")

        self.scale_factor = get_tensor_sequence(scale_factor, nd) if scale_factor else None
        self.size = get_tensor_sequence(size, nd, (torch.int, torch.long)) if size else None

        self.align_corners = align_corners
        self.interpolation_mode = interpolation_mode

    def apply(self, input, **params):
        # determine size based on original args
        if self.size is None:
            size = tuple(int(d * f) for d, f in zip(input.shape[-self.nd :], self.scale_factor))
        else:
            size = tuple(self.size.tolist())

        osh = (*input.shape[: -self.nd], *size)

        # ensure input dimensionality matches torch expectations
        if input.ndim < self.nd + 2:
            input = input.view(*[1] * (self.nd + 2 - input.ndim), *input.shape)

        return F_t.interpolate(input, size, mode=self.interpolation_mode, align_corners=self.align_corners).view(*osh)


class RandomResize(KeyedNdTransform):
    """
    Apply a random scale factor or size, one of which must be provided, to named inputs
    Scales all named inputs the same way

    Args:
        scale_factor (Optional[sequence or number]): Optional argument to scale by a random factor regardless of input size. Expects, or attempts to construct, a sequence of shape (nd,2) to represent minimum and maximum scaling factors
        size (Optional[sequence or number]): Optional argument to randomly scale to a given size. Expects, or attempts to construct, a sequence of shape (nd,2) to represent minimum and maximum sizes.
        interpolation_mode (str): algorithm used for upsampling. For nd <= 3, current available modes are: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        align_corners (bool): see torch.nn.functional.interpolate for an explanation
    """

    def __init__(self, scale_factor=None, size=None, interpolation_mode="area", align_corners=None, p=1, nd=3, keys="*"):
        super().__init__(p, nd, keys)

        self.size: Optional[List[Any]] = None
        self.scale_factor: Optional[List[Any]] = None

        if self.nd > 3:
            raise NotImplementedError("Scaling for nd > 3 is not yet implemented")

        if (scale_factor is None and size is None) or (scale_factor is not None and size is not None):
            raise ValueError("Expected one of scale_factor or size")

        # if scale_factor, ensure potential scalar values use (1-v,1+v) and that sequences are length 2 with min <= max
        if scale_factor is not None:
            if isinstance(scale_factor, (torch.Tensor, numpy.ndarray)):
                scale_factor = scale_factor.tolist()
            if isinstance(scale_factor, (float, int)):
                self.scale_factor = [extract_min_max(scale_factor, base=1.0) for i in range(self.nd)]
            elif isinstance(scale_factor, (tuple, list)):
                if len(scale_factor) == 2 and (isinstance(scale_factor[0], float) or isinstance(scale_factor[1], float)):
                    self.scale_factor = [scale_factor for i in range(self.nd)]
                elif len(scale_factor) != self.nd:
                    raise ValueError(
                        f"Expected a list of len nd ({nd}), not {len(scale_factor)}. Did you accidentally provide a flattened list?"
                    )
                else:
                    self.scale_factor = [extract_min_max(s, base=1.0) for s in scale_factor]
            else:
                raise TypeError(f"Did not understand scale_factor type {type(scale_factor)}")

        if size is not None:
            if isinstance(size, (torch.Tensor, numpy.ndarray)):
                size = size.tolist()
            if isinstance(size, (tuple, list)):
                if len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int):
                    self.size = [size for i in range(self.nd)]
                elif len(size) != self.nd:
                    raise ValueError(
                        f"Expected a list of len nd ({self.nd}), not {len(size)}. Did you accidentally provide a flattened list?"
                    )
                else:
                    self.size = [extract_min_max(s, allow_value=False) for s in size]
                    self.size = [(int(a), int(b)) for a, b in self.size]
            else:
                raise TypeError(f"Did not understand size type {type(size)}. Did you mean to write (size,size)?")

        self.align_corners = align_corners
        self.interpolation_mode = interpolation_mode

    def get_parameters(self, **inputs):
        if self.size:
            size = [torch.randint(s0, s1 + 1, size=(1,)).item() for s0, s1 in self.size]
            return {"size": size}
        elif self.scale_factor:
            scale_factor = [(s0 - s1) * torch.rand(1).item() + s1 for s0, s1 in self.scale_factor]
            return {"scale_factor": scale_factor}
        raise RuntimeError("Somehow found neither self.size nor self.scale_factor")

    def apply(self, input, **params):
        sh = input.shape

        # ensure input dimensionality matches torch expectations
        if input.ndim < self.nd + 2:
            input = input.view(*[1] * (self.nd + 2 - input.ndim), *input.shape)

        if "size" in params:
            scaled = F_t.interpolate(
                input, size=params["size"], scale_factor=None, mode=self.interpolation_mode, align_corners=self.align_corners
            )
            return scaled.view(*sh[: -self.nd], *scaled.shape[-self.nd :])
        scaled = F_t.interpolate(
            input, size=None, scale_factor=params["scale_factor"], mode=self.interpolation_mode, align_corners=self.align_corners
        )
        return scaled.view(*sh[: -self.nd], *scaled.shape[-self.nd :])
