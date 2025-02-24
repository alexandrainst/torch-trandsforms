"""
Contains rotation transforms
"""

import numpy
import torch

from ._functional import rotate
from ._utils import get_tensor_sequence
from .base import KeyedNdTransform

__all__ = ["RandomRotate90"]


class RandomRotate90(KeyedNdTransform):  # note the use of NdTransform as base class
    """
    Rotates the input 90 degrees around a randomly determined axis
    """

    def __init__(self, nd=3, p=0.5, keys="*"):
        super().__init__(p=p, nd=nd, keys=keys)
        self.options = self._get_options(nd)

    def _get_options(self, nd):
        """
        Create potential rotations

        Args:
            nd (int): Number of trailing dimensions to potentially rotate

        Returns:
            list: All possible combinations of rotation dimensions
        """
        options = []

        for i in range(nd):
            for j in range(nd):
                if not i == j:
                    options.append((-i - 1, -j - 1))

        return options

    def get_parameters(self, **inputs):
        """
        Randomly selects a rotation option

        Args:
            inputs (any): Not used

        Returns:
            dict: 'rot':rotation tuple
        """

        randint = torch.randint(len(self.options), (1,)).item()
        rotation = self.options[randint]
        return {"rot": rotation}

    def apply_transform(self, input, **params):
        """
        Applies the rotation combination

        Args:
            input (torch.Tensor): Tensor to rotate 90 degrees
            **params (dict): Contains rotation dimensions (tuple) under 'rot' key

        Returns:
            torch.Tensor: Rotated tensor
        """
        rot = params["rot"]
        return torch.rot90(input, dims=rot)


class RandomRotate(KeyedNdTransform):
    """
    Applies a random rotation in the trailing `nd` axes
    Currently only implemented for `nd` <= 3
    """

    def __init__(self, rotation, sample_mode="bilinear", padding_mode="zeros", align_corners=None, p=0.5, nd=3, keys="*"):
        super().__init__(p, nd, keys)
        if self.nd > 3:
            raise NotImplementedError(f"Arbitrary rotation is only implemented for nd <= 3, got {nd}")
        if self.nd == 1:
            raise NotImplementedError("Arbitrary rotation in 1D is not available")

        # attempt to extract -rot,+rot from rotation according to input type
        # raise TypeError or ValueError depending on typing and input values
        if isinstance(rotation, (float, int)):
            self.rotation = [get_tensor_sequence(float(rotation), 2, torch.float)]
        elif isinstance(rotation, (torch.Tensor, numpy.ndarray)):
            if isinstance(rotation, numpy.ndarray):
                rotation = torch.from_numpy(rotation)
            if rotation.ndim == 0:
                self.rotation = [get_tensor_sequence(rotation.float(), 2, torch.float)]
            else:
                self.rotation = [get_tensor_sequence(rot.float(), 2, torch.float) for rot in rotation]
        elif isinstance(rotation, (tuple, list)):
            self.rotation = [get_tensor_sequence(float(rot), 2, torch.float) for rot in rotation]
        else:
            raise TypeError(f"Did not understand typing of rotation, got {type(rotation)}")

        if len(self.rotation) != 3 and len(self.rotation) != 1:
            raise ValueError(f"Expected rotation to have 1 or 3 len, got {len(self.rotation)}")

        if self.nd == 2 and not len(self.rotation) == 1:
            raise ValueError(f"Expected 1 rotational axis for nd = 2, got {len(self.rotation)}")
        elif self.nd == 3 and not len(self.rotation) == 3:
            raise ValueError(f"Expected 3 rotational axis for nd = 3, got {len(self.rotation)}")

        # set min rot to -rot
        for idx in range(len(self.rotation)):
            self.rotation[idx][0] = -self.rotation[idx][1]

        # set parameters for rotation
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def get_parameters(self, **inputs):
        """
        Generate a random rotation in the uniform distribution -rot,rot for rot in each rotation dimension
        """
        rotation = [(rot[0] - rot[1]) * torch.rand((1,)).item() + rot[1] for rot in self.rotation]
        return {"rot": rotation}

    def apply_transform(self, input, **params):
        osh = input.shape
        # check that input dimensionality is appropriate for grid_sample or make it so
        if input.ndim < self.nd + 2:
            input = input.view(*[1] * (self.nd + 2 - input.ndim), *input.shape)

        return rotate(
            input,
            params["rot"],
            input.size(),
            sample_mode=self.sample_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).view(*osh)
