"""
Contains value-level transformations, normally noise intended for specific keys
"""

from numbers import Number

import torch

from .base import KeyedTransform


class UniformNoise(KeyedTransform):
    """
    The simplest form of noise, Uniform Noise adds noise to every value of the input input
        based on a uniform distribution of [low, hi)

    Args:
        low: minimum value to add
        hi: maximum value to add
    """

    def __init__(self, low=-1, hi=1, p=0.5, keys=[]):
        super().__init__(p=p, keys=keys)
        if not isinstance(low, (float, int)):
            raise ValueError("low must be a real number")
        if not isinstance(hi, (float, int)):
            raise ValueError("hi must be a real number")

        self.low = low
        self.hi = hi

    def apply(self, input, **params):
        """
        Applies a randomly generated uniform noise to the input (if input is in keys)

        Args:
            input (torch.Tensor): input tensor to operate on and generate noise tensor based off of
            **params (any): not used

        Returns:
            torch.Tensor: tensor with added uniform noise
        """
        dtype = input.dtype if torch.is_floating_point(input) else torch.float
        noise = (self.low - self.hi) * torch.rand_like(input, device=input.device, dtype=dtype) + self.hi
        return input + noise
