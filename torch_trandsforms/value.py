"""
Contains value-level transformations, normally noise intended for specific keys
"""

import torch

from .base import KeyedNdTransform, KeyedTransform

__all__ = ["Normalize", "UniformNoise"]


class Normalize(KeyedNdTransform):
    """
    Normalize the named input tensors at the -N-1th dimension
    The dimensionality must be specified (to avoid collisions within dimension sizes)
    Can accept single values for mean and std for channel-less data
    Also accepts ND-normalization parameters (i.e. normalization in multiple dimensions at once)
        This may be useful for data from different sets, timeseries, or similar

    Args:
        mean: mean value of the set (as number, kd-iterable, or torch.tensor)
        std: standard deviation of the set (as number, kd-iterable, or torch.tensor)

    Example:
        >>> \"\"\"
        >>> this tensor is 4D, but in our case it is a 3D-volume with 2 value-channels
        >>> consider it a box, where each corner has a measurement of foo and a measurement of bar
        >>> the dimensions have the following notation, then: CxDxHxW, where C is our target dimension
        >>> in fact, the tensor could have any number of leading dimensions like so: Bx...xCxDxHxW
        >>> \"\"\"
        >>> tensor = torch.arange(16, dtype=torch.float, device='cuda').view(2,2,2,2)
        >>> norm = Normalize(mean=[3.5, 11.5], std=4.761, p=1., nd=3).to("cuda")  # note nd = 3 (and that mean and std accept single values)
        >>> norm(tensor=tensor)["tensor"]
        >>> tensor([[[[-0.0725,  0.1376],
        >>>  [ 0.3476,  0.5577]],
        >>>
        >>>  [[ 0.7677,  0.9777],
        >>>   [ 1.1878,  1.3978]]],
        >>>
        >>>
        >>> [[[ 1.5180,  1.7280],
        >>>   [ 1.9381,  2.1481]],
        >>>
        >>>  [[ 2.3581,  2.5682],
        >>>   [ 2.7782,  2.9883]]]], device='cuda:0')
    """

    def __init__(self, mean, std, p=1.0, keys="*", nd=3):
        super().__init__(p=p, keys=keys, nd=nd)
        assert isinstance(
            mean, (int, float, tuple, list, torch.Tensor)
        ), f"Mean must be a real number, iterable, or torch.tensor, got {type(mean)}"
        assert isinstance(
            std, (int, float, tuple, list, torch.Tensor)
        ), f"Std must be a real number, iterable, or torch.tensor, got {type(std)}"

        self.mean = self._get_tensor(mean)
        self.std = self._get_tensor(std)

        if torch.any(self.std == 0.0):
            raise ValueError(f"Value in std is 0, this will produce a division by zero error ({std})")

    def _get_tensor(self, value):
        no_channel = False
        if isinstance(value, (int, float)):
            value = [value]
            no_channel = True
        if isinstance(value, (tuple, list)):
            value = torch.tensor(value)
        if not torch.is_floating_point(value):
            value = value.float()

        value = value.view((*value.shape, *[1] * (self.nd - no_channel)))

        return value

    def apply(self, input, **params):
        mean = torch.broadcast_to(self.mean, input.shape)
        std = torch.broadcast_to(self.std, input.shape)

        input = (input - mean) / std

        return input


class UniformNoise(KeyedTransform):
    """
    The simplest form of noise, Uniform Noise adds noise to every value of the input
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
