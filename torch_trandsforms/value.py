"""
Contains value-level transformations, normally noise intended for specific keys
"""

import warnings

import numpy
import torch

from .base import KeyedNdTransform, KeyedTransform

__all__ = ["Normalize", "UniformNoise", "SaltAndPepperNoise", "AdditiveBetaNoise", "GaussianNoise"]


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

    def __init__(self, low=-1, hi=1, p=0.5, keys="*"):
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


class SaltAndPepperNoise(KeyedNdTransform):
    """
    Introduces salt-and-pepper noise in the tensor
    Uses a beta distribution with expected alpha and beta < 1 to produce heavy tailed noise
    Makes no attempts to correlate inputs (i.e. no shared parameters)

    Args:
        prob (float): 0-1 range of indices to overwrite (1 meaning every index will be noise)
        low (float): minimum value to replace with
        hi (float): maximum value to replace with
        a (float or torch.Tensor): alpha value for the beta distribution (likely < 1). Can generate on any device using a=torch.tensor(a, device=device)
        b (float or torch.Tensor): beta value for the beta distribution (likely < 1). Can generate on any device using b=torch.tensor(b, device=device)

    Example:
        >>> tensor = torch.arange(16).view(2,2,2,2)  # CxDxHxW
        >>> noiser = SaltAndPepperNoise(prob=0.1, nd=3)  # generates 3D probs, overwriting an entire voxel with 1 value
        >>> noiser = SaltAndPepperNoise(prob=0.1, nd=4)  # generates 4D probs, overwriting individual values
        >>>
        >>> image = torch.rand(3, 224, 224)  # image of size 224x224 (CxHxW)
        >>> noiser = SaltAndPepperNoise(prob=0.1, nd=2)  # generates probabilities on a pixel-level (i.e. greyscale noise)
        >>> noiser = SaltAndPepperNoise(prob=0.1, nd=3)  # generates probabilities on a color-level (i.e. R/G/B noise)
    """

    def __init__(self, prob, low=-1, hi=1, a=0.5, b=0.5, p=0.5, nd=3, keys="*"):
        super().__init__(p, nd, keys)
        assert isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0, f"prob must be a number between 0 and 1 (got {prob})"
        self.prob = prob

        assert isinstance(low, (int, float, torch.FloatTensor)), f"low must be a number (got {prob})"
        assert isinstance(hi, (int, float, torch.FloatTensor)) and low < hi, f"hi must be a number greater than low"
        self.low = low
        self.hi = hi

        assert isinstance(a, (float, torch.Tensor)) and 0 < a, f"a must be a float or tensor greater than 0"
        assert isinstance(b, (float, torch.Tensor)) and 0 < b, f"b must be a float or tensor greater than 0"
        self.dist = torch.distributions.beta.Beta(a, b)

    def apply(self, input, **params):
        probs = torch.broadcast_to(torch.rand(*input.shape[-self.nd :], device=self.dist.concentration1.device) < self.prob, input.shape)
        values = torch.broadcast_to((self.low - self.hi) * self.dist.sample(input.shape[-self.nd :]) + self.hi, input.shape)
        input[probs] = values[probs]
        return input


class AdditiveBetaNoise(SaltAndPepperNoise):
    """
    Adds noise sampled from a beta distribution to the input
    Makes no attempts to correlate inputs (i.e. no shared parameters)

    Args:
        prob (float): 0-1 range of indices to add (1 meaning every index will have added noise)
        low (float): minimum value to add with
        hi (float): maximum value to add with
        a (float or torch.Tensor): alpha value for the beta distribution (likely < 1). Can generate on any device using a=torch.tensor(a, device=device)
        b (float or torch.Tensor): beta value for the beta distribution (likely < 1). Can generate on any device using b=torch.tensor(b, device=device)

    Example:
        >>> tensor = torch.arange(16).view(2,2,2,2)  # CxDxHxW
        >>> noiser = AdditiveBetaNoise(prob=0.1, nd=3)  # generates 3D probs, adding 1 value to an entire voxel
        >>> noiser = AdditiveBetaNoise(prob=0.1, nd=4)  # generates 4D probs, adding individual values to each channel in a voxel (when sample_p < prob)
        >>>
        >>> image = torch.rand(3, 224, 224)  # image of size 224x224 (CxHxW)
        >>> noiser = AdditiveBetaNoise(prob=0.1, nd=2)  # generates probabilities on a pixel-level (i.e. additive greyscale noise)
        >>> noiser = AdditiveBetaNoise(prob=0.1, nd=3)  # generates probabilities on a color-level (i.e. additive R/G/B noise)
    """

    def apply(self, input, **params):
        probs = torch.broadcast_to(torch.rand(*input.shape[-self.nd :], device=self.dist.concentration1.device) < self.prob, input.shape)
        values = torch.broadcast_to((self.low - self.hi) * self.dist.sample(input.shape[-self.nd :]) + self.hi, input.shape)
        input[probs] += values[probs]
        return input


class GaussianNoise(KeyedNdTransform):
    """
    Adds gaussian sampled noise to the named inputs
    Matches the mean (or std) input to the shape of the input tensor (broadcasting),
        so any iterable must match the size of the N+1th dimension of the input tensor (likely the channel dimension)
    Accepts ND mean and std inputs for multi-dimension sampling (finetuned generation)

    Args:
        mean (float, list, or torch.Tensor): The mean of the gaussian distribution. Can generate on any device using mean=torch.tensor(mean, device=device)
        std (float, list, or torch.Tensor): The standard deviation of the gaussian.

    Example:
        >>> tensor = torch.arange(16).view(2,2,2,2)  # a box where each corner has two channel values: foo and bar (C*D*H*W)
        >>> std = 1.
        >>> mean = torch.tensor([0.5, 0.75], device="cuda:0")
        >>> noiser = GaussianNoise(mean=mean, std=std, nd=3)  # valid, generates values with different means per channel on GPU
        >>>
        >>> std = [5,1]
        >>> noiser = GaussianNoise(mean=mean, std=std, nd=3)  # also valid, produces values with different means and stds per channel
        >>>
        >>> mean = [1,2,3]
        >>> noiser = GaussianNoise(mean=mean, std=std, nd=3)  # RuntimeError, mean's C-size does not match the input tensor
        >>>
        >>> mean = torch.arange(4, dtype=torch.float, device="cuda:0").view(2,2)
        >>> noiser = GaussianNoise(mean=mean, std=std, nd=2)  # technically valid, but produces results you likely did not want (mean targets C,D while std will target D)
        >>> # traNDsforms will attempt to fix this for you with, essentially, std.view(2,1), but it will warn you
        >>>
        >>> std = std.view(2,1)
        >>> noiser = GaussianNoise(mean=mean, std=std, nd=2)  # fixes the "problem" above manually
    """

    def __init__(self, mean=0.0, std=1.0, p=1, nd=3, keys="*"):
        super().__init__(p, nd, keys)
        mean, std = self._validate_args(mean, std)

        self.mean = mean.view(*mean.shape, *[1] * nd)
        self.std = std.view(*std.shape, *[1] * nd)

    def _validate_args(self, mean, std):
        if isinstance(mean, (list, float, int, numpy.ndarray)):
            mean = torch.tensor(mean)
        if isinstance(std, (list, float, int, numpy.ndarray)):
            std = torch.tensor(std)

        if not isinstance(mean, torch.Tensor) or mean.is_complex():
            raise TypeError("Mean must be torch.tensor or convertible type with real numbers")
        if not isinstance(std, torch.Tensor) or std.is_complex():
            raise TypeError("Standard deviation must be torch.tensor or convertible type with real numbers")

        if not torch.is_floating_point(mean):
            mean = mean.float()
        if not torch.is_floating_point(std):
            std = std.float()

        if torch.any(std <= 0):
            raise ValueError("Standard deviation must be greater than 0")

        if mean.ndim != std.ndim and mean.ndim > 0 and std.ndim > 0:
            warnings.warn(f"Found differing number of dimensions between mean ({mean.ndim}) and std ({std.ndim})", UserWarning)
            if mean.ndim < std.ndim:
                mean = mean.view(*mean.shape, *[1] * (std.ndim - mean.ndim))
            else:
                std = std.view(*std.shape, *[1] * (mean.ndim - std.ndim))

        if mean.device != std.device:
            std = std.to(mean.device)

        return mean, std

    def apply(self, input, **params):
        sample = torch.normal(mean=torch.broadcast_to(self.mean, input.shape), std=torch.broadcast_to(self.std, input.shape))
        return input + sample
