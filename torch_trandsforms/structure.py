"""Structural classes like Compose, Random Apply, etc"""

import copy

import torch

__all__ = ["Compose", "RandomApply"]


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


class RandomApply:
    """
    Randomly applies an amount of transforms

    Args:
        transforms (list of `transform` objects): list of transform objects to choose from
        min (int): minimum amount of transforms to choose from (default: 0)
        max (int): maximum amount of transforms to choose from (default: 1)
        p (list of `float` or None): list of probabilities with which to choose from (must be same length as `transforms`) (default: None)
        allow_same (bool): allow the use of the same transform multiple times? (default: False)

    Example:
        >>> torch_trandsforms.Compose([
        >>>    torch_trandsforms.UniformNoise(p=0.5, low=0, hi=0.1, keys=["data", "extra_data"])
        >>>    torch_trandsforms.RandomApply([  # don't forget to set the transforms' `p` to 1. or they may be ignored even if they are chosen
        >>>        torch_trandsforms.RandomRotate90(p=1., nd=3, keys='*'),
        >>>    ], min=1, max=4, allow_same=True)
        >>> ])
    """

    def __init__(self, transforms, min=0, max=1, p=None, allow_same=False):
        assert min >= 0, "min must not be negative"
        assert isinstance(min, int), "min must be an integer"
        assert isinstance(max, int), "max must be an integer"
        assert max > min, "max must be greater than min"

        assert isinstance(transforms, list), "transforms must be a list (not a single transform, for example)"

        if p is not None:
            assert isinstance(p, list), "list of probabilities must be iterable"
            if not len(transforms) == len(p):
                raise ValueError(f"List of probabilities ({len(p)}) must be equal to length of transforms ({len(transforms)})")

        if not allow_same:
            if max > len(transforms):
                raise ValueError("If not allow_same, max may not be greater than length of transforms")

        self.transforms = transforms
        self.min = min
        self.max = max
        self.probs = p
        self.allow_same = allow_same

    def __call__(self, **inputs):
        """
        For N (drawn from U(min,max)), select a random
        """
        N = torch.randint(self.min, self.max + 1, (1,)).item()

        probs = copy.copy(self.probs) if self.probs is not None else [1.0] * len(self.transforms)
        transforms = copy.copy(self.transforms)

        for n in range(int(N)):
            maxsum = sum(probs)
            cumsum = [sum(probs[:i]) - probs[0] for i in range(1, len(probs))]

            p = maxsum * torch.rand((1,)).item()

            for i in range(1, len(cumsum)):  # find where the random value slots in
                if p >= cumsum[i]:
                    inputs = transforms[i](**inputs)
                    break

            if not self.allow_same:
                del probs[i]
                del transforms[i]

        return inputs

    def __repr__(self) -> str:
        """stolen from Compose"""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            try:
                format_string += f"    {t}"
            except AttributeError:
                format_string += f"    {t.__class__.__name__}"
        format_string += "\n)"
        return format_string
