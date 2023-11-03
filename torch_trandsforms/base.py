import random

import torch
import torchvision


class BaseTransform(torch.nn.Module):
    """Base transformation class providing basic utility for all ND transforms
    Users can extend this using `class MyTransform(BaseTransform)`
     - Any extension MUST override the apply method thus: `def apply(self, input, **params):`
     - additionally, users can override `get_parameters(self, **inputs)`, returning a dict of randomization parameters to apply to each **input. This is unpacked in `apply` as shown above
    IMPORTANT NOTE: Operations do and should expect the common PyTorch order of input tensors:
     - For example, 3*H*W for RGB images or (1*)D*H*W for common volumes and C*D*H*W for multi-channel volume data
     - Any Transform should operate equally on 1*D*H*W as D*H*W if specified, or at the very least explicitly state the expected dimensionality
     - See for example the `RandomRotate90` transform for inspiration (explicitly operating on the last `nd` dimensions)

    Args:
        p (float): Probability of the transform activating on call - set to 1 to force use
    """

    def __init__(self, p=1.0):
        assert 0 <= p <= 1, "p value must be in the range 0-1 (inclusive)"
        self.p = p

    def get_parameters(self, **inputs):
        # return the parameters used for all inputs
        # used to homogenize the randomization across all input types (data, mask, target, etc...)
        return {}

    def __call__(self, **inputs):
        """
        Calls `apply` on each input with same params for each

        Args:
            inputs (any): keyword args containing `torch.Tensor`s or whatever else the transform calls for

        Returns:
            dict: resulting outputs dict where the relevant inputs have been transformed
        """
        if torch.rand(1).item() < self.p:
            params = self.get_parameters(**inputs)

            for key, value in inputs.items():
                inputs[key] = self.apply(value, **params)

        return inputs

    def apply(self, input, **params):
        raise NotImplementedError("BaseTransform is a superclass with no utility. Extend it using `class MyTransform(BaseTransform)`")


class KeyedTransform(BaseTransform):
    """
    As BaseTransform but operates only on the keys provided (i.e. only on "target", "data", etc)
    Useful for things that do not operate on the structure of the inputs (such as rotation), but instead on the values (such as noise) that are not desirable to introduce in the target data (or vice versa)

    Args:
        keys (list): List of `inputs` keys to operate on
    """

    def __init__(self, p=1.0, keys=[]):
        super().__init__(p=p)
        if len(keys) == 0:
            RuntimeWarning(f"Keyed transform {self.__class__.__name__} expected at least one key but got 0")
        self.keys = keys

    def __call__(self, **inputs):
        """
        Calls `apply` on each named input (given during __init__) with same params for each

        Args:
            inputs (any): keyword args containing `torch.Tensor`s or whatever else the transform calls for

        Returns:
            dict: resulting outputs dict where the relevant inputs have been transformed
        """
        if torch.rand(1).item() < self.p:
            params = self.get_parameters(**inputs)

            for key, value in inputs.items():
                if key in self.keys:
                    inputs[key] = self.apply(value, **params)

        return inputs


class NdTransform(BaseTransform):
    """
    A dimensionality-aware transform

    Args:
        nd (int): Number of dimensions, trailing, to operate on
    """

    def __init__(self, p=1.0, nd=3):
        super().__init__(p=p)

        assert 0 < nd, "nd (num dimensions) must be greater than 0"
        assert isinstance(nd, int), "nd must be an integer value"
        self.nd = nd
