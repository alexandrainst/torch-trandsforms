import random

import torch
import torchvision

__all__ = ["BaseTransform", "KeyedTransform", "NdTransform", "KeyedNdTransform"]


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
        super().__init__()
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

    def __repr__(self):
        return self.__class__.__name__ + f"(p = {self.p})"


class KeyedTransform(BaseTransform):
    """
    As BaseTransform but operates only on the keys provided (i.e. only on "target", "data", etc)
    Useful for things that do not operate on the structure of the inputs (such as rotation), but instead on the values (such as noise) that are not desirable to introduce in the target data (or vice versa)
    Note that any KeyedTransform can operate on all given inputs if `keys == '*'` (default)

    Args:
        keys (`list` or `'*'`): List of inputs keys to operate on, or '*' to indicate any given key
    """

    def __init__(self, p=1.0, keys="*", **kwargs):
        super().__init__(p=p, **kwargs)
        if len(keys) == 0:  # only warns if keys is a 0-length iterable
            RuntimeWarning(f"Keyed transform {self.__class__.__name__} expected at least one key but got 0")
        self.keys = keys

    def __call__(self, **inputs):
        """
        Calls `apply` on each named input (given during __init__) with same params for each
        Extends `**params` with the input key under 'key_name' for use in apply

        Args:
            inputs (any): keyword args containing `torch.Tensor`s or whatever else the transform calls for

        Returns:
            dict: resulting outputs dict where the relevant inputs have been transformed
        """
        if torch.rand(1).item() < self.p:
            params = self.get_parameters(**inputs)

            for key, value in inputs.items():
                if key in self.keys or self.keys == "*":
                    params["key_name"] = key
                    inputs[key] = self.apply(value, **params)

        return inputs

    def __repr__(self):
        return self.__class__.__name__ + f"(p = {self.p}, keys = {self.keys})"


class NdTransform(BaseTransform):
    """
    A dimensionality-aware transform

    Args:
        nd (int): Number of dimensions, trailing, to operate on
    """

    def __init__(self, p=1.0, nd=3, **kwargs):
        super().__init__(p=p, **kwargs)

        assert 0 < nd, "nd (num dimensions) must be greater than 0"
        assert isinstance(nd, int), "nd must be an integer value"
        self.nd = nd

    def _check_nd_compliance(self, key, value):
        """Checks that input is correct dimensionality (if it even is a tensor)
        this does not check for compliance with expected input types"""
        if isinstance(value, torch.Tensor) and value.ndim < self.nd:
            raise ValueError(f"Expected {key} to have at least {self.nd} dimensions, got {value.ndim}")

    def __call__(self, **inputs):
        """
        Calls `apply` on each input with same params for each, ensuring all comply with dimensionality first

        Args:
            inputs (any): keyword args containing `torch.Tensor`s or whatever else the transform calls for

        Returns:
            dict: resulting outputs dict where the relevant inputs have been transformed
        """
        for key, value in inputs.items():
            self._check_nd_compliance(key, value)

        return super().__call__(**inputs)

    def __repr__(self):
        return self.__class__.__name__ + f"(p = {self.p}, nd = {self.nd})"


class KeyedNdTransform(KeyedTransform, NdTransform):
    """
    A dimensionality-aware transform that operates only on the keys provided (i.e. only on "target", "data", etc)
    See ``KeyedTransform`` and ``NdTransform`` for more info on why this is necessary

    Args:
        p (`float`): Probability of the transform activating on call - set to 1 to force use
        keys (`list` or `'*'`): List of inputs keys to operate on, or '*' to indicate any given key
        nd (`int`): Number of dimensions, trailing, to operate on
    """

    def __init__(self, p=1.0, nd=3, keys="*"):
        super().__init__(p=p, keys=keys, nd=nd)

    def __call__(self, **inputs):
        """
        Calls `apply` on each named input (given during __init__) with same params for each,
            ensuring a named input complies with dimensionality.
            Keep in mind this functionality only activates when the call to apply would,
            i.e. when the random generator is lower than the threshold and the key exists in the
            list of named keys (or keys=="*")
        Extends `**params` with the input key under 'key_name' for use in `apply`

        Args:
            inputs (any): keyword args containing `torch.Tensor`s or whatever else the transform calls for

        Returns:
            dict: resulting outputs dict where the relevant inputs have been transformed
        """
        if torch.rand(1).item() < self.p:
            params = self.get_parameters(**inputs)

            for key, value in inputs.items():
                self._check_nd_compliance(key, value)
                if key in self.keys or self.keys == "*":
                    params["key_name"] = key
                    inputs[key] = self.apply(value, **params)

        return inputs

    def __repr__(self):
        return self.__class__.__name__ + f"(p = {self.p}, nd = {self.nd}, keys = {self.keys})"
