import torch

from .base import BaseTransform


class RandomRotate90(BaseTransform):  # note the use of BaseTransform as base class
    """
    Rotates the input 90 degrees around a randomly determined axis
    """

    def __init__(self, nd=3, p=0.5):
        super().__init__(p=p, nd=nd)
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

    def apply(self, input, **params):
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
