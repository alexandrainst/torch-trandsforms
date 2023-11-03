import pytest
import torch

from torch_trandsforms.rotation import RandomRotate90
from torch_trandsforms.structure import Compose
from torch_trandsforms.value import UniformNoise


def test_compose():
    """test Compose"""

    transform = Compose([RandomRotate90(p=1.0), UniformNoise(p=1.0)])

    assert "RandomRotate90" in str(transform)
    assert "UniformNoise" in str(transform)

    tensor = torch.arange(16, dtype=torch.float).view(2, 2, 2, 2)
    transformed = transform(tensor=tensor)["tensor"]

    assert transformed.shape == (2, 2, 2, 2)
    assert transformed.dtype == torch.float
