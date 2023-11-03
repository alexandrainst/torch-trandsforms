import pytest
import torch

from torch_trandsforms.rotation import RandomRotate90
from torch_trandsforms.structure import Compose, RandomApply
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


@pytest.mark.parametrize(
    ("min", "max", "N_t", "p", "allow_same", "expected_error"),
    [
        (-1, 2, 1, None, False, AssertionError),
        (3, 1, 1, None, False, AssertionError),
        (1, 2, 4, [1, 2, 3], False, ValueError),
        (4, 5, 3, [1, 1, 1], False, ValueError),
        (1, 3, 3, [1, 2, 1], True, None),
    ],
)
def test_random_apply(min, max, N_t, p, allow_same, expected_error):
    """test RandomApply"""

    rot = RandomRotate90(p=1)

    if expected_error is not None:
        with pytest.raises(expected_error):
            applier = RandomApply([rot] * N_t, min=min, max=max, p=p, allow_same=allow_same)
    else:
        applier = RandomApply([rot] * N_t, min=min, max=max, p=p, allow_same=allow_same)
        tensor = torch.arange(16, dtype=torch.float).view(2, 2, 2, 2)

        transformed = applier(tensor=tensor)["tensor"]

        assert transformed.shape == (2, 2, 2, 2)
        assert transformed.dtype == torch.float
