from contextlib import nullcontext
from numbers import Number

import pytest
import torch

from torch_trandsforms.value import Normalize, UniformNoise


@pytest.mark.parametrize(
    ("lo", "hi", "dtype"),
    [(-1, 5, torch.float), ("hi", 3, torch.float), (1, None, torch.float), (400, 600, torch.long)],
)
def test_uniform(lo, hi, dtype):
    """Test UniformNoise"""
    if not isinstance(lo, Number) or not isinstance(hi, Number):
        with pytest.raises(ValueError):
            noiser = UniformNoise(lo, hi)
    else:
        noiser = UniformNoise(lo, hi, keys=["tensor"], p=1.0)
        tensor = torch.zeros((4, 4), dtype=dtype)

        noised = noiser(tensor=tensor)["tensor"]
        assert noised.shape == (4, 4)
        assert torch.all(noised >= lo)
        assert torch.all(noised < hi)


@pytest.mark.parametrize(
    ("mean", "std", "nd", "input", "expected"),
    [
        (None, 1, 3, (), AssertionError),
        (0, None, 3, (), AssertionError),
        (0, 0, 3, (), ValueError),
        ([0.0], [1, 2], 3, (2, 48, 24, 12), None),
        (0, 1.0, 3, (36, 36, 36), None),
        ([1, 2, 3], 1, 3, (2, 48, 24, 12), RuntimeError),
        ([1, 2, 3], [1, 2, 3], 2, (3, 24, 24, 24), RuntimeError),
        ([[1, 2], [2, 3]], 1.0, 2, (2, 2, 16, 16), None),
    ],
)
def test_normalize(mean, std, nd, input, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        normalizer = Normalize(mean, std, nd=nd, p=1.0)

        assert normalizer.mean.ndim >= nd
        assert normalizer.std.ndim >= nd

        tensor = torch.ones(input)
        result = normalizer(tensor=tensor)["tensor"]

        assert tensor.shape == result.shape
        assert torch.is_floating_point(result)
