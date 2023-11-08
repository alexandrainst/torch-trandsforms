from typing import ContextManager, Union

from contextlib import AbstractContextManager, nullcontext
from numbers import Number

import pytest
import torch

from torch_trandsforms.value import AdditiveBetaNoise, GaussianNoise, Normalize, SaltAndPepperNoise, UniformNoise


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


@pytest.mark.parametrize(
    ("prob", "low", "hi", "a", "b", "expected"),
    [
        (None, -1, 1, 0.5, 0.5, AssertionError),
        (1, "None", 1, 0.5, 0.5, AssertionError),
        (1, -1, -2, 0.5, 0.5, AssertionError),
        (1, -1, 1, "None", 0.5, AssertionError),
        (1, -1, 1, 0.5, -1, AssertionError),
        (1, -1, 1, 0.5, 0.5, None),
        (0.5, -100, 1000, 0.1, 0.9, None),
        (100, -100, 1000, 0.9, 0.1, AssertionError),
    ],
)
def test_saltandpepper(prob, low, hi, a, b, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        noiser = SaltAndPepperNoise(prob=prob, low=low, hi=hi, a=a, b=b, p=1)
        tensor = torch.zeros(4, 4, 4, 4)
        noised = noiser(tensor=tensor)["tensor"]

        assert torch.all(low <= noised)
        assert torch.all(noised <= hi)
        assert tensor.shape == noised.shape


@pytest.mark.parametrize(
    ("prob", "low", "hi", "a", "b", "expected"),
    [
        (None, -1, 1, 0.5, 0.5, AssertionError),
        (1, "None", 1, 0.5, 0.5, AssertionError),
        (1, -1, -2, 0.5, 0.5, AssertionError),
        (1, -1, 1, "None", 0.5, AssertionError),
        (1, -1, 1, 0.5, -1, AssertionError),
        (1, -1, 1, 0.5, 0.5, None),
        (0.5, -100, 1000, 0.1, 0.9, None),
        (100, -100, 1000, 0.9, 0.1, AssertionError),
    ],
)
def test_additivebetanoise(prob, low, hi, a, b, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        noiser = AdditiveBetaNoise(prob=prob, low=low, hi=hi, a=a, b=b, p=1)
        tensor = torch.zeros(4, 4, 4, 4)
        noised = noiser(tensor=tensor)["tensor"]

        assert torch.all(low <= noised)
        assert torch.all(noised <= hi)
        assert tensor.shape == noised.shape


@pytest.mark.parametrize(
    ("mean", "std", "nd", "expected"),
    [
        (0, 1, 3, type(None)),
        ([3, 1], 1, 3, type(None)),
        (0, [1, 2], 3, type(None)),
        (0, torch.arange(4).view(2, 2) + 1, 2, type(None)),
        (0, -1, 3, ValueError),
        (None, 1, 3, TypeError),
        (0, None, 3, TypeError),
        (torch.arange(4).view(2, 2), torch.tensor(1), 3, RuntimeError),
        ("special_cuda_case", torch.tensor(1.0), 3, type(None)),
    ],
)
def test_gaussiannoise(mean, std, nd, expected):
    if mean == "special_cuda_case" and not torch.cuda.is_available():
        expected = RuntimeError
    with pytest.raises(expected) if issubclass(expected, Exception) else nullcontext():
        if mean == "special_cuda_case":
            mean = torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float64)

        tensor = torch.arange(16).view(2, 2, 2, 2)
        noiser = GaussianNoise(mean=mean, std=std, nd=nd)
        tensor = tensor.to(noiser.mean.device)
        result = noiser(tensor=tensor)["tensor"]
        assert tensor.shape == result.shape


@pytest.mark.parametrize(
    ("mean", "std", "nd", "expected"),
    [
        (torch.arange(4).view(2, 2), torch.arange(2) + 1, 2, UserWarning),
        (torch.arange(2), torch.arange(4).view(2, 2) + 1, 2, UserWarning),
        (torch.arange(4).view(2, 2), torch.tensor(1), 2, type(None)),
        (torch.arange(4).view(2, 2), torch.arange(4).view(2, 2) + 1, 2, type(None)),
        (0, torch.arange(4).view(2, 2) + 1, 2, type(None)),
    ],
)
def test_gausswarning(mean, std, nd, expected):
    cm: Union[pytest.WarningsRecorder, AbstractContextManager[None]] = (
        pytest.warns(expected) if issubclass(expected, Warning) else nullcontext()
    )
    with cm:
        tensor = torch.arange(16).view(2, 2, 2, 2)
        noiser = GaussianNoise(mean=mean, std=std, nd=nd)
        result = noiser(tensor=tensor)["tensor"]
        assert tensor.shape == result.shape
