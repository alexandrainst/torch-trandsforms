from contextlib import nullcontext
from math import prod

import numpy
import pytest
import torch

from torch_trandsforms.shape import CenterCrop, Crop, RandomCrop, RandomFlip, RandomPadding, RandomResize, Resize


def test_crop():
    crop = Crop(4)
    tensor = torch.arange(4 * 4 * 4 * 4).view(4, 4, 4, 4)
    with pytest.raises(NotImplementedError):
        crop(tensor=tensor)


@pytest.mark.parametrize(
    ("shape", "size", "nd", "padding", "expected"),
    [
        ((4, 4, 4, 4), (2, 2, 2), 3, None, (4, 2, 2, 2)),
        ((4, 4, 4), numpy.array((2, 2, 2)), 3, None, (2, 2, 2)),
        ((4, 4, 4), (2, 2, 2), 2, None, ValueError),
        ((4, 4, 4, 4), torch.tensor([4, 2, 2, 2]), 4, None, (4, 2, 2, 2)),
        ((1, 1, 1), 5, 2, None, RuntimeError),
        ((1, 1, 1), 5, 2, 0.0, (1, 5, 5)),
    ],
)
def test_centercrop(shape, size, nd, padding, expected):
    with pytest.raises(expected) if not isinstance(expected, tuple) else nullcontext():
        tensor_1 = torch.zeros(shape)
        tensor_2 = torch.ones(shape, dtype=torch.long)
        cropper = CenterCrop(size, padding, nd=nd)
        cropped_1 = cropper(tensor_1=tensor_1, tensor_2=tensor_2)["tensor_1"]
        cropped_2 = cropper(tensor_1=tensor_1, tensor_2=tensor_2)["tensor_2"]
        assert cropped_1.shape == expected
        assert cropped_1.shape[-nd:] == cropped_2.shape[-nd:]


@pytest.mark.parametrize(
    ("shape", "size", "nd", "padding", "expected"),
    [
        ((4, 4, 4, 4), (2, 2, 2), 3, None, (4, 2, 2, 2)),
        ((4, 4, 4), numpy.array((2, 2, 2)), 3, None, (2, 2, 2)),
        ((4, 4, 4), (2, 2, 2), 2, None, ValueError),
        ((4, 4, 4, 4), torch.tensor([4, 2, 2, 2]), 4, None, (4, 2, 2, 2)),
        ((1, 1, 1), 5, 2, None, RuntimeError),
        ((1,), 5, 1, 0.0, (5,)),
    ],
)
def test_randomcrop(shape, size, nd, padding, expected):
    with pytest.raises(expected) if not isinstance(expected, tuple) else nullcontext():
        tensor_1 = torch.zeros(shape)
        tensor_2 = torch.ones(shape, dtype=torch.long)
        cropper = RandomCrop(size, padding, nd=nd)
        cropped_1 = cropper(tensor_1=tensor_1, tensor_2=tensor_2)["tensor_1"]
        cropped_2 = cropper(tensor_1=tensor_1, tensor_2=tensor_2)["tensor_2"]
        assert cropped_1.shape == expected
        assert cropped_1.shape[-nd:] == cropped_2.shape[-nd:]


@pytest.mark.parametrize(
    ("shape", "nd", "expected"),
    [
        ((4, 4, 4, 4), 3, None),
        ((8, 8, 8), 3, None),
        ((4, 4, 0), 1, None),
        ((4, 4), 3, ValueError),
        ((4, 1, 3, 1, 2, 1, 8, 9, 10), 8, None),
        ((4,), 6, ValueError),
    ],
)
def test_randomflip(shape, nd, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor_1 = torch.arange(prod(shape)).view(*shape)
        tensor_2 = torch.arange(prod(shape)).view(*shape)
        flipper = RandomFlip(p=1.0, nd=nd)
        result = flipper(t=tensor_1, p=tensor_2)
        assert result["t"].shape == result["p"].shape
        assert torch.all(result["t"] == result["p"])


@pytest.mark.parametrize(
    ("size", "scale_factor", "nd", "expected"),
    [
        (None, None, 3, ValueError),
        (10, None, 3, None),
        (None, 1.5, 3, None),
        (10, 1.5, 3, ValueError),
        (10, None, 4, NotImplementedError),
    ],
)
def test_scale(size, scale_factor, nd, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = torch.arange(3 * 8 * 8 * 8).view(3, 8, 8, 8).float()
        scaler = Resize(scale_factor=scale_factor, size=size, nd=nd, p=1.0)
        result = scaler(tensor=tensor)["tensor"]
        assert result.ndim == tensor.ndim
        if size:
            assert torch.all(torch.isclose(torch.tensor(result.shape[-nd:], dtype=torch.float), torch.tensor(size, dtype=torch.float)))
        elif scale_factor:
            size = scale_factor * torch.tensor(tensor.shape[-nd:])
            size = size.int()
            assert torch.all(torch.isclose(torch.tensor(result.shape[-nd:], dtype=torch.float), size.float()))


@pytest.mark.parametrize(
    ("size", "scale_factor", "nd", "expected"),
    [
        (None, None, 3, ValueError),
        (10, None, 3, TypeError),
        (None, 0.5, 3, None),
        (10, 0.5, 3, ValueError),
        (10, None, 4, NotImplementedError),
        (torch.tensor([6, 10]), None, 3, None),
        (None, numpy.array((0.8, 1.2)), 3, None),
        ([(3, 4), (5, 6), (7, 8), (9, 1)], None, 3, ValueError),
        (None, [(3, 4), (5, 6), (7, 8), (9, 1)], 3, ValueError),
        (([6, 8], [4, 8], [10, 12]), None, 3, None),
        (None, [torch.tensor(0.2), 0.4, (1, 2)], 3, None),
        (None, "this will fail", 3, TypeError),
        ("special_case", None, 3, RuntimeError),
    ],
)
def test_randomscale(size, scale_factor, nd, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = torch.arange(3 * 8 * 8 * 8).view(3, 8, 8, 8).float()

        if size == "special_case":
            scaler = RandomResize(scale_factor=0.2, size=None, nd=nd, p=1.0)
            scaler.scale_factor = None
            scaler.size = None
            scaler(tensor=tensor)

        scaler = RandomResize(scale_factor=scale_factor, size=size, nd=nd, p=1.0)
        result = scaler(tensor=tensor)["tensor"]
        assert result.ndim == tensor.ndim
        re_size = torch.tensor(result.shape)
        if size is not None:
            t_size_min = torch.tensor(scaler.size)[:, 0]
            t_size_max = torch.tensor(scaler.size)[:, 1]
            assert torch.all(re_size[-nd:] >= t_size_min)
            assert torch.all(re_size[-nd:] <= t_size_max)
        elif scale_factor is not None:
            t_size_min = (torch.tensor(scaler.scale_factor)[:, 0] * torch.tensor([8.0, 8.0, 8.0])).floor()
            t_size_max = (torch.tensor(scaler.scale_factor)[:, 1] * torch.tensor([8.0, 8.0, 8.0])).floor()
            assert torch.all(re_size[-nd:] >= t_size_min)
            assert torch.all(re_size[-nd:] <= t_size_max)


@pytest.mark.parametrize(
    ("input_size", "min_pad", "max_pad", "value", "nd", "expected"),
    [
        ((10, 10, 10), 0, 0, 0.0, 3, None),
        ((5,), 5, 5, 5.5, 1, None),
        ((3, 3, 3), 1, 7, torch.tensor([1.0, 2.0, 3.0]), 2, None),
        ((4, 8, 8, 8), "failure", 8, 0.0, 3, AssertionError),
        ((4, 8, 8, 8), 8, "failure", 0.0, 3, AssertionError),
        ((4, 8, 8, 8), 10, 8, 0.0, 3, AssertionError),
    ],
)
def test_randompadding(input_size, min_pad, max_pad, value, nd, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = torch.zeros(input_size)

        padder = RandomPadding(min_pad, max_pad, value, p=1.0, nd=nd)

        output = padder(tensor=tensor)["tensor"]

        for i in range(1, nd + 1):
            assert input_size[-i] + 2 * min_pad <= output.shape[-i] <= input_size[-i] + 2 * max_pad
