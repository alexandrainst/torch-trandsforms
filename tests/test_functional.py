from contextlib import nullcontext
from math import prod

import numpy
import pytest
import torch

from torch_trandsforms._functional import crop, pad, rotate


@pytest.mark.parametrize(
    ("in_shape", "padding", "value", "expected"),
    [
        ((4, 4, 4), (0, 0), 0, (4, 4, 4)),
        ((4, 4, 4), (1, 1, 1, 1, 1, 1), 0.0, (6, 6, 6)),
        ((4, 4, 4), (1, 2, 1, 3), "reflect", (4, 8, 7)),
        ((4, 4, 4), (40, 40), torch.tensor([1, 2, 3, 4]), (4, 4, 84)),
        ((4, 4, 4), (40, 40), numpy.array([0.0, 2, 3, 4]), (4, 4, 84)),
        ((4, 4, 4), "wrong", 0.0, TypeError),
        ((4, 4, 4), (2, 2, 1, 1), "fail", NotImplementedError),
        ((4, 4, 4), (2, 2, 1, 1), {"this": "willfail"}, RuntimeError),
    ],
)
def test_pad(in_shape, padding, value, expected):
    print(expected)
    with pytest.raises(expected) if not isinstance(expected, tuple) else nullcontext():
        tensor = torch.zeros(in_shape)
        padded = pad(tensor, padding, value)
        assert padded.shape == expected
        if isinstance(value, float):
            test_value = torch.tensor(value)
        else:
            test_value = value
        if isinstance(value, torch.Tensor):
            test_value = torch.broadcast_to(test_value.view(*test_value.shape, *[1] * (len(padding) // 2)), padded.shape)
            assert test_value in padded


@pytest.mark.parametrize(
    ("pos", "size", "pad", "expected"),
    [
        (1, 2, None, TypeError),
        ((1,), (2,), None, (4, 4, 2)),
        ((1, 1), (2, 2), None, (4, 2, 2)),
        ((-1, -1), (2, 2), None, RuntimeError),
        ((-1, -1), (2, 2), 0.0, (4, 2, 2)),
        ((3, 3), torch.tensor((2, 2)), None, RuntimeError),
        (numpy.array((3, 3)), (2, 2), 0.0, (4, 2, 2)),
    ],
)
def test_crop(pos, size, pad, expected):
    with pytest.raises(expected) if not isinstance(expected, tuple) else nullcontext():
        tensor = torch.zeros((4, 4, 4))
        cropped = crop(tensor, pos, size, pad)
        assert cropped.shape == expected


@pytest.mark.parametrize(
    ("input_shape", "angle", "out_size", "sample_mode", "padding_mode", "align_corners", "expected"),
    [
        ((1, 3, 4, 4), 90, None, "nearest", "zeros", True, None),
        ((10, 10, 24, 24, 24), [45, 0, 42], (10, 10, 12, 12, 12), "bilinear", "reflection", False, None),
        ((4, 5, 6), 7, (4, 5, 6), "nearest", "zeros", False, NotImplementedError),
        ((1, 4, 5, 6), 7, None, "nearest", "zeros", False, None),
        ((1, 3, 24, 24, 24, 24), [6, 7, 8, 9, 10, 11], None, "nearest", "zeros", True, NotImplementedError),
        ((1, 3, 24, 24, 24), [10, 20], None, "nearest", "zeros", True, NotImplementedError),
    ],
)
def test_rotate(input_shape, angle, out_size, sample_mode, padding_mode, align_corners, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = torch.arange(prod(input_shape), dtype=torch.float).view(*input_shape)
        if out_size is None:
            out_size = input_shape
        result = rotate(tensor, angle, out_size, sample_mode, padding_mode, align_corners)
        assert result.shape == out_size
