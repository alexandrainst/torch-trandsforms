from contextlib import nullcontext
from math import prod

import numpy
import pytest
import torch

from torch_trandsforms._functional import affine_grid_sampling, crop, pad, rotate
from torch_trandsforms._utils import get_affine_matrix, get_output_size, get_rot_3d


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
    ("input_shape", "angle", "out_size", "mode", "sample_mode", "padding_mode", "align_corners", "expected"),
    [
        ((1, 3, 4, 4), 90, None, "crop", "nearest", "zeros", True, None),
        ((10, 10, 24, 24, 24), [45, 0, 42], (10, 10, 12, 12, 12), "nothing", "bilinear", "reflection", False, None),
        ((4, 5, 6), 7, (4, 5, 6), "nothing", "nearest", "zeros", False, NotImplementedError),
        ((1, 4, 5, 6), 7, None, "nothing", "nearest", "zeros", False, None),
        ((1, 3, 24, 24, 24, 24), [6, 7, 8, 9, 10, 11], None, "crop", "nearest", "zeros", True, NotImplementedError),
        ((1, 3, 24, 24, 24), [10, 20], None, "nothing", "nearest", "zeros", True, NotImplementedError),
    ],
)
def test_rotate(input_shape, angle, out_size, mode, sample_mode, padding_mode, align_corners, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = torch.arange(prod(input_shape), dtype=torch.float).view(*input_shape)
        if out_size is None:
            out_size = input_shape
        result = rotate(tensor, angle, out_size, mode, sample_mode, padding_mode, align_corners)
        assert result.shape == out_size


@pytest.mark.parametrize(
    ("input_shape", "angle", "translation", "mode"),
    [
        ((1, 1, 6, 8, 6), [0, 90, 90], [0, 0, 0], "crop"),
        ((1, 1, 6, 6, 9), [90, 0, 90], [0, 0, 0], "minpad"),
        ((1, 1, 6, 6, 6), [90, 90, 0], [0, 0, 0], "maxpad"),
        ((1, 1, 4, 4, 6), [90, 90, 90], [0, 0, 0], "nothing"),
        ((1, 1, 6, 6, 8), [0, 45, 45], [0, 0, 0], "crop"),
        ((1, 1, 6, 9, 6), [45, 0, 45], [0, 0, 0], "minpad"),
        ((1, 1, 6, 6, 4), [45, 45, 0], [0, 0, 0], "maxpad"),
        ((1, 1, 6, 6, 6), [45, 45, 45], [0, 0, 0], "nothing"),
        ((1, 1, 9, 9, 9), [45, 45, 0], [-1, 3, 0], "crop"),
        ((1, 1, 6, 6, 6), [45, 0, 45], [-1, 0, 0], "minpad"),
        ((1, 1, 6, 6, 4), [0, 45, 45], [0, 0, -4], "maxpad"),
        ((1, 1, 6, 6, 6), [45, 45, 45], [1, 0, 0], "nothing"),
        ((1, 1, 6, 6, 6), [0, 0, 0], [1, 2, 3], "cheese melt"),
    ],
)
def test_affine_grid_sampling(input_shape, angle, translation, mode):
    from warnings import warn

    rots = get_rot_3d(angle)
    mat = get_affine_matrix(rots, torch.tensor(translation).unsqueeze(-1), 3)
    max_size = [1, 1, *get_output_size(input_shape[-3:], mat, round=True)]

    with pytest.raises(ValueError) if mode == "cheese melt" else nullcontext():
        size = max_size if mode != "cheese melt" else None
        sampled = affine_grid_sampling(torch.rand(input_shape), theta=mat, size=input_shape, mode=mode)

        if mode in ("crop", "nothing"):
            assert torch.equal(torch.tensor(sampled.shape), torch.tensor(input_shape))
        elif mode == "minpad":
            assert torch.equal(torch.tensor(sampled.shape), torch.tensor(max_size))
        elif mode == "maxpad":
            mx = torch.tensor(size[-3:]).max()  # type: ignore
            cubed_shape = torch.tensor([1, 1, mx, mx, mx])
            assert torch.equal(torch.tensor(sampled.shape), cubed_shape)
        else:
            raise ValueError("test mode failed")
