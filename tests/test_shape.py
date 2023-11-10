from contextlib import nullcontext

import numpy
import pytest
import torch

from torch_trandsforms.shape import CenterCrop, Crop, RandomCrop


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
