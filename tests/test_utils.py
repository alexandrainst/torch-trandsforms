from contextlib import nullcontext

import numpy
import pytest
import torch

from torch_trandsforms._utils import extract_min_max, get_affine_matrix, get_rot_2d, get_rot_3d, get_tensor_sequence


@pytest.mark.parametrize(
    ("x", "sequence_length", "acceptable_types", "expected"),
    [
        (42, 3, (torch.int, torch.long), None),
        ([16], 7, torch.long, None),
        (42.0, 5, (torch.float), None),
        ([6, 9], 2, None, None),
        ((6, 9), 2, torch.long, None),
        (numpy.array([6, 9]), 2, torch.long, None),
        (42.0, 3, (torch.int, torch.long), ValueError),
        ([6, 9], 1, None, ValueError),
        (numpy.array([4, 2, 5]), 2, torch.long, ValueError),
        ("hello", 2, torch.long, TypeError),
    ],
)
def test_sequencer(x, sequence_length, acceptable_types, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        tensor = get_tensor_sequence(x, sequence_length, acceptable_types)
        assert len(tensor == sequence_length)
        if acceptable_types:
            assert tensor.dtype == acceptable_types or tensor.dtype in acceptable_types


@pytest.mark.parametrize(
    ("angle", "expected"), [(None, TypeError), ([0, 1], ValueError), (90, None), (0.0, None), (torch.tensor(7654321), None)]
)
def test_rot2d(angle, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        rot_mat = get_rot_2d(angle)
        assert rot_mat.shape == (2, 2)


@pytest.mark.parametrize(
    ("angles", "expected"),
    [(None, TypeError), (90, TypeError), ([0, 1, 2], None), ((720, 360, 180), None), (torch.tensor([4, 5, 6]), None)],
)
def test_rot3d(angles, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        rot_mat = get_rot_3d(angles)
        assert rot_mat.shape == (3, 3)


@pytest.mark.parametrize(
    ("nd", "rotation", "translation", "expected"),
    [
        (None, None, None, ValueError),
        (3, None, None, None),
        (None, get_rot_2d(90), None, None),
        (None, get_rot_3d((0, 90, 180)), None, None),
        (None, None, torch.tensor((1, 2, 3)).view(-1, 1), None),
    ],
)
def test_affiner(nd, rotation, translation, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        affine = get_affine_matrix(rotation, translation, nd)
        assert affine.shape[-2] == affine.shape[-1] - 1


@pytest.mark.parametrize(
    ("input", "base", "allow_value", "expected"),
    [
        (5, 0, True, None),
        (5, 0, False, TypeError),
        ([6, 0], 0, True, AssertionError),
        ((1.0, 2.0, 3.0), 400.0, False, ValueError),
        (torch.tensor(4.0), 10.0, False, TypeError),
        (numpy.array((4.0, 5.0)), 0, True, None),
        (numpy.array([[[1.0, 2.0]]]), 0, True, ValueError),
        (torch.tensor([3, 4, 5]), 0, True, ValueError),
        ("failure case", 0, False, TypeError),
    ],
)
def test_extract_min_max(input, base, allow_value, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        v = extract_min_max(input, base, allow_value)
        assert type(v) == tuple
        assert len(v) == 2
        assert v[0] < v[1]
