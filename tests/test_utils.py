from contextlib import nullcontext

import numpy
import pytest
import torch

from torch_trandsforms._utils import get_tensor_sequence


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
