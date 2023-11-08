"""Tests for hello function."""
import pytest
import torch

from torch_trandsforms.rotation import RandomRotate90


@pytest.mark.parametrize(
    ("nd", "expected"),
    [(2, 2), (3, 6), (4, 12), (8, 56)],
)
def test_rotate90(nd, expected):
    """Test RandomRotate90"""
    rotator = RandomRotate90(nd=nd, p=1.0)
    assert len(rotator.options) == expected

    tensor_1 = torch.arange(2**nd).view(*[2] * nd)
    tensor_2 = torch.arange(2**nd).view(*[2] * nd)
    tensor_3 = torch.arange(2**nd).view(*[2] * nd) + 1

    results = rotator(tensor_1=tensor_1, tensor_2=tensor_2, tensor_3=tensor_3)

    assert tensor_1.shape == results["tensor_1"].shape
    assert tensor_2.shape == results["tensor_2"].shape
    assert tensor_1.shape == results["tensor_3"].shape

    assert torch.all(results["tensor_1"] == results["tensor_2"])
    assert torch.all(results["tensor_1"] == results["tensor_3"] - 1)
