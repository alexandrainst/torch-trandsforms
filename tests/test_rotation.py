"""Tests for hello function."""
from contextlib import nullcontext
from math import prod

import numpy
import pytest
import torch

from torch_trandsforms.rotation import RandomRotate, RandomRotate90


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


@pytest.mark.parametrize(
    ("input_shape", "nd", "rotation", "keys", "expected"),
    [
        ((1, 3, 6, 6), 2, 45, ["foo", "bar"], None),
        ((3, 6, 6), 2, 45, ["foo", "bar"], None),
        ((1, 3, 6, 6, 6), 3, [45, 45, 45], ["foo", "bar"], None),
        (None, 4, None, ["foo", "bar"], NotImplementedError),
        (None, 1, None, ["foo", "bar"], NotImplementedError),
        ((1, 3, 6, 6), 2, torch.tensor(45), ["foo", "bar"], None),
        ((1, 3, 6, 6, 6), 3, numpy.array([45, 45, 45]), ["foo", "bar"], None),
        ((1, 3, 6, 6), 2, "please rotate up to 45 degrees", ["foo", "bar"], TypeError),
        ((1, 3, 6, 6, 6), 3, [45], ["foo", "bar"], ValueError),
        ((1, 3, 6, 6), 2, (45, 6, 4), ["foo", "bar"], ValueError),
        ((1, 3, 6, 6), 2, (45, 6), ["foo", "bar"], ValueError),
    ],
)
def test_rotate(input_shape, nd, rotation, keys, expected):
    """Test RandomRotate"""
    with pytest.raises(expected) if expected is not None else nullcontext():
        rotator = RandomRotate(rotation, nd=nd, keys=keys, align_corners=True)

        foo = torch.arange(prod(input_shape)).view(*input_shape).float()
        bar = torch.arange(prod(input_shape)).view(*input_shape).float()

        res = rotator(foo=foo, bar=bar)

        assert res["foo"].shape == foo.shape
        assert res["bar"].shape == foo.shape

        if "foo" in keys and "bar" in keys:
            assert torch.all(torch.isclose(res["foo"], res["bar"]))

        # do it again
        res = rotator(foo=res["foo"], bar=res["bar"])

        assert res["foo"].shape == foo.shape
        assert res["bar"].shape == foo.shape

        if "foo" in keys and "bar" in keys:
            assert torch.all(torch.isclose(res["foo"], res["bar"]))
