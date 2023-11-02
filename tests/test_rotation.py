"""Tests for hello function."""
import pytest

from torch_trandsforms.rotation import RandomRotate90


@pytest.mark.parametrize(
    ("nd", "expected"),
    [(2, 2), (3, 6), (4, 12), (8, 56)],
)
def test_rotate90(nd, expected):
    """Test RandomRotate90"""
    rotator = RandomRotate90(nd=nd, p=1.0)
    assert len(rotator.options) == expected
