from numbers import Number

import pytest
import torch

from torch_trandsforms.value import UniformNoise


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
