from typing import Any, Dict

import pytest
import torch

from torch_trandsforms.base import *


@pytest.mark.parametrize(
    ("cl"),
    [(BaseTransform), (NdTransform), (KeyedTransform), (KeyedNdTransform)],
)
def test_base_classes(cl):
    transform = cl(p=1.0)

    tensor = torch.arange(4).view(2, 2)
    input: Dict[str, Any] = {"input": tensor}

    no_input: Dict[str, Any] = {}

    parameters = transform.get_parameters()
    assert isinstance(parameters, dict)
    assert len(parameters.keys()) == 0

    with pytest.raises(NotImplementedError):
        transform.apply(**input, **parameters)

    results = transform(**no_input)
    assert results == no_input

    with pytest.raises(NotImplementedError):
        transform(input=None)

    assert transform.__class__.__name__ in repr(transform)

    if hasattr(transform, "_check_nd_compliance"):
        with pytest.raises(ValueError):
            transform._check_nd_compliance("input", input["input"])
