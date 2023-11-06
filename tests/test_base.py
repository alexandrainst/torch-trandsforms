import pytest

from torch_trandsforms.base import *


@pytest.mark.parametrize(
    ("cl"),
    [(BaseTransform), (NdTransform), (KeyedTransform), (KeyedNdTransform)],
)
def test_base_classes(cl):
    transform = cl()

    parameters = transform.get_parameters()
    assert isinstance(parameters, dict)
    assert len(parameters.keys()) == 0

    with pytest.raises(NotImplementedError):
        transform.apply(None, **parameters)

    assert transform.__class__.__name__ in repr(transform)
