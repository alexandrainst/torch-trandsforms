from contextlib import nullcontext

import pytest
import torch

from torch_trandsforms.ops import Cast


@pytest.mark.parametrize(
    ("dtypes", "keys", "expected"),
    [
        (torch.float32, "*", None),
        ("float", ["foo", "bar"], None),
        (["torch.float32", torch.long], ["foo", "bar"], None),
        (["blah"], "*", ValueError),
        (["float", "long"], "*", ValueError),
        (["brain"], ["foo", "bar"], ValueError),
        (None, "*", TypeError),
    ],
)
def test_cast(dtypes, keys, expected):
    with pytest.raises(expected) if expected is not None else nullcontext():
        caster = Cast(dtype=dtypes, keys=keys)

        foo = torch.arange(16).view(2, 2, 2, 2)
        bar = torch.arange(16).view(2, 2, 2, 2)
        baz = torch.arange(16).view(2, 2, 2, 2)

        converted = caster(foo=foo, bar=bar, baz=baz)

        for key in converted.keys():
            c_dtype = converted[key].dtype
            t_dtype = torch.int64
            if keys == "*":
                t_dtype = caster._extract_dtypes(dtypes)
            elif key in keys:
                if isinstance(caster.dtypes, torch.dtype):
                    t_dtype = caster.dtypes
                else:
                    t_dtype = caster.dtypes[caster.keys.index(key)]

            assert c_dtype == t_dtype
