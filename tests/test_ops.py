from contextlib import nullcontext

import pytest
import torch

from torch_trandsforms.ops import Cast, ToDevice


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
    """Tests Cast and its alias ConvertDtype"""
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


@pytest.mark.parametrize(
    ("devices", "keys", "expected"),
    [
        (torch.device("cpu"), "*", None),
        ("cpu", ["foo", "bar"], None),
        (["cuda", "cpu"], ["foo", "bar"], None),
        (["blah"], "*", RuntimeError),
        (["cuda", "cpu"], "*", ValueError),
        (["cuda"], ["foo", "bar"], ValueError),
        (None, "*", TypeError),
    ],
)
def test_todevice(devices, keys, expected):
    """Tests ToDevice and its alias To"""

    # change `expected` to RuntimeError when cuda is necessary but unavailable
    if not torch.cuda.is_available() and expected is None:
        if (isinstance(devices, torch.device) and devices.type == "cuda") or (isinstance(devices, str) and devices == "cuda"):
            expected = RuntimeError
        elif isinstance(devices, list):
            for dev in devices:
                if (isinstance(dev, torch.device) and dev.type == "cuda") or (isinstance(dev, str) and dev == "cuda"):
                    expected = RuntimeError

    # test ToDevice with `expected` error (if any)
    with pytest.raises(expected) if expected is not None else nullcontext():
        caster = ToDevice(device=devices, keys=keys)

        foo = torch.arange(16).view(2, 2, 2, 2)
        bar = torch.arange(16).view(2, 2, 2, 2)
        baz = torch.arange(16).view(2, 2, 2, 2)

        converted = caster(foo=foo, bar=bar, baz=baz)

        for key in converted.keys():
            c_dev = converted[key].device
            t_dev = torch.device("cpu")
            if keys == "*":
                t_dev = caster._extract_device(devices)
            elif key in keys:
                if isinstance(caster.device, torch.device):
                    t_dev = caster.device
                else:
                    t_dev = caster.device[caster.keys.index(key)]

            assert c_dev.type == t_dev.type
            if c_dev.index and t_dev.index:
                assert c_dev.index == t_dev.index
