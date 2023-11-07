"""Contains transforms targeting pytorch tensor operations"""

import torch

from .base import KeyedTransform

__all__ = ["Cast", "ConvertDtype"]

type_str_to_type = {
    "float": torch.float32,
    "float32": torch.float32,
    "double": torch.float64,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "cfloat": torch.complex64,
    "complex128": torch.complex128,
    "cdouble": torch.complex128,
    "float16": torch.float16,
    "half": torch.float16,
    "binary16": torch.float16,
    "bfloat16": torch.bfloat16,
    "brain": torch.bfloat16,
    "uint8": torch.uint8,
    "char": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "bool": torch.bool,
}


class Cast(KeyedTransform):
    """
    Converts the named tensors to the specified dtype
    Does not do safe conversion and does not check under/overflow

    Args:
        dtype (str or torch.dtype, or list of dtypes): dtype to change the tensor to. If string, attempts to infer from best efforts. If list, compares to list idx of `keys` (default: torch.float32)

    Example:
        >>> caster = Cast(torch.float32, keys="*")  # converts all inputs to float32
        >>> caster = Cast("float", keys=["foo", "bar"])  # converts inputs named foo and bar to float32 (ignores baz, qux, and garply)
        >>> caster = Cast(["torch.float32", torch.long], keys=["foo", "bar"])  # converts foo to torch.float32 and bar to torch.int64
        >>> caster = Cast(["blah"], keys="*")  # raises ValueError
        >>> caster = Cast(["float", "long"], keys="*")  # raises ValueError (because len(dtypes) != len(keys))
        >>> caster = Cast(["brain"], keys=["foo", "bar"])  # raises ValueError (because len(dtypes) != len(keys))
    """

    def __init__(self, dtype, p=1, keys="*", **kwargs):
        super().__init__(p, keys, **kwargs)
        self.dtypes = self._extract_dtypes(dtype)
        if isinstance(self.dtypes, list) and self.keys == "*":
            raise ValueError(f"If dtype is list, keys must be specified (and have the same length)")
        elif isinstance(self.dtypes, list) and isinstance(self.keys, list):
            if len(self.dtypes) != len(self.keys):
                raise ValueError(f"list of dtypes (len = {len(self.dtypes)}) must equal length of keys (len = {len(self.keys)})")

    def _extract_dtypes(self, dtype):
        """
        Attempts to extract the dtype from the input dtype (converting str or list to torch.dtype(s))
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            try:
                return type_str_to_type[dtype]
            except KeyError as e:
                try:
                    return type_str_to_type[dtype.replace("torch.", "")]
                except KeyError as ee:
                    raise ValueError(f"Could not convert {dtype} to torch.dtype - valid str dtypes are: {type_str_to_type.keys()}")
        if isinstance(dtype, list):
            return [self._extract_dtypes(d) for d in dtype]
        raise TypeError(f"Found wrong argument type {type(dtype)}, expected one of `str`, `torch.dtype`, or `list`")

    def _get_current_dtype(self, key):
        if isinstance(self.dtypes, torch.dtype):
            return self.dtypes
        return self.dtypes[self.keys.index(key)]

    def apply(self, input, **params):
        ckey = params["key_name"]
        dtype = self._get_current_dtype(ckey)
        return input.to(dtype)


class ConvertDtype(Cast):
    """
    Alias for `Cast`
    """

    pass
