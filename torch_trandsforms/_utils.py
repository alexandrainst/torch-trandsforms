"""Utility functions"""

from typing import Sequence

from numbers import Number

import numpy
import torch


def get_tensor_sequence(x, sequence_length, acceptable_types=None):
    """
    Extract a tensor sequence of specified length and typing from input

    Args:
        x (Number, array-like, or torch.tensor): Input to convert to tensor of sequence length
        sequence_length (int): Length of the tensor
        acceptable_types (tuple of torch.dtype): Acceptable types to return. Raises ValueError if the input, after conversion to tensor, is not one of these. Default None (any)

    Returns:
        torch.tensor: tensor version of the input `x` of length `sequence_length` with one of `acceptable_types` dtype

    Raises:
        ValueError: If the converted tensor's dtype is not in the list of acceptable_types, or the length of the input does not match the expected sequence_length
        TypeError: If the input type could not be converted to torch.Tensor
    """
    if isinstance(x, Number):
        x = torch.tensor([x] * sequence_length)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x)
    elif isinstance(x, numpy.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Could not convert {type(x)} to torch.Tensor")
    if isinstance(acceptable_types, torch.dtype):
        acceptable_types = (acceptable_types,)
    if isinstance(acceptable_types, Sequence) and not x.dtype in acceptable_types:
        raise ValueError(
            f"Could not convert type {x.dtype} to one of acceptable types ({acceptable_types}). Ensure the input type is convertible to these types."
        )
    if x.ndim == 0 or x.shape[0] == 1:
        x = x.expand(sequence_length)
    elif x.shape[0] != sequence_length:
        raise ValueError(f"Expected sequence of length {sequence_length}, 0, or 1. Got length {x.shape[0]}")
    return x
