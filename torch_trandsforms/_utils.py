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


def angle2radians(angle):
    if isinstance(angle, torch.Tensor):
        return angle * (torch.pi / 180.0)
    return torch.tensor(angle * (torch.pi / 180.0))


def get_rot_2d(angle):
    """
    Get the 2D rotation matrix for affine_grid

    Args:
        angle (float): angle in degrees

    Returns:
        torch.tensor: 2x2 rotation matrix

    Raises:
        ValueError: If given a sequence of list or tuple (non-array-like) with length != 1
    """
    if isinstance(angle, (list, tuple)):
        if not len(angle) == 1:
            raise ValueError(f"Length of angle should be 1 (or 0, i.e. scalar)")
        angle = angle[0]

    a = angle2radians(angle)
    R = torch.tensor([[torch.cos(a), -torch.sin(a)], [torch.sin(a), torch.cos(a)]])
    return R


def get_rot_3d(angles):
    """
    Get the combined 3D rotation matrix for affine_grid

    Args:
        angles (sequence of float): angles in degrees (D,H,W order)

    Returns:
        torch.tensor: 3x3 rotation matrix
    """
    alpha, beta, gamma = (angle2radians(angle) for angle in angles)
    roll = torch.tensor([[1.0, 0.0, 0.0], [0.0, torch.cos(gamma), -torch.sin(gamma)], [0.0, torch.sin(gamma), torch.cos(gamma)]])

    yaw = torch.tensor([[torch.cos(alpha), -torch.sin(alpha), 0.0], [torch.sin(alpha), torch.cos(alpha), 0.0], [0.0, 0.0, 1.0]])

    pitch = torch.tensor([[torch.cos(beta), 0.0, torch.sin(beta)], [0.0, 1.0, 0.0], [-torch.sin(beta), 0.0, torch.cos(beta)]])

    R = roll @ yaw @ pitch
    return R


def get_affine_matrix(rotation=None, translation=None, nd=None):
    """
    Creates an affine rotation and translation matrix. If nd and inputs are None, returns an Identity matrix

    Args:
        rotation (Optional[torch.tensor]): Tensor representation of rotation (N*N)
        translation (Optional[torch.tensor]): Tensor representation of translation (N*1)
        nd (Optional[int]): If nd is given and other inputs are None, creates identity versions of those inputs

    Returns:
        torch.tensor: Affine transformation matrix of size N*(N+1)

    Raises:
        ValueError: one of nd or rotation must not be None
    """
    if nd is None:
        if rotation is not None:
            nd = rotation.shape[-1]
        elif translation is not None:
            nd = translation.shape[0]
        else:
            raise ValueError("One of nd or rotation must be non-None")

    if rotation is None:
        rotation = torch.eye(nd)

    if translation is None:
        translation = torch.zeros(rotation.shape[-2]).view(-1, 1)

    return torch.cat([rotation, translation], dim=-1)
