"""Basic transform functionality"""

from numbers import Number

import torch
import torch.nn.functional as t_F

from ._utils import get_affine_matrix, get_rot_2d, get_rot_3d, get_tensor_sequence


def pad(x, padding, value=0.0):
    """
    Pad the input given the padding tuple
    `padding` refers to a torch.nn.functional.pad-style padding tuple, i.e. starting at the last dimension and operating on `len(padding) // 2` dimensions

    Args:
        x (torch.tensor): tensor to pad
        padding (tuple): padding tensor (see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html for information)
        value (float, str, or torch.tensor): If number, does constant fill with the value. If str, attempts to use https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html padding modes (implemented for <=3 dimensions of padding)
            If torch.tensor, fills the new padded area with the given value(s).
            This works in ND, i.e. padding in 3 dimensions with a 0, 1, or 2D value works for a 5D input.
            However, for ND tensors the input must be directly broadcastable to the leading N dimensions of the input (i.e. dimensions must be 1 or equal to input dimension)
            Most of the time, a 0D-value is equivalent (but slower) to a scalar value (so choose `0.0` over torch.tensor(0.0))

    Returns:
        torch.tensor: padded input

    Example:
        >>> x = torch.arange(16).view(4,4)
        >>> pad(x, (1,2), value=400)
        >>> # tensor([[400,   0,   1,   2,   3, 400, 400],
        [400,   4,   5,   6,   7, 400, 400],
        [400,   8,   9,  10,  11, 400, 400],
        [400,  12,  13,  14,  15, 400, 400]])
        >>> pad(x.float(), (1,2), value='reflect')
        >>> # tensor([[ 1.,  0.,  1.,  2.,  3.,  2.,  1.],
        [ 5.,  4.,  5.,  6.,  7.,  6.,  5.],
        [ 9.,  8.,  9., 10., 11., 10.,  9.],
        [13., 12., 13., 14., 15., 14., 13.]])
        >>> pad(x.float(), (1,2), value=torch.tensor([400.0, 300.0, 200.0, 100.0]))
        >>> # tensor([[400.,   0.,   1.,   2.,   3., 400., 400.],
        [300.,   4.,   5.,   6.,   7., 300., 300.],
        [200.,   8.,   9.,  10.,  11., 200., 200.],
        [100.,  12.,  13.,  14.,  15., 100., 100.]])

    TODO: Surely this can be cleaned and accelerated using scatter?
    """
    dim = len(padding) // 2

    if isinstance(value, float):  # if constant, we can simply pad with this number
        padded_x = t_F.pad(x, padding, mode="constant", value=value)
    elif isinstance(value, str):  # torch padding mode with None value
        padded_x = t_F.pad(x, padding, mode=value)
    else:  # Channel-wise padding is not implemented for ND in torch, so we do it here
        padded_x = t_F.pad(x, padding, mode="constant", value=0)

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        pad_value = value.to(padded_x.device).to(padded_x.dtype)  # converts padding to expected parameters (device and dtype)
        pad_value = torch.broadcast_to(pad_value, padded_x.shape[:-dim])
        pad_value = pad_value.view((*pad_value.shape, *[1] * dim))  # particular ordering of broadcasting and viewing
        pad_value = torch.broadcast_to(pad_value, padded_x.shape)  # final broadcast for indexing

        for idx, (l, r) in enumerate(zip(padding[::2], padding[1::2])):
            indexor = torch.ones((padded_x.shape[-idx - 1]), dtype=torch.bool)
            if r > 0:
                indexor[l:-r] = False
            else:
                indexor[l:] = False
            indexor = indexor.view((*[1] * (dim - idx - 1), -1, *[1] * idx))  # append 1's to massage index dimensions
            indexor = torch.broadcast_to(indexor, padded_x.shape[-dim:])  # indexor has same shape as padded_x's trailing N dimensions

            padded_x[..., indexor] = pad_value[..., indexor]  # finally set the edge's padded indices to the padding value provided

    return padded_x


def crop(x, pos, size, padding=None):
    """
    Crop the input `x` at index `pos` with size `size`

    Args:
        x (torch.tensor): Input tensor from which to crop
        pos (array-like): ND sequence of position to crop (such that the output crop becomes [...,pos:pos+size]). NOTE that this is NOT center-cropping
        size (array-like): ND sequence of size to crop.
        padding (str, float or array-like): Padding value (for leading dimension(s)) or base torch padding modes for <=3D padding.
            If None, and the intended crop extends the input shape, raises IndexError, so must be sanitized first (pos+size <= shape)
            Similarly, if padding is None, any negative values in pos will be interpreted as negative indexing and raise a RuntimeError
            Otherwise, will shift pos to align with the padded input (shifted to the right by padding amount) to always return a correctly sized output

    Returns:
        torch.tensor: The output crop

    Example:
        >>> x = torch.arange(4*4*4).view(4,4,4)
        >>> crop(x, (1,1), (2,2))
        >>> # tensor([[[ 5,  6],
         [ 9, 10]],

        [[21, 22],
         [25, 26]],

        [[37, 38],
         [41, 42]],

        [[53, 54],
         [57, 58]]])
        >>> crop(x, (1,1,2), (2,2,1))
        >>> # tensor([[[22],
         [26]],

        [[38],
         [42]]])
        >>> crop(x, (-1,-1,-1), (6,6,6))
        >>> # IndexError: index out of range in self
        >>> crop(x, (-1,-1,-1), (6,6,6), padding=0)
        >>> # tensor([[[ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  1,  2,  3,  0],
         [ 0,  4,  5,  6,  7,  0],
         [ 0,  8,  9, 10, 11,  0],
         [ 0, 12, 13, 14, 15,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0],
         [ 0, 16, 17, 18, 19,  0],
         [ 0, 20, 21, 22, 23,  0],
         [ 0, 24, 25, 26, 27,  0],
         [ 0, 28, 29, 30, 31,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0],
         [ 0, 32, 33, 34, 35,  0],
         [ 0, 36, 37, 38, 39,  0],
         [ 0, 40, 41, 42, 43,  0],
         [ 0, 44, 45, 46, 47,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0],
         [ 0, 48, 49, 50, 51,  0],
         [ 0, 52, 53, 54, 55,  0],
         [ 0, 56, 57, 58, 59,  0],
         [ 0, 60, 61, 62, 63,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0]]])
    """
    pos = get_tensor_sequence(pos, len(pos), torch.long)
    size = get_tensor_sequence(size, len(size), torch.long)
    assert len(pos) == len(size), f"Cannot crop using differing lengths for pos and size (got {len(pos)} and {len(size)})"
    assert len(pos) <= x.ndim, f"Cannot crop using input of dimensionality less than the length of the crop parameters"

    c_d = len(pos)
    shape_t = torch.tensor(x.shape[-c_d:])

    if padding is not None and (torch.any(pos + size > shape_t) or torch.any(pos < 0)):
        padding_l = torch.abs(torch.min(torch.tensor(0), pos))  # padding l is only any pos (corner value) less than 0
        pos += padding_l  # align pos to new corner before calculating padding_r
        padding_r = torch.abs(
            torch.min(torch.tensor(0), shape_t - (pos + size))
        )  # padding r is any corner value (pos+size) larger than shape_t
        padding_params = tuple(
            torch.stack((padding_l.flip(0), padding_r.flip(0)), dim=-1).flatten().tolist()
        )  # flip padding order for torch.pad

        padded_x = pad(x, padding_params, padding)
    else:
        padded_x = x

    for idx, (p, s) in enumerate(zip(pos, size)):
        padded_x = torch.index_select(padded_x, idx - c_d, torch.arange(p, p + s, device=padded_x.device))

    return padded_x


def affine_grid_sampling(input, theta, size=None, sample_mode="bilinear", padding_mode="zeros", align_corners=None):
    """
    Placeholder for generating flow fields and grid sampling in one
    TODO: align_corners automatic decision if None
    TODO: padding_mode implement custom padding
    TODO: allow ...-leading input dimensionality
    TODO: theta sizing adjustment
    TODO: size expansion and typing
    TODO: input N + theta N alignment

    Args:
        input (torch.Tensor): The input tensor to transform
        theta (torch.Tensor): Batch of affine matrices
        size (torch.Size): The output size
        sample_mode (str): See https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html `mode` for information on sampling. (default: bilinear)
        padding_mode (str): See https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html `padding_mode` for information on padding. (default: zeros)
        align_corners (bool): Whether to consider the edge pixel or its center the input's -1,1 extents. If None (default) will

    Returns:
        torch.tensor: The grid-sampled input, as determined by theta (the affine transformation matrix).
    """

    if theta.ndim < 3 or theta.shape[0] == 1:
        theta = torch.broadcast_to(theta, (input.shape[0], *theta.shape[-2:]))
    grid = t_F.affine_grid(theta, size or input.Size(), align_corners=align_corners)
    return t_F.grid_sample(input, grid, mode=sample_mode, padding_mode=padding_mode, align_corners=align_corners)


def rotate(input, angle, out_size=None, sample_mode="bilinear", padding_mode="zeros", align_corners=None):
    """
    Uses torch grid_sample to rotate spatial and volumetric data
    Currently not implemented for >3D (TODO: Implement ND grid sampling)

    Args:
        input (torch.Tensor): input tensor to rotate
        angle (float or sequence of float): angle(s) to rotate the input in trailing order
        out_size (Optional[sequence]): The output size. If None, uses input.size() to determine the output size
        sample_mode (str): See https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html `mode` for information on sampling. (default: bilinear)
        padding_mode (str): See https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html `padding_mode` for information on padding. (default: zeros)
        align_corners (bool): Whether to consider the edge pixel or its center the input's -1,1 extents. If None (default) will

    Returns:
        torch.tensor: rotated input (see affine_grid_sampling for more information)

    Raises:
        NotImplementedError: If angle assumes rotation in anything but 2 or 3 dimensions (i.e. 1 or 3 inputs)
    """

    if isinstance(angle, (float, int)):
        angle = get_tensor_sequence(float(angle), 1, torch.float32)

    if len(angle) > 3:
        raise NotImplementedError(
            "Rotation for ND>3 has not yet been implemented. Please see https://github.com/alexandrainst/torch-trandsforms/issues for related issues"
        )

    if len(angle) == 1:
        theta = get_rot_2d(angle)
        theta = get_affine_matrix(rotation=theta)
    elif len(angle) == 3:
        theta = get_rot_3d(angle)
        theta = get_affine_matrix(rotation=theta)
    else:
        raise NotImplementedError(f"Rotation for len(angle) = {len(angle)} is not implemented")

    print(theta)

    return affine_grid_sampling(
        input, theta, out_size or input.size(), sample_mode=sample_mode, padding_mode=padding_mode, align_corners=align_corners
    )
