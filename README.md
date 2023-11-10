# torch traNDsforms

<div align="center">

[![Build status](https://github.com/alexandrainst/torch-trandsforms/workflows/build/badge.svg?branch=main&event=push)](https://github.com/alexandrainst/torch-trandsforms/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-trandsforms.svg)](https://pypi.org/project/torch-trandsforms/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/alexandrainst/torch-trandsforms/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/alexandrainst/torch-trandsforms/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/alexandrainst/torch-trandsforms/releases)
[![License](https://img.shields.io/github/license/alexandrainst/torch-trandsforms)](https://github.com/alexandrainst/torch-trandsforms/blob/main/LICENSE)
![Coverage Report](assets/images/coverage.svg)

A pytorch-first transform library for ND data, such as multi-channel 3D volumes

torch traNDsforms is in early alpha and may have bugs and a significant lack of usefulness :)

</div>

## Features

torch traNDsforms is an easy to use transform library for N-dimensional PyTorch data

 - Compatible with nearly any `torch.Tensor`
 - One transform pipeline for all your data using `KeyedTransforms`
 - Customizable and lightweight
 - No superfluous dependencies
 - Collaborative

## Installation

In early alpha, install torch traNDsforms like this:

```bash
pip install git+ssh://git@github.com/alexandrainst/torch-trandsforms.git/
```

Soon enough, you will be able to run the following:

```bash
pip install torch_trandsforms
```

or

```bash
poetry add torch-trandsforms
```

or

```bash
conda install torch_trandsforms
```

## Usage

Creating the RandomRotate90 class, as an example of customizing your own transform:

```python
import torch
from torch_trandsforms.base import BaseTransform

class RandomRotate90(BaseTransform):  # note the use of BaseTransform as base class here
    """
    Rotates the input 90 degrees around a randomly determined axis
    NOTE: This is the not actual implementation of RandomRotate90
    """
    def __init__(self, nd=3, p=0.5):
        super().__init__(p = p, nd = nd)
        self.options = self._get_options(nd)

    def _get_options(self, nd):
        """
        Create potential rotations based on the nd argument
        This can be lower than the number of dimensions of the actual input
            in case you do not want a leading dimension to be rotated
        """
        options = []

        for i in range(nd):
            for j in range(nd):
                if not i == j:
                    options.append((-i-1, -j-1))

        return options
    
    def get_parameters(self, **inputs):
        """
        overrides the base get_parameters to choose a random
            rotation option for each input
        """
        rotation = random.choice(self.options)
        return {'rot':rotation}
    
    def apply(self, input, **params):
        """
        apply MUST be overwritten 
        It is applied to each input sequentially, and thus must have
            parameters that are exactly equal for each instance,
            meaning most likely NO randomization here
        """
        rot = params['rot']
        return torch.rot90(input, dims=rot)
```

And we can now use our class to demonstrate the library functionality:

```python
tensor = torch.arange(16).view(2,2,2,2)  # create a 4D tensor
another_tensor = torch.arange(16).view(2,2,2,2)  # create an exactly equal tensor for demonstration

print(tensor)
print(another_tensor)

random_rotator = RandomRotate90(nd=2, p=1.)  # we only want the last two dimensions to be rotateable but it should rotate every time (p=1)

transformed = random_rotator(data=tensor, foo=another_tensor)  # "data" is arbitrary, it is the key that will be returned, demonstrated by "foo"

print(transformed['data'])
print(transformed['foo'])
```

## Support

Please use [Issues](https://github.com/alexandrainst/torch-trandsforms/issues) for any issues, feature requests, or general feedback.

## Roadmap

For now, traNDsforms is in early alpha. That will continue for a while, while basic functionality is implemented.

The roadmap is determined by the collaborative efforts of every user that provides feedback, reports bugs, or produces pull requests. Thank you!

For now, the roadmap looks something like this:
 - [x] Implement basic functionality (normalize, dtype changing, change device)
 - [x] Implement value-level noise functionality (uniform, salt and pepper, gaussian)
 - [ ] Implement structural transforms (scaling, cropping)
 - [ ] More examples, including better visuals
 - [ ] Development structure: Lock main && publish
 - [ ] Move basic functionality to _functional and _utils

Later additions (and reasons for postponing):
 - [ ] Arbitrary rotations (missing efficient ND computation)
 - [ ] Gaussian Blur (missing implementation of ND convolution)
 - [ ] Affine transformations (missing efficient ND computation)

Potential additions:
 - [ ] Geometric operations using PyTorch Geometric
 - [ ] Point clouds, meshes using PyTorch 3D
 - [ ] Data loading, sampling, and structures
 - [ ] torchscript compatibility

## Contributing

See [Contributing](https://github.com/alexandrainst/torch-trandsforms/blob/main/CONTRIBUTING.md)

## Authors

The project is maintained by developers at the [Alexandra Institute](https://alexandra.dk/)

 - Oliver G. Hjermitslev (ohjerm) <oliver.gyldenberg@alexandra.dk>

...to be expanded...

## License

See the [MIT License](https://github.com/alexandrainst/torch-trandsforms/blob/main/LICENSE)

## 📃 Citation

```bibtex
@misc{torch-trandsforms,
  author = {Alexandra Institute},
  title = {A pytorch-first transform library for ND data, such as multi-channel 3D volumes},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alexandrainst/torch-trandsforms}}
}
```
