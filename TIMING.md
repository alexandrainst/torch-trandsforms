## traNDsforms timing

Automatically generated with test_speed.py

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

**torch**: 2.5.1+cu124

**torchvision**: 0.20.1+cu124

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0006 | 0.0000 | 0.0137 | 0.0000 |
| UniformNoise | 0.0191 | 0.0000 | 0.1320 | 0.0000 |
| Normalize | 0.0031 | 0.0000 | 0.0227 | 0.0000 |
| SaltAndPepperNoise | 0.0426 | 0.0011 | 0.3265 | 0.0068 |
| AdditiveBetaNoise | 0.0538 | 0.0014 | 0.3501 | 0.0093 |
| GaussianNoise | 0.0132 | 0.0012 | 0.1423 | 0.0057 |
| RandomFlip | 0.0002 | 0.0000 | 0.0111 | 0.0000 |
| RandomRotate | 0.0422 | 0.0018 | 0.4836 | 0.0150 |
| RandomPadding | 0.0005 | 0.0000 | 0.0186 | 0.0001 |
| RandomBlock | 0.0001 | 0.0000 | 0.0002 | 0.0000 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0018 | 0.0150 | 0.4458 | 0.1431 |
| RandomCrop | 0.0048 | 0.0136 | 0.4218 | 0.1149 |
| Resize | 0.0017 | 0.0058 | 0.0521 | 0.0005 |
| RandomResize | 0.0017 | 0.0064 | 0.0524 | 0.0001 |
