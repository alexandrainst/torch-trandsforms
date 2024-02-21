## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

**torch**: 2.2.0+cu121

**torchvision**: 0.17.0+cu121

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0007 | 0.0000 | 0.0104 | 0.0000 |
| UniformNoise | 0.0172 | 0.0000 | 0.1336 | 0.0000 |
| Normalize | 0.0010 | 0.0000 | 0.0176 | 0.0000 |
| SaltAndPepperNoise | 0.0435 | 0.0012 | 0.3254 | 0.0074 |
| AdditiveBetaNoise | 0.0451 | 0.0015 | 0.3482 | 0.0103 |
| GaussianNoise | 0.0170 | 0.0012 | 0.1412 | 0.0057 |
| RandomFlip | 0.0004 | 0.0000 | 0.0093 | 0.0000 |
| RandomRotate | 0.0436 | 0.0017 | 0.5095 | 0.0123 |
| RandomPadding | 0.0006 | 0.0000 | 0.0160 | 0.0001 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0016 | 0.0113 | 0.3970 | 0.2539 |
| RandomCrop | 0.0045 | 0.0165 | 0.3915 | 0.0814 |
| Resize | 0.0020 | 0.0053 | 0.0472 | 0.0005 |
| RandomResize | 0.0022 | 0.0061 | 0.0468 | 0.0002 |
