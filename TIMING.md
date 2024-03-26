## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

**torch**: 2.2.1+cu121

**torchvision**: 0.17.1+cu121

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0006 | 0.0000 | 0.0093 | 0.0000 |
| UniformNoise | 0.0150 | 0.0000 | 0.1343 | 0.0000 |
| Normalize | 0.0012 | 0.0000 | 0.0190 | 0.0000 |
| SaltAndPepperNoise | 0.0491 | 0.0012 | 0.3660 | 0.0074 |
| AdditiveBetaNoise | 0.0490 | 0.0015 | 0.4212 | 0.0103 |
| GaussianNoise | 0.0156 | 0.0012 | 0.1647 | 0.0057 |
| RandomFlip | 0.0006 | 0.0000 | 0.0131 | 0.0000 |
| RandomRotate | 0.0499 | 0.0018 | 0.6502 | 0.0146 |
| RandomPadding | 0.0008 | 0.0000 | 0.0206 | 0.0001 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0019 | 0.0141 | 0.5988 | 0.1427 |
| RandomCrop | 0.0049 | 0.0247 | 0.6699 | 0.1209 |
| Resize | 0.0024 | 0.0114 | 0.0861 | 0.0006 |
| RandomResize | 0.0025 | 0.0069 | 0.0535 | 0.0002 |
