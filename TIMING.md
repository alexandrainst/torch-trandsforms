## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0031 | 0.0003 | 0.0093 | 0.0001 |
| UniformNoise | 0.0147 | 0.0002 | 0.1263 | 0.0003 |
| Normalize | 0.0024 | 0.0001 | 0.0194 | 0.0000 |
| SaltAndPepperNoise | 0.0438 | 0.0010 | 0.3422 | 0.0055 |
| AdditiveBetaNoise | 0.0483 | 0.0012 | 0.3594 | 0.0086 |
| GaussianNoise | 0.0169 | 0.0005 | 0.1527 | 0.0026 |
| RandomFlip | 0.0017 | 0.0001 | 0.0099 | 0.0000 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0017 | 0.0175 | 0.4313 | 0.0854 |
| RandomCrop | 0.0021 | 0.0118 | 0.4192 | 0.0841 |
