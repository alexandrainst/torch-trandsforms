## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0023 | 0.0002 | 0.0092 | 0.0001 |
| UniformNoise | 0.0124 | 0.0002 | 0.1125 | 0.0003 |
| Normalize | 0.0014 | 0.0001 | 0.0144 | 0.0000 |
| SaltAndPepperNoise | 0.0352 | 0.0009 | 0.2833 | 0.0055 |
| AdditiveBetaNoise | 0.0369 | 0.0012 | 0.2986 | 0.0085 |
| GaussianNoise | 0.0110 | 0.0004 | 0.1225 | 0.0025 |
| RandomFlip | 0.0006 | 0.0001 | 0.0072 | 0.0000 |
| RandomRotate | 0.0383 | 0.0011 | 0.3269 | 0.0091 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0022 | 0.0084 | 0.3346 | 0.0729 |
| RandomCrop | 0.0021 | 0.0090 | 0.3269 | 0.0719 |
| Resize | 0.0012 | 0.0041 | 0.0373 | 0.0001 |
| RandomResize | 0.0011 | 0.0041 | 0.0372 | 0.0001 |
