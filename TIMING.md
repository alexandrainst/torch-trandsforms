## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0033 | 0.0002 | 0.0094 | 0.0001 |
| UniformNoise | 0.0145 | 0.0002 | 0.1266 | 0.0003 |
| Normalize | 0.0027 | 0.0001 | 0.0203 | 0.0000 |
| SaltAndPepperNoise | 0.0441 | 0.0009 | 0.3356 | 0.0055 |
| AdditiveBetaNoise | 0.0445 | 0.0012 | 0.3419 | 0.0085 |
| GaussianNoise | 0.0164 | 0.0005 | 0.1406 | 0.0025 |
| RandomFlip | 0.0012 | 0.0001 | 0.0091 | 0.0000 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0015 | 0.0187 | 0.4090 | 0.0846 |
| RandomCrop | 0.0018 | 0.0107 | 0.4413 | 0.0858 |
