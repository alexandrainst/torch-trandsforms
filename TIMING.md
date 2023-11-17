## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0027 | 0.0003 | 0.0091 | 0.0002 |
| UniformNoise | 0.0150 | 0.0002 | 0.1252 | 0.0003 |
| Normalize | 0.0024 | 0.0001 | 0.0169 | 0.0000 |
| SaltAndPepperNoise | 0.0437 | 0.0010 | 0.3250 | 0.0055 |
| AdditiveBetaNoise | 0.0453 | 0.0012 | 0.3447 | 0.0084 |
| GaussianNoise | 0.0149 | 0.0004 | 0.1388 | 0.0025 |
| RandomFlip | 0.0006 | 0.0001 | 0.0091 | 0.0000 |
| RandomRotate | 0.0443 | 0.0016 | 0.3919 | 0.0110 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0025 | 0.0114 | 0.3932 | 0.0966 |
| RandomCrop | 0.0020 | 0.0111 | 0.4324 | 0.0815 |
