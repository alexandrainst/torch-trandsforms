## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0028 | 0.0002 | 0.0086 | 0.0001 |
| UniformNoise | 0.0144 | 0.0002 | 0.1219 | 0.0003 |
| Normalize | 0.0016 | 0.0001 | 0.0164 | 0.0000 |
| SaltAndPepperNoise | 0.0409 | 0.0010 | 0.3159 | 0.0055 |
| AdditiveBetaNoise | 0.0429 | 0.0012 | 0.3258 | 0.0085 |
| GaussianNoise | 0.0141 | 0.0004 | 0.1353 | 0.0025 |
| RandomFlip | 0.0014 | 0.0001 | 0.0083 | 0.0000 |
| RandomRotate | 0.0425 | 0.0014 | 0.3807 | 0.0121 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0019 | 0.0126 | 0.3747 | 0.0772 |
| RandomCrop | 0.0018 | 0.0126 | 0.3724 | 0.0749 |
| Resize | 0.0027 | 0.0058 | 0.0526 | 0.0002 |
| RandomResize | 0.0023 | 0.0060 | 0.0522 | 0.0003 |
