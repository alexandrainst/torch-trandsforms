## traNDsforms timing

Automatically generated with test_speed.py

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

**torch**: 2.2.2+cu121

**torchvision**: 0.17.2+cu121

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0006 | 0.0000 | 0.0087 | 0.0000 |
| UniformNoise | 0.0146 | 0.0000 | 0.1257 | 0.0000 |
| Normalize | 0.0008 | 0.0000 | 0.0207 | 0.0000 |
| SaltAndPepperNoise | 0.0489 | 0.0012 | 0.3371 | 0.0074 |
| AdditiveBetaNoise | 0.0457 | 0.0015 | 0.3559 | 0.0103 |
| GaussianNoise | 0.0140 | 0.0012 | 0.1360 | 0.0057 |
| RandomFlip | 0.0002 | 0.0000 | 0.0087 | 0.0000 |
| RandomRotate | 0.0434 | 0.0016 | 0.5589 | 0.0115 |
| RandomPadding | 0.0007 | 0.0000 | 0.0155 | 0.0001 |


### Block Transforms

Transforms use random padding (C=10) where applicable

Input size: 10x64x64x64. Testing output sizes:

| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |
|-------|----------|----------|-------------|------------------|
| CenterCrop | 0.0015 | 0.0099 | 0.3787 | 0.1121 |
| RandomCrop | 0.0035 | 0.0148 | 0.4060 | 0.0776 |
| Resize | 0.0021 | 0.0055 | 0.0476 | 0.0106 |
| RandomResize | 0.0015 | 0.0108 | 0.0770 | 0.0002 |
