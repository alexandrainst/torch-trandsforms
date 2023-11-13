## traNDsforms timing

### System Info

**CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU

### Basic Transforms

| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |
|-------|-------------|------------------|----------------|---------------------|
| RandomRotate90 | 0.0033 | 0.0002 | 0.0086 | 0.0001 |
| UniformNoise | 0.0139 | 0.0002 | 0.1201 | 0.0003 |
| Normalize | 0.0016 | 0.0001 | 0.0167 | 0.0000 |
| SaltAndPepperNoise | 0.0453 | 0.0011 | 0.3221 | 0.0056 |
| AdditiveBetaNoise | 0.0482 | 0.0012 | 0.3806 | 0.0085 |
| GaussianNoise | 0.0187 | 0.0004 | 0.1591 | 0.0025 |
| RandomFlip | 0.0017 | 0.0001 | 0.0106 | 0.0000 |
