# Examples of Use

Below are examples of outputs from a number of traNDsforms

The input cube is an RGB cube of size 3,36,36,36

The output size is expected to be 3,36,36,36 unless otherwise specified (crop, for example)

The colour values are scaled to the range 0-1. The original min/max values are attached as titles.

## Compose and Random Apply

```python
transform = Compose([
        RandomResize(0.3, p=0.75),
        RandomRotate([180,180,180], sample_mode="nearest", p=0.9),
        RandomCrop(24, padding=0, p=1.0),
        RandomApply([
            UniformNoise(p=1.0, low=-0.2, hi=0.2),
            GaussianNoise(std=0.05, p=1.0),
            SaltAndPepperNoise(0.2, low=0.0, hi=1.0, p=1.0, copy_input=True)
        ], min=1, max=1)
    ])
```

![Compose](./figures/Compose.png)

## Additive Beta Noise

![AdditiveBetaNoise](./figures/AdditiveBetaNoise.png)

## Gaussian Noise

![GaussianNoise](./figures/GaussianNoise.png)

## RandomCrop

![RandomCrop](./figures/RandomCrop.png)

## Random Flip

![RandomFlip](./figures/RandomFlip.png)

## Random Resize

![RandomResize](./figures/RandomResize.png)

## Random Rotate

![RandomRotate](./figures/RandomRotate.png)

## Random Rotate 90

![RandomRotate90](./figures/RandomRotate90.png)

## Salt and Pepper Noise

![SaltAndPepperNoise](./figures/SaltAndPepperNoise.png)

## Uniform Noise

![UniformNoise](./figures/UniformNoise.png)
