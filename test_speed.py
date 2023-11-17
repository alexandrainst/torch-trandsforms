import subprocess
import time

import torch

from torch_trandsforms.rotation import RandomRotate, RandomRotate90
from torch_trandsforms.shape import CenterCrop, RandomCrop, RandomFlip
from torch_trandsforms.value import AdditiveBetaNoise, GaussianNoise, Normalize, SaltAndPepperNoise, UniformNoise


def write_file_head(file):
    cpu = subprocess.check_output("cat /proc/cpuinfo  | grep 'name'| uniq", shell=True).decode().strip().split(": ")[-1]
    gpu = subprocess.check_output("nvidia-smi -L", shell=True).decode().strip().replace("GPU 0: ", "").split(" (UUID")[0]

    file.write("## traNDsforms timing\n\n")
    file.write("### System Info\n\n")
    file.write(f"**CPU**: {cpu}\n\n**GPU**: {gpu}\n\n")
    file.write("### Basic Transforms\n\n")
    file.write("| Class | 10x64x64x64 | 10x64x64x64 CUDA | 10x128x128x128 | 10x128x128x128 CUDA |\n")
    file.write("|-------|-------------|------------------|----------------|---------------------|\n")


def test_standard(file, cl):
    tensor64 = torch.rand((10, 64, 64, 64))
    tensor128 = torch.rand((10, 128, 128, 128))

    # check specific transforms with needed args:
    if cl == Normalize:
        transform = cl(0.0, 1.0, p=1.0, nd=3)
    elif cl == SaltAndPepperNoise or cl == AdditiveBetaNoise:
        transform = cl(0.1, p=1.0)
    elif cl == RandomRotate:
        transform = cl([180.0, 180.0, 180.0], sample_mode="bilinear", padding_mode="zeros", align_corners=True, p=1.0)
    else:
        transform = cl(p=1.0)

    start_time = time.time()
    transform(tensor=tensor64)
    t64_time = time.time() - start_time

    start_time = time.time()
    transform(tensor=tensor128)
    t128_time = time.time() - start_time

    tensor64 = tensor64.to("cuda:0")
    tensor128 = tensor128.to("cuda:0")

    # check specific transforms with needed args:
    if cl == Normalize:
        transform = cl(torch.tensor(0.01, device="cuda:0"), torch.tensor(1.01, device="cuda:0"), p=1.0)
    elif cl == SaltAndPepperNoise or cl == AdditiveBetaNoise:
        transform = cl(0.1, a=torch.tensor(0.5, device="cuda:0"), b=torch.tensor(0.5, device="cuda:0"), p=1.0)
    elif cl == GaussianNoise:
        transform = cl(torch.tensor(0.01, device="cuda:0"), p=1.0)
    elif cl == RandomRotate:
        transform = cl([180.0, 180.0, 180.0], sample_mode="bilinear", padding_mode="zeros", align_corners=True, p=1.0)
    else:
        transform = cl(p=1.0)

    start_time = time.time()
    transform(tensor=tensor64)
    t64_cuda_time = time.time() - start_time

    start_time = time.time()
    transform(tensor=tensor128)
    t128_cuda_time = time.time() - start_time

    file.write(f"| {transform.__class__.__name__} | {t64_time:.4f} | {t64_cuda_time:.4f} | {t128_time:.4f} | {t128_cuda_time:.4f} |\n")


def write_crop_head(file):
    file.write("\n\n")
    file.write("### Block Transforms\n\n")
    file.write("Transforms use random padding (C=10) where applicable\n\n")
    file.write("Input size: 10x64x64x64. Testing output sizes:\n\n")
    file.write("| Class | 32x32x32 | 64x64x64 | 128x128x128 | 128x128x128 CUDA |\n")
    file.write("|-------|----------|----------|-------------|------------------|\n")


def test_block(file, cl):
    tensor = torch.rand((10, 64, 64, 64))
    padding = torch.rand((10,))

    t32 = cl(32, padding=padding, p=1.0, nd=3)
    t64 = cl(64, padding=padding, p=1.0, nd=3)
    t128 = cl(128, padding=padding, p=1.0, nd=3)
    t128_cuda = cl(128, padding=padding.to("cuda:0"), p=1.0, nd=3)

    start_time = time.time()
    t32(tensor=tensor)
    t32_time = time.time() - start_time

    start_time = time.time()
    t64(tensor=tensor)
    t64_time = time.time() - start_time

    start_time = time.time()
    t128(tensor=tensor)
    t128_time = time.time() - start_time

    tensor = tensor.to("cuda:0")

    start_time = time.time()
    t128_cuda(tensor=tensor)
    t128_cuda_time = time.time() - start_time

    file.write(f"| {cl.__name__} | {t32_time:.4f} | {t64_time:.4f} | {t128_time:.4f} | {t128_cuda_time:.4f} |\n")


def main():
    torch.manual_seed(451)

    classes = [RandomRotate90, UniformNoise, Normalize, SaltAndPepperNoise, AdditiveBetaNoise, GaussianNoise, RandomFlip, RandomRotate]
    block_classes = [CenterCrop, RandomCrop]

    if torch.cuda.is_available():  # only run on CUDA systems
        with open("TIMING.md", "w+") as file:
            write_file_head(file)
            for cl in classes:
                test_standard(file, cl)
            write_crop_head(file)
            for cl in block_classes:
                test_block(file, cl)
    else:
        raise OSError("This script should not run on systems without cuda")


if __name__ == "__main__":
    main()
