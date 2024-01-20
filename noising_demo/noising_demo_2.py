import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = 64
args.dataset_path = r"dog"

# dataloader = get_data(args)

diff = Diffusion(device="cpu")

# image = next(iter(dataloader))[0]

import torchvision
from PIL import Image

image = Image.open("dog/dog.0.jpg")

transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
                torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

image = transforms(image)

t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
