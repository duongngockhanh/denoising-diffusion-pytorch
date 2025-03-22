import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self,
                 noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 img_size=256,
                 img_channels=3,
                 device="cuda"):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(params):
    device = params["device"]
    img_channels = params["image_channels"]
    dataloader = train_loader
    model = UNet(c_in=1, c_out=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"])
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=params["image_size"],
                          img_channels=img_channels,
                          device=device)

    for epoch in range(params["epochs"]+1):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for _, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        os.makedirs(os.path.join("results", params["run_name"]), exist_ok=True)
        os.makedirs(os.path.join("models", params["run_name"]), exist_ok=True)
        torch.save(model.state_dict(), os.path.join("models", params["run_name"], f"ckpt.pt"))
        if epoch % 5 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", params["run_name"], f"{epoch}.jpg"))


def launch():
    params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 3e-4,
        'epochs': 50,
        'image_size': 32,
        'image_channels': 1,
        'run_name': 'default_run'
    }

    train(params)


if __name__ == '__main__':
    launch()