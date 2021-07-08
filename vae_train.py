from logger import Logger
from model.models import *
from dataset import SpeedDataset

import torch
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

import sys

device = 'cuda:0'

dataset = SpeedDataset(root="data", transform=ToTensor(), train = True)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

vae_model = VAE(latent_size = 1000, input_size = dataset[0][0].unsqueeze(0).shape).to(device)
optimizer = optim.Adam(vae_model.parameters(), lr = 1e-4)

logger = Logger("VAE")

def main_loop():
    epoch = 1
    while True:
        for batch_no, (images, labels) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            recon, mu, logvar = vae_model(images)
            loss = vae_loss_function(recon, images, mu, logvar)

            loss.backward()
            optimizer.step()

            print(f"[EPOCH {epoch}][BATCH {batch_no}][LOSS {loss.item()}]")

            logger.log_image("Reconstruction", recon[0].detach().cpu(), episode = epoch)
            logger.log_image("Original", images[0].detach().cpu(), episode = epoch)

        epoch += 1

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
try:
    main_loop()
except KeyboardInterrupt:
    print("Saving VAE model")
    torch.save(vae_model.state_dict(), "pytorch_models/vae_model.pt")
    sys.exit(0)
