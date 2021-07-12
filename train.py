from utils import *
from model.models import SpeedModel
from dataset import SpeedDataset, RandomBatchSampler
from logger import Logger
 
import torch.optim as optim
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler
import sys

try:
    train_dataset = SpeedDataset('data', transform=ToTensor(), train=True)
    batch_sampler = RandomBatchSampler(SequentialSampler(train_dataset), batch_size = 10, drop_last = False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers = 8)

    latent_size = 1000
    vae_kwargs = {"latent_size": latent_size, "input_size": train_dataset[0][0].unsqueeze(0).shape}
    snail_kwargs = {"input_size": latent_size, "seq_length": 10}
    speedmodel = SpeedModel(snail_kwargs=snail_kwargs, vae_kwargs=vae_kwargs)

    device = 'cuda:1'
    optimizer = optim.Adam(speedmodel.parameters(), lr = 1e-4)

    dojo = SpeedDojo()
    logger = Logger('SppedModel')

    dojo.train(model=speedmodel, dataloader=train_dataloader, optimizer=optimizer, device=device, criteria=None, logger=logger)

except KeyboardInterrupt:
    print("Saving Speed model")
    torch.save(speedmodel.state_dict(), "pytorch_models/speedmodel.pt")
    sys.exit(0)