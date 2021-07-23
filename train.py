from utils import *
from model.models import SpeedModel
from dataset import SpeedDataset, RandomBatchSampler
from logger import Logger
 
import torch.optim as optim
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler, Subset
import sys
import numpy as np

try:
    speeddataset = SpeedDataset('data', transform=ToTensor())
    val_size = int(0.2*len(speeddataset))
    test_dataset = Subset(speeddataset, list(range(0, val_size)))
    train_dataset = Subset(speeddataset, list(range(val_size, len(speeddataset))))
    
    batch_sampler = RandomBatchSampler(SequentialSampler(train_dataset), batch_size = 2, drop_last = False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers = 8)

    batch_sampler_test = RandomBatchSampler(SequentialSampler(test_dataset), batch_size = 2, drop_last = False)
    test_dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler_test, num_workers = 8)

    latent_size = 1000
    vae_kwargs = {"latent_size": latent_size, "input_size": train_dataset[0][0].unsqueeze(0).shape}
    snail_kwargs = {"input_size": latent_size, "seq_length": 2}

    speedmodel = SpeedModel(snail_kwargs=snail_kwargs, vae_kwargs=vae_kwargs)
    print("Loading VAE Model")
    speedmodel.vae.load_state_dict(torch.load('pytorch_models/vae_model.pt'))

    device = 'cuda:1'
    optimizer = optim.Adam(speedmodel.snail.parameters(), lr = 1e-4)

    dojo = SpeedDojo()
    logger = Logger('SpeedModel')

    dojo.train(model=speedmodel, train_dataloader=train_dataloader, optimizer=optimizer, device=device, criteria=None, logger=logger, test_dataloader=test_dataloader)

except KeyboardInterrupt:
    print("Saving Speed model")
    torch.save(speedmodel.state_dict(), "pytorch_models/speedmodel.pt")
    sys.exit(0)
