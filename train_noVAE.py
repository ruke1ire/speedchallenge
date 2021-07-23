from utils import *
from model.models import SpeedModel_noVAE
from dataset import SpeedDataset
from logger import Logger
 
import torch.optim as optim
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler, Subset, RandomSampler, BatchSampler
import sys
import numpy as np

try:
    speeddataset = SpeedDataset('data', transform=ToTensor())
    val_size = int(0.2*len(speeddataset))
    test_dataset = Subset(speeddataset, list(range(0, val_size)))
    train_dataset = Subset(speeddataset, list(range(val_size, len(speeddataset))))
    
    batch_sampler = BatchSampler(RandomSampler(train_dataset), batch_size = 40, drop_last = False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers = 4)

    batch_sampler_test = BatchSampler(RandomSampler(test_dataset), batch_size = 100, drop_last = False)
    test_dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler_test, num_workers = 0)

    speedmodel = SpeedModel_noVAE(input_size=train_dataset[0][0].unsqueeze(0).shape)

    device = 'cuda:1'
    optimizer = optim.Adam(speedmodel.parameters(), lr = 1e-4)

    dojo = SpeedDojo_noVAE()
    logger = Logger('SpeedModel_noVAE')

    dojo.train(model=speedmodel, train_dataloader=train_dataloader, optimizer=optimizer, device=device, criteria=None, logger=logger, test_dataloader=test_dataloader)

except KeyboardInterrupt:
    print("Saving Speed model")
    torch.save(speedmodel.state_dict(), "pytorch_models/speedmodel_noVAE.pt")
    sys.exit(0)
