from utils import *
from model.models import SpeedModel_noVAE
from dataset import SpeedDataset

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler, BatchSampler, Subset
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

speeddataset = SpeedDataset('data', transform=ToTensor())
val_size = int(0.2*len(speeddataset))
test_dataset = Subset(speeddataset, list(range(0, val_size)))
train_dataset = Subset(speeddataset, list(range(val_size, len(speeddataset))))

batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size = 10, drop_last = False)
train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers = 8)

batch_sampler_test = BatchSampler(SequentialSampler(test_dataset), batch_size = 10, drop_last = False)
test_dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler_test, num_workers = 8)

device = 'cuda:0'
speedmodel = SpeedModel_noVAE(input_size=train_dataset[0][0].unsqueeze(0).shape)
speedmodel.load_state_dict(torch.load('pytorch_models/speedmodel_noVAE.pt'))
speedmodel = speedmodel.eval()
speedmodel = speedmodel.to(device)

vel_actual = []
vel_preds = []

with torch.no_grad():
    for i, (image_batch, label_batch) in enumerate(tqdm(train_dataloader)):
        vel_actual.append(label_batch)

        image_batch = image_batch.to(device)
        label_batch = label_batch.float().to(device)

        vel_pred = speedmodel(image_batch)
        vel_preds.append(vel_pred.detach().cpu())
#        if i == 40:
#            break

vel_preds = torch.cat(vel_preds).squeeze(0)
vel_actual = torch.cat(vel_actual).squeeze(0)

fig = plt.figure()
plt.plot(vel_preds, label="predicted")
plt.plot(vel_actual, label="actual")
plt.legend()
plt.savefig("train_vel_pred_noVAE.png")

vel_actual = []
vel_preds = []

with torch.no_grad():
    for i, (image_batch, label_batch) in enumerate(tqdm(test_dataloader)):
        vel_actual.append(label_batch)

        image_batch = image_batch.to(device)
        label_batch = label_batch.float().to(device)

        vel_pred = speedmodel(image_batch)
        vel_preds.append(vel_pred.cpu())

vel_preds = torch.cat(vel_preds).squeeze(0)
vel_actual = torch.cat(vel_actual).squeeze(0)
print(vel_preds.shape)
print(vel_actual.shape)

fig = plt.figure()
plt.plot(vel_preds, label="predicted")
plt.plot(vel_actual, label="actual")
plt.legend()
plt.savefig("test_vel_pred_noVAE.png")
