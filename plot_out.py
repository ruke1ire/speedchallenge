from utils import *
from model.models import SpeedModel
from dataset import SpeedDataset
 
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler, BatchSampler, Subset
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

speeddataset = SpeedDataset('data', transform=ToTensor())
val_size = int(0.2*len(speeddataset))
train_dataset = Subset(speeddataset, list(range(0, len(speeddataset)-val_size)))
test_dataset = Subset(speeddataset, list(range(len(speeddataset)-val_size, len(speeddataset))))

batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size = 10, drop_last = False)
train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers = 4)

batch_sampler_test = BatchSampler(SequentialSampler(test_dataset), batch_size = 10, drop_last = False)
test_dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler_test, num_workers = 4)

device = 'cuda:1'
latent_size = 1000
vae_kwargs = {"latent_size": latent_size, "input_size": train_dataset[0][0].unsqueeze(0).shape}
snail_kwargs = {"input_size": latent_size, "seq_length": 10}
speedmodel = SpeedModel(snail_kwargs=snail_kwargs, vae_kwargs=vae_kwargs)
speedmodel.load_state_dict(torch.load('pytorch_models/speedmodel.pt'))
speedmodel = speedmodel.eval()
speedmodel = speedmodel.to(device)

vel_actual = []
vel_preds = []

with torch.no_grad():
    for i, (image_batch, label_batch) in enumerate(tqdm(train_dataloader)):
        vel_actual.append(label_batch)

        image_batch = image_batch.to(device)
        label_batch = label_batch.float().to(device)

        vel_pred, vae_out = speedmodel(image_batch)
        vel_preds.append(vel_pred.cpu())
        if i == 100:
            break

vel_preds = torch.cat(vel_preds).squeeze(0)
vel_actual = torch.cat(vel_actual).squeeze(0)

fig = plt.figure()
plt.plot(vel_preds, label="predicted")
plt.plot(vel_actual, label="actual")
plt.legend()
plt.savefig("train_vel_pred.png")

vel_actual = []
vel_preds = []

with torch.no_grad():
    for i, (image_batch, label_batch) in enumerate(tqdm(test_dataloader)):
        vel_actual.append(label_batch)

        image_batch = image_batch.to(device)
        label_batch = label_batch.float().to(device)

        vel_pred, vae_out = speedmodel(image_batch)
        vel_preds.append(vel_pred.cpu())

vel_preds = torch.cat(vel_preds).squeeze(0)
vel_actual = torch.cat(vel_actual).squeeze(0)
print(vel_preds.shape)
print(vel_actual.shape)

fig = plt.figure()
plt.plot(vel_preds, label="predicted")
plt.plot(vel_actual, label="actual")
plt.legend()
plt.savefig("test_vel_pred.png")
