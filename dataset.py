from torch.utils.data import Dataset
import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import ToPILImage, ToTensor
import os

class SpeedDataset(Dataset):
    def __init__(self, root, transform, train=True):
        train_path = os.path.join(root, 'train')
        test_path = os.path.join(root, 'test')
        self.train_label_path = os.path.join(root, 'train.txt')
        if train == True:
            self.data = ImageFolder(train_path, transform=transform)
        elif train == False:
            self.data = ImageFolder(test_path, transform=transform)
        else:
            raise Exception(f"invalid train value train={train}")

        self.train = train
    def __getitem__(self, idx):
        if self.train:
            images, _ = self.data[idx]
            labels = torch.tensor(np.loadtxt(self.train_label_path))[idx]
            return (images, labels)
        else:
            return self.data[idx]    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    trainset = SpeedDataset(root="data", transform=ToTensor(), train = True)
    loader = DataLoader(trainset, batch_size=20, shuffle=False)
    x,y = iter(loader).next()
    print(x.shape)
    print(y.shape)
