from torch.utils.data import Dataset
import numpy as np 
import torch
from torch.utils.data import Sampler
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

class RandomBatchSampler(Sampler): # thank you name-yy@github
    r"""Yield a mini-batch of indices with random batch order

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral "
                             "value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fragment_size = len(data_source) % batch_size

    def __iter__(self):
        batch_indices = range(0, len(self.data_source) - self.fragment_size, self.batch_size)

        for batch_indices_idx in torch.randperm(len(self.data_source) // self.batch_size):
            yield list(range(batch_indices[batch_indices_idx], batch_indices[batch_indices_idx]+self.batch_size))

        if self.fragment_size > 0 and not self.drop_last:
            yield list(range(len(self.data_source) - self.fragment_size, len(self.data_source)))

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
#    trainset = SpeedDataset(root="data", transform=ToTensor(), train = True)
#    loader = DataLoader(trainset, batch_size=20, shuffle=False)
#    x,y = iter(loader).next()
#    print(x.shape)
#    print(y.shape)

    from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

    train_dataset = SpeedDataset('data', transform=ToTensor(), train=True)
    batch_sampler = RandomBatchSampler(SequentialSampler(train_dataset), batch_size = 20, drop_last = False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    while True:
        x,y = iter(train_dataloader).next()
        print(x.shape)
        print(y)
        input()


