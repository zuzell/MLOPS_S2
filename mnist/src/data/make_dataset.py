# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch


from torch.utils.data import  Dataset
from torchvision import transforms
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    print(os.getcwd())
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_paths = [input_filepath + f"/corruptmnist/train_{i}.npz" for i in range(5)]

    X_train = np.concatenate(
        [np.load(train_file)["images"] for train_file in train_paths]
    )
    Y_train = np.concatenate(
        [np.load(train_file)["labels"] for train_file in train_paths]
    )

    X_test = np.load(input_filepath + "/corruptmnist/test.npz")["images"]
    Y_test = np.load(input_filepath + "/corruptmnist/test.npz")["labels"]

    train = MNISTdata(X_train, Y_train, transform=transform)
    test = MNISTdata(X_test, Y_test, transform=transform)

    torch.save(train, output_filepath + "/train.pth")
    torch.save(test, output_filepath + "/test.pth")

class MNISTdata(Dataset):
    def __init__(self, data, targets, transform=None, additional_transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.additional_transform = additional_transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        if self.additional_transform:
            x = self.additional_transform(x)

        return x.float(), y

    def __len__(self):
        return len(self.data)

class MNISTdata(Dataset):
    def __init__(self, data, targets, transform=None, additional_transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.additional_transform = additional_transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        if self.additional_transform:
            x = self.additional_transform(x)

        return x.float(), y

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    main()
