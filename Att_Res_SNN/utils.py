""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from conf import settings
from torch.utils.data.distributed import DistributedSampler


def get_network(args):
    """return given network"""
    if args.net == "resnet18":
        from models.resnet_2 import resnet18

        net = resnet18()
    elif args.net == "resnet34":
        from models.resnet_2 import resnet34

        net = resnet34()
    elif args.net == "resnet104":
        from models.resnet_2 import resnet104

        net = resnet104()
    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    return net


def get_training_dataloader(sampler=None, batch_size=16, num_workers=2, shuffle=True):
    traindir = "/gdata/ImageNet2012/train"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    ImageNet_training = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    if sampler is not None:
        ImageNet_training_loader = DataLoader(
            ImageNet_training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            sampler=DistributedSampler(ImageNet_training),
        )
    else:
        ImageNet_training_loader = DataLoader(
            ImageNet_training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
        )

    return ImageNet_training_loader


def get_test_dataloader(sampler=None, batch_size=16, num_workers=2, shuffle=False):
    valdir = "/gdata/ImageNet2012/val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    ImageNet_test = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),  # 320
                transforms.CenterCrop(224),  # 288
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    if sampler is not None:
        ImageNet_test_loader = DataLoader(
            ImageNet_test,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            sampler=DistributedSampler(ImageNet_test),
        )
    else:
        ImageNet_test_loader = DataLoader(
            ImageNet_test,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    return ImageNet_test_loader
