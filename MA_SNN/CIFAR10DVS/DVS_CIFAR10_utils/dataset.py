import torch
from CIFAR10DVS.DVS_CIFAR10_data_process.DVS_CIFAR10_dataloaders import create_datasets
from torch.utils.data import ConcatDataset

def create_data(config):
    # Data set
    if isinstance(config.dt, int):
        if config.onlyTest == False:
            config.train_dataset = create_datasets(
                config.savePath,
                train=True,
                is_train_Enhanced=config.is_train_Enhanced,
                ds=config.ds,
                dt=config.dt * 1000,
                chunk_size_train=config.T,
            )

        config.test_dataset = create_datasets(
            config.savePath,
            train=False,
            ds=config.ds,
            dt=config.dt * 1000,
            chunk_size_test=config.T,
            clip=config.clip
        )
    else:
        train_list = []
        test_list = []
        for dt in config.dt:
            train_list.append(create_datasets(
            config.savePath,
            train=True,
            is_train_Enhanced=config.is_train_Enhanced,
            ds=config.ds,
            dt=dt * 1000,
            chunk_size_train=config.T,
            ))
            test_list.append(create_datasets(
            config.savePath,
            train=False,
            ds=config.ds,
            dt=dt * 1000,
            chunk_size_test=config.T,
            clip=config.clip
            ))
        config.train_dataset = ConcatDataset(train_list)
        config.test_dataset = ConcatDataset(test_list)
    # Data loader
    if config.onlyTest == False:
        config.train_loader = torch.utils.data.DataLoader(
            config.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            pin_memory=config.pip_memory,
            num_workers = config.num_workers,

        )
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        pin_memory=config.pip_memory,
        num_workers=config.num_workers,
    )
