import torch
from DVSGestures.DVS_gesture_data_process.DVS_Gesture_dataloders import create_datasets


def create_data(config):
    # Data set
    if config.onlyTest == False:
        config.train_dataset = create_datasets(
            config.savePath,
            train=True,
            is_train_Enhanced=config.is_train_Enhanced,
            ds=config.ds,
            dt=config.dt * 1000,
            chunk_size_train=config.T,
            is_spike=config.is_spike,
            interval_scaling = config.interval_scaling,
        )

    config.test_dataset = create_datasets(
        config.savePath,
        train=False,
        ds=config.ds,
        dt=config.dt * 1000,
        chunk_size_test=config.T,
        clip=config.clip,
        is_spike=config.is_spike,
        interval_scaling = config.interval_scaling,
    )
    # Data loader
    if config.onlyTest == False:
        config.train_loader = torch.utils.data.DataLoader(
            config.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            num_workers=config.num_work,
            pin_memory=config.pip_memory)
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory)
