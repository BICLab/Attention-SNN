# train.py
#!/usr/bin/env	python3
import os
import sys
import argparse
import time
from datetime import datetime

# import apex
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader


def train(epoch, args):
    running_loss = 0
    start = time.time()
    net.train()
    correct = 0.0
    num_sample = 0
    for batch_index, (images, labels) in enumerate(ImageNet_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        num_sample += images.size()[0]
        optimizer.zero_grad()
        with autocast():
            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_iter = (epoch - 1) * len(ImageNet_training_loader) + batch_index + 1
        if batch_index % 10 == 9:
            print(
                "Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                    running_loss / 10,
                    optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(ImageNet_training_loader.dataset),
                )
            )
            print("training time consumed: {:.2f}s".format(time.time() - start))
            if args.local_rank == 0:
                writer.add_scalar("Train/avg_loss", running_loss / 10, n_iter)
                writer.add_scalar(
                    "Train/avg_loss_numpic", running_loss / 10, n_iter * args.b
                )
            running_loss = 0
    finish = time.time()
    if args.local_rank == 0:
        writer.add_scalar("Train/acc", correct / num_sample * 100, epoch)
    print(
        "Training accuracy: {:.2f} of epoch {}".format(
            correct / num_sample * 100, epoch
        )
    )
    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, args):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    real_batch = 0
    for (images, labels) in ImageNet_test_loader:
        real_batch += images.size()[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print("Evaluating Network.....")
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s".format(
            test_loss * args.b / len(ImageNet_test_loader.dataset),
            correct.float() / real_batch * 100,
            finish - start,
        )
    )

    if args.local_rank == 0:
        # add informations to tensorboard
        writer.add_scalar(
            "Test/Average loss",
            test_loss * args.b / len(ImageNet_test_loader.dataset),
            epoch,
        )
        writer.add_scalar("Test/Accuracy", correct.float() / real_batch * 100, epoch)

    return correct.float() / len(ImageNet_test_loader.dataset)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-gpu", action="store_true", default=True, help="use gpu or not"
    )
    parser.add_argument("-b", type=int, default=100, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--end_early", default=False, type=bool, help="end early")
    args = parser.parse_args()
    print(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    net = get_network(args)
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=True
    )

    # to load pretrain model
    map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
    #     net.load_state_dict(
    #         torch.load("/home/common/hyf/Xian/Resnet34/checkpoint/resnet34/600/0.6/ImageNet_T=1ACC/Monday_11_April_2022_10h_49m_47s/resnet34-50-regular.pth", map_location=map_location))

    num_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # data preprocessing:
    ImageNet_training_loader = get_training_dataloader(
        num_workers=16,
        batch_size=args.b // num_gpus,
        shuffle=False,
        sampler=1,  # to enable sampler for DDP
    )

    ImageNet_test_loader = get_test_dataloader(
        num_workers=16, batch_size=args.b // num_gpus, shuffle=False, sampler=1
    )
    # learning rate should go with batch size.
    b_lr = args.lr

    loss_function = CrossEntropyLabelSmooth()
    optimizer = optim.SGD(
        [{"params": net.parameters(), "initial_lr": b_lr}],
        momentum=0.9,
        lr=b_lr,
        weight_decay=1e-5,
    )  # SGD MOMENTUM
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0
    )
    iter_per_epoch = len(ImageNet_training_loader)
    LOG_INFO = "ImageNet_T=1ACC"
    checkpoint_path = os.path.join(
        settings.CHECKPOINT_PATH,
        args.net,
        str(args.b),
        str(args.lr),
        LOG_INFO,
        settings.TIME_NOW,
    )

    # use tensorboard
    if args.local_rank == 0:
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(
            log_dir=os.path.join(
                settings.LOG_DIR,
                args.net,
                str(args.b),
                str(args.lr),
                LOG_INFO,
                settings.TIME_NOW,
            )
        )

    # create checkpoint folder to save model
    if args.local_rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    #     for epoch in range(1, 51):
    #         train_scheduler.step()
    if args.end_early:
        exit()
    for epoch in range(1, settings.EPOCH + 1):
        train(epoch, args)

        train_scheduler.step()
        acc = eval_training(epoch, args)  # test

        # save model
        if epoch > (settings.EPOCH - 5) and best_acc < acc and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="best"),
            )
            best_acc = acc
            continue
        elif epoch >= (settings.EPOCH - 5) and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="regular"),
            )
            continue
        elif (not epoch % settings.SAVE_EPOCH) and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="regular"),
            )
            continue

    if args.local_rank == 0:
        writer.close()
