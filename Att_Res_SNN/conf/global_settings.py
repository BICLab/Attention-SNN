""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
CIFAR10_TRAIN_STD = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
#DATA_PATH = '/home/hyf/Desktop/CIFAR10/cifar100'
#total training epoches
EPOCH = 250
MID_EPOCH = 20
MILESTONES = [50, 70, 105, 125]

'''
EPOCH = 55
MILESTONES = [20, 35, 45]
'''
#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 5








