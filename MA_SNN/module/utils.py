import os
import math
import random
import numpy as np
import torch


def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def str2bool(v):
    if v == 'True':
        return True
    else:
        return False


def global_avgpool2d(x):
    batch_size = x.shape[0]
    channel_size = x.shape[1]
    return x.reshape(batch_size, channel_size, -1).mean(dim=2)


def winner_take_all(x, sparsity_ratio):
    k = math.ceil(sparsity_ratio * x.shape[1])
    winner_idx = x.topk(k, 1)[1]
    winner_mask = torch.zeros_like(x)
    winner_mask.scatter_(1, winner_idx, 1)
    x = x * winner_mask

    return x, winner_mask