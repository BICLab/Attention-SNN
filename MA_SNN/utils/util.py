from torch import nn


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    :param lr_decay_epoch:
    :param init_lr:
    :param epoch:
    :type optimizer: object
    """

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# init method
def paramInit(model, method='xavier'):
    scale = 0.05
    for name, w in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
                w *= scale
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass
#######################################################
