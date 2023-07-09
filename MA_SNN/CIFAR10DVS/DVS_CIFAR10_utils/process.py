import time
from CIFAR10DVS.DVS_CIFAR10_utils.train import train
from CIFAR10DVS.DVS_CIFAR10_utils.test import test
import torch
from utils import util
import os

def process(config):
    config.best_acc = 0
    config.best_epoch = 0

    config.epoch_list = []
    config.loss_train_list = []
    config.loss_test_list = []
    config.acc_train_list = []
    config.acc_test_list = []

    if config.pretrained_path != None:
        pre_dict = torch.load(config.pretrained_path)['net']
        model_dict = config.model.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pre_dict)
        config.model.load_state_dict(model_dict)
        print('loading model...')

    for config.epoch in range(config.num_epochs):
        if config.onlyTest == False:
            # train

            config.model.train()
            train(config=config)


            config.train_loss = config.train_loss / len(config.train_loader)
            config.epoch_list.append(config.epoch + 1)
            config.train_acc = 100. * \
                            float(config.train_correct) / float(len(config.train_dataset))
            print('T:', config.dt)
            print('epoch:', config.epoch + 1)
            print('Train datasets total:', len(config.train_dataset))
            print('Tarin loss:%.5f' % config.train_loss)
            print('Train acc: %.3f' % config.train_acc)
            if config.lr_scheduler:

                config.scheduler.step(config.train_loss)
            config.loss_train_list.append(config.train_loss)
            config.acc_train_list.append(config.train_acc)

        # test
        with torch.no_grad():


            config.model.eval()

            test(config=config)


            config.test_loss = config.test_loss / len(config.test_loader)
            config.test_acc = 100. * float(config.test_correct) / \
                              float(len(config.test_dataset))
            config.loss_test_list.append(config.test_loss)

            print('Test datasets total:', len(config.test_dataset))
            print('Test loss:%.5f' % config.test_loss)
            print('Test acc: %.3f' % config.test_acc)

            config.acc_test_list.append(config.test_acc)

            if config.test_acc > config.best_acc:
                config.best_epoch = config.epoch + 1
                config.best_acc = config.test_acc

                print('Saving..')
                config.state = {'net': config.model.state_dict(),
                                'acc': config.test_acc,
                                'epoch': (config.epoch + 1),
                                'acc_record': config.acc_test_list,
                                }

                if not os.path.exists(config.modelPath):
                    os.makedirs(config.modelPath)
                torch.save(
                    config.state,
                    config.modelPath + os.sep + config.modelNames,
                    _use_new_zipfile_serialization=False,
                )
            print('beat acc:', config.best_acc, 'beat epoch:', config.best_epoch)