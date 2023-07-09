import pandas as pd
import os

def save_csv(config):
    config.epoch_list.append(config.best_epoch)
    config.acc_test_list.append(config.best_acc)

    lists = [config.loss_train_list,
             config.loss_test_list,
             config.acc_train_list,
             config.acc_test_list]
    csv = pd.DataFrame(
        data=lists,
        index=['Train_Loss',
               'Test_Loss',
               'Train_Accuracy',
               'Test_Accuracy'],
        columns=config.epoch_list)
    csv.index.name = 'Epochs'

    if not os.path.exists(config.recordPath):
        os.makedirs(config.recordPath)
    csv.to_csv(config.recordPath + os.sep + config.recordNames)