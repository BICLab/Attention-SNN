import torch
from utils import util
from DVSGait.DVS_Gait_utils.dataset import create_data
from DVSGait.CNN.Networks.Att_SNN import create_net
from DVSGait.CNN.Config import configs
from DVSGait.DVS_Gait_utils.process import process
from DVSGait.DVS_Gait_utils.save import save_csv


def main():

    config = configs()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config.device)

    config.device_ids = range(torch.cuda.device_count())
    print(config.device_ids)

    config.name = (
        config.attention
        + "_SNN(CNN)-DVS-Gesture_dt="
        + str(config.dt)
        + "ms"
        + "_T="
        + str(config.T)
    )
    config.modelNames = config.name + ".t7"
    config.recordNames = config.name + ".csv"

    print(config)

    create_data(config=config)

    create_net(config=config)

    print(config.model)

    print(util.get_parameter_number(config.model))

    process(config=config)

    print("best acc:", config.best_acc, "best_epoch:", config.best_epoch)

    save_csv(config=config)
