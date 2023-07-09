"""
@deprecated DVS128 Gait Dataset
@author Huanhuan Gao <gaohuanh@gmail.com>
"""
import math
import os, bisect, sys
from DVSGait.DVS_gait_data_process.transforms import *
import os
from torch.utils.data import Dataset
from DVSGait.DVS_gait_data_process.events_timeslices import *
import scipy.io as scio
import time

rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
import numpy as np
from torch.utils.data import Dataset
import random


class DVS128GaitDataset_average_distribution(Dataset):
    def __init__(self,
                 path,
                 T=10,
                 train=True,
                 is_spike=False,
                 ds=None,
                 ):
        super(DVS128GaitDataset_average_distribution, self).__init__()
        if ds is None:
            ds = [1, 1]
        self.train = train
        self.T = T
        self.is_spike = is_spike
        self.ds = ds

        npy_path = os.path.join(path, 'npy')
        if not os.path.exists(npy_path):
            DVS128_Gait_txt_to_npy(path)

        if self.train:
            train_npy_path = os.path.join(npy_path, 'train')
            self.train_data = np.load(os.path.join(train_npy_path, 'train_data.npy'), allow_pickle=True)
            self.train_target = np.load(os.path.join(train_npy_path, 'train_target.npy'), allow_pickle=True)

        else:
            test_npy_path = os.path.join(npy_path, 'test')
            self.test_data = np.load(os.path.join(test_npy_path, 'test_data.npy'), allow_pickle=True)
            self.test_target = np.load(os.path.join(test_npy_path, 'test_target.npy'), allow_pickle=True)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            data = self.train_data[idx]
            data[:, 0] -= data[0, 0]
            dt = math.ceil((data[-1][0] - data[0][0]) / self.T)
            data = my_chunk_evs_pol_dvs(data=data,
                                        dt=dt,
                                        T=self.T,
                                        ds=self.ds)
            if self.is_spike:
                data = np.int64(data > 0)

            target_idx = self.train_target[idx]
            label = np.zeros((21))
            label[target_idx] = 1.0

            return data, label

        else:

            data = self.test_data[idx]
            data[:, 0] -= data[0, 0]
            dt = math.ceil((data[-1][0] - data[0][0]) / self.T)
            data = my_chunk_evs_pol_dvs(data=data,
                                        dt=dt,
                                        T=self.T,
                                        ds=self.ds)
            data = np.expand_dims(data,0)
            target_idx = self.test_target[idx]
            label = np.zeros((21))
            label[target_idx] = 1.0
            label = np.expand_dims(label, 0)
            return data, label



class DVS128GaitDataset(Dataset):
    def __init__(self,
                 path,
                 dt=1000,
                 T=10,
                 train=True,
                 is_train_Enhanced=False,
                 clips=1,
                 is_spike=False,
                 ds=None,
                 ):
        super(DVS128GaitDataset, self).__init__()
        if ds is None:
            ds = [1, 1]
        self.train = train
        self.dt = dt
        self.T = T
        self.is_train_Enhanced = is_train_Enhanced
        self.clips = clips
        self.is_spike = is_spike
        self.ds = ds

        npy_path = os.path.join(path, 'npy')
        if not os.path.exists(npy_path):
            DVS128_Gait_txt_to_npy(path)

        if self.train:
            train_npy_path = os.path.join(npy_path, 'train')
            self.train_data = np.load(os.path.join(train_npy_path, 'train_data.npy'), allow_pickle=True)
            self.train_target = np.load(os.path.join(train_npy_path, 'train_target.npy'), allow_pickle=True)

        else:
            test_npy_path = os.path.join(npy_path, 'test')
            self.test_data = np.load(os.path.join(test_npy_path, 'test_data.npy'), allow_pickle=True)
            self.test_target = np.load(os.path.join(test_npy_path, 'test_target.npy'), allow_pickle=True)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            data = self.train_data[idx]

            data = sample_train(
                data=data,
                dt=self.dt,
                T=self.T,
                is_train_Enhanced=self.is_train_Enhanced,
            )

            data,ts = my_chunk_evs_pol_dvs(data=data,
                                        dt=self.dt,
                                        T=self.T,
                                        ds=self.ds)
            if self.is_spike:
                data = np.int64(data > 0)

            target_idx = self.train_target[idx]
            label = np.zeros((21))
            label[target_idx] = 1.0

            return data, label

        else:

            data = self.test_data[idx]

            data = sample_test(
                data=data,
                dt=self.dt,
                T=self.T,
                clips=self.clips,
            )

            target_idx = self.test_target[idx]
            label = np.zeros((21))
            label[target_idx] = 1.0

            data_temp = []
            target_temp = []
            for i in range(self.clips):

                temp,_ = my_chunk_evs_pol_dvs(data=data[i],
                                            dt=self.dt,
                                            T=self.T,
                                            ds=self.ds)

                if self.is_spike:
                    temp = np.int64(temp > 0)

                data_temp.append(temp)

                target_temp.append(label)

            data = np.array(data_temp)
            target = np.array(target_temp)

            return data, target


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def my_chunk_evs_pol_dvs(data, dt=1000, T=500, ds=[1, 1]):
    t_start = data[0][0]
    ts = range(t_start, t_start + T * dt, dt)
    chunks = np.zeros([len(ts)] + [2] + [int(128 / ds[0])] + [int(128 / ds[1])], dtype='int64')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:, 0], t + dt)
        if idx_end > idx_start:
            ee = data[idx_start:idx_end, 1:]
            pol, x, y = ee[:, 2], (ee[:, 0] // ds[0]).astype(np.int), (ee[:, 1] // ds[1]).astype(np.int)
            np.add.at(chunks, (i, pol, x, y), 1)
        idx_start = idx_end
    return chunks,ts


def get_tmad_slice(times, addrs, start_time, T):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time + T) + idx_beg
        return np.column_stack([times[idx_beg:idx_end], addrs[idx_beg:idx_end]])
    except IndexError:
        raise IndexError("Empty batch found")


def sample_train(data,
                 T=60,
                 dt=1000,
                 is_train_Enhanced=False
                 ):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1] - T * dt)

    start_time = random.randint(tbegin, tend) if is_train_Enhanced else tbegin

    tmad = get_tmad_slice(data[:, 0],
                          data[:, 1:4],
                          start_time,
                          T * dt)

    # tmad[:, 0] -= tmad[0, 0]
    return tmad
    # return tmad[:, [0, 3, 1, 2]]


def sample_test(data,
                T=60,
                clips=10,
                dt=1000
                ):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1])

    tmad = get_tmad_slice(data[:, 0],
                          data[:, 1:4],
                          tbegin,
                          tend - tbegin)
    # 初试从零开始
    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clips * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clips * T * dt - (end_time - start_time)) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clips * T * dt) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg
        temp.append(tmad[idx_beg:idx_end])

    return temp


def DVS128_Gait_txt_to_npy(path):
    save_path = os.path.join(path, 'npy')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train
    train_path = os.path.join(path, 'train')
    save_train_path = os.path.join(save_path, 'train')

    if not os.path.exists(save_train_path):
        os.makedirs(save_train_path)

    for root, dirs, files in os.walk(train_path):
        break

    print("root", root)  # 当前目录路径
    print("dirs", dirs)  # 当前路径下所有子目录

    train_data = []
    train_target = []
    for dir in dirs:
        train_path_sub = os.path.join(root, dir)
        files = os.listdir(train_path_sub)
        print(dir)
        for file in files:
            print(file)
            train_path_file = os.path.join(train_path_sub, file)
            data = []
            with open(train_path_file) as f:
                for line in f.readlines():
                    temp = line.split()
                    data.append(np.array(temp).astype(np.int64))

            train_data.append(np.array(data))
            train_target.append(int(dir))

    save_train_path_file_data = os.path.join(save_train_path, 'train_data.npy')
    save_train_path_file_target = os.path.join(save_train_path, 'train_target.npy')

    np.save(save_train_path_file_data, np.array(train_data))
    np.save(save_train_path_file_target, np.array(train_target))

    # test
    test_path = os.path.join(path, 'test')
    save_test_path = os.path.join(save_path, 'test')

    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)

    for root, dirs, files in os.walk(test_path):
        break

    print("root", root)  # 当前目录路径
    print("dirs", dirs)  # 当前路径下所有子目录

    test_data = []
    test_target = []
    for dir in dirs:
        test_path_sub = os.path.join(root, dir)
        files = os.listdir(test_path_sub)
        print(dir)
        for file in files:
            print(file)
            test_path_file = os.path.join(test_path_sub, file)
            data = []
            with open(test_path_file) as f:
                for line in f.readlines():
                    temp = line.split()
                    data.append(np.array(temp).astype(np.int64))

            test_data.append(np.array(data))
            test_target.append(int(dir))

    save_test_path_file_data = os.path.join(save_test_path, 'test_data.npy')
    save_test_path_file_target = os.path.join(save_test_path, 'test_target.npy')

    np.save(save_test_path_file_data, np.array(test_data))
    np.save(save_test_path_file_target, np.array(test_target))


def test():
    import torch, time
    path = '/home/huanhuan.gao/data/DVS128_Gait'
    batch_size = 64

    train_dataset = DVS128GaitDataset(path,
                                      train=True,
                                      dt=1000 * 20,
                                      T=60,
                                      )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=4,
                                               pin_memory=False)

    test_dataset = DVS128GaitDataset(path,
                                     train=False,
                                     dt=1000 * 20,
                                     T=60,
                                     clips=1,
                                     )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=4,
                                              pin_memory=False)

    i = 1
    start_time = time.time()
    for idx, (input, labels) in enumerate(train_loader):
        print(i)
        i += 1
    print(time.time() - start_time)
    start_time = time.time()
    i = 1
    for idx, (input, labels) in enumerate(test_loader):
        print(i)
        i += 1
    print(time.time() - start_time)
    start_time = time.time()


def test_DVS128GaitDataset_average_distribution():
    import torch, time
    path = './CVPR2019/DVS128_Gait'
    batch_size = 64

    train_dataset = DVS128GaitDataset_average_distribution(path,
                                                           T=60,
                                                           )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0,
                                               pin_memory=False)

    test_dataset = DVS128GaitDataset_average_distribution(path,
                                                          train=False,
                                                          T=60,
                                                          )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              pin_memory=False)

    i = 1
    start_time = time.time()
    for idx, (input, labels) in enumerate(train_loader):
        print(i)
        i += 1
    print(time.time() - start_time)
    start_time = time.time()
    i = 1
    for idx, (input, labels) in enumerate(test_loader):
        print(i)
        i += 1
    print(time.time() - start_time)
    start_time = time.time()

def create_datasets(root=None,
                    train=True,
                    chunk_size_train=60,
                    chunk_size_test=60,
                    ds=4,
                    dt=1000,
                    transform_train=None,
                    transform_test=None,
                    target_transform_train=None,
                    target_transform_test=None,
                    n_events_attention=None,
                    clip=10,
                    is_train_Enhanced=False,
                    ):
    if isinstance(ds, int):
        ds = [ds, ds]

    size = [2, 128 // ds[0], 128 // ds[1]]

    if n_events_attention is None:
        def default_transform():
            return Compose([
                ToTensor()
            ])
    else:
        def default_transform():
            return Compose([
                ToTensor()
            ])

    if transform_train is None:
        transform_train = default_transform()
    if transform_test is None:
        transform_test = default_transform()

    if target_transform_train is None:
        target_transform_train = Compose(
            [Repeat(chunk_size_train), toOneHot(10)])
    if target_transform_test is None:
        target_transform_test = Compose(
            [Repeat(chunk_size_test), toOneHot(10)])

    if train:

        train_d = DVS128GaitDataset(root,
                                      train=train,
                                      dt=dt,
                                      T=chunk_size_train,
                                      is_train_Enhanced=is_train_Enhanced,
                                      ds=ds)
        print(len(train_d))
        return train_d
    else:
        test_d = DVS128GaitDataset(root,
                                     train=train,
                                     dt=dt,
                                     T=chunk_size_test,
                                     ds=ds)
        return test_d


if __name__ == '__main__':
    # test()
    test_DVS128GaitDataset_average_distribution()
