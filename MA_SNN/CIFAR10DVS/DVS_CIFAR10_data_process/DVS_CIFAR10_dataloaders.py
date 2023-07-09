from CIFAR10DVS.DVS_CIFAR10_data_process.transforms import *
import os
from torch.utils.data import Dataset
from CIFAR10DVS.DVS_CIFAR10_data_process.events_timeslices import *
import scipy.io as scio
import time
mapping = {0: 'airplane',
           1: 'automobile',
           2: 'bird',
           3: 'cat',
           4: 'deer',
           5: 'dog',
           6: 'frog',
           7: 'horse',
           8: 'ship',
           9: 'truck'}


class DVS_CIFAR10_Dataset(Dataset):
    def __init__(self,
                 root,
                 train=False,
                 transform=None,
                 target_transform=None,
                 chunk_size=500,
                 clip=10,
                 is_train_Enhanced=False,
                 dt=1000,
                 size=[2, 128, 128],
                 ds=4
                 ):
        super(DVS_CIFAR10_Dataset, self).__init__()

        self.n = 0
        self.root = root
        self.train = train
        self.chunk_size = chunk_size
        self.clip = clip
        self.is_train_Enhanced = is_train_Enhanced
        self.dt = dt
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.ds = ds

        if train:
            self.class_dir_train = os.listdir(self.root)
            self.n = int(len(self.class_dir_train) * 0.9 * 1000)
        else:
            self.class_dir_test = os.listdir(self.root)
            self.n = int(len(self.class_dir_test) * 0.1 * 1000)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Important to open and close in getitem to enable num_workers>0

        if self.train:

            assert idx < 9000
            class_id = idx // 900
            idx_id = idx % 900

            root_train = os.path.join(self.root, self.class_dir_train[class_id])
            matdata_dir = os.path.join(root_train, str(idx_id))
            data = scio.loadmat(matdata_dir)
            data = pd.DataFrame(data['out'])
            data = data.values
            data = np.delete(data, [1, 2], axis = 1)
            target = class_id
            data = sample_train(data,
                                T=self.chunk_size,
                                dt=self.dt,
                                is_train_Enhanced=self.is_train_Enhanced)

            data = my_chunk_evs_pol_dvs(data=data,
                                        dt=self.dt,
                                        T=self.chunk_size,
                                        size=self.size,
                                        ds=self.ds)

            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target
        else:
            assert idx < 1000
            class_id = idx // 100
            idx_id = idx % 100 + 900

            root_test = os.path.join(self.root, self.class_dir_test[class_id])
            matdata_dir = os.path.join(root_test, str(idx_id))
            data = scio.loadmat(matdata_dir)
            data = pd.DataFrame(data['out'])
            data = data.values
            data = np.delete(data, [1, 2], axis = 1)
            target = class_id

            data = sample_test(data,
                               T=self.chunk_size,
                               clip=self.clip,
                               dt=self.dt)

            data_temp = []
            target_temp = []
            for i in range(self.clip):

                if self.transform is not None:
                    temp = my_chunk_evs_pol_dvs(data=data[i],
                                                dt=self.dt,
                                                T=self.chunk_size,
                                                size=self.size,
                                                ds=self.ds)

                    data_temp.append(self.transform(temp))

                if self.target_transform is not None:
                    target_temp.append(self.target_transform(target))

            data = torch.stack(data_temp)
            target = torch.stack(target_temp)

            return data, target


def sample_train(data,
                 T=60,
                 dt=1000,
                 is_train_Enhanced=False
                 ):
    try:
        np.random.seed(int(time.time()*1000)%2**32)
        tbegin = data[:, 0][0]
        tend = np.maximum(0, data[:, 0][-1] - T * dt)

        start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0
        tmad = get_tmad_slice(data[:, 0],
                            data[:, 1:4],
                            start_time,
                            T * dt)
        tmad[:, 0] -= tmad[0, 0]
    except:
        pass
    return tmad


def sample_test(data,
                T=60,
                clip=10,
                dt=1000
                ):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1])

    tmad = get_tmad_slice(data[:, 0],
                          data[:, 1:4],
                          tbegin,
                          tend - tbegin)

    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clip * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clip * T * dt - (end_time - start_time)) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clip * T * dt) / clip))
        for j in range(clip):
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

        train_d = DVS_CIFAR10_Dataset(root,
                                      train=train,
                                      transform=transform_train,
                                      target_transform=target_transform_train,
                                      chunk_size=chunk_size_train,
                                      is_train_Enhanced=is_train_Enhanced,
                                      dt=dt,
                                      size=size,
                                      ds=ds)
        return train_d
    else:
        test_d = DVS_CIFAR10_Dataset(root,
                                     transform=transform_test,
                                     target_transform=target_transform_test,
                                     train=train,
                                     chunk_size=chunk_size_test,
                                     clip=clip,
                                     dt=dt,
                                     size=size,
                                     ds=ds)
        return test_d



