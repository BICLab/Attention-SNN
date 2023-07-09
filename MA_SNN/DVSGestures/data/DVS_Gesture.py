import os
import sys

rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import tarfile
import os
import h5py
import numpy as np
import struct
from DVSGestures.DVS_gesture_data_process.events_timeslices import *




def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)




def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    import glob
    fns = []
    for i in range(start_id, end_id):
        search_mask = directory + os.sep + \
                      filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns





def aedat_to_events(filename):

    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename,
                        skiprows=1,
                        delimiter=',',
                        dtype='uint32')

    events = []
    with open(filename, 'rb') as f:


        for i in range(5):
            _ = f.readline()

        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize),
                                            'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)

    events = np.column_stack(events)
    events = events.astype('uint32')

    clipped_events = np.zeros([4, 0], 'uint32')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events,
                                          events[:, start:end]])

    return clipped_events.T, labels





def create_hdf5(path, save_path):
    # print('path', path, 'save_path', save_path)

    # Train
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'train')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)

    fns_train = gather_aedat(path, 1, 24)
    assert len(fns_train) == 98

    for i in range(len(fns_train)):
        print(str(i + 1))
        data, labels_starttime = aedat_to_events(fns_train[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]

        for lbls_idx in range(len(lbls)):
            print(lbls_idx)
            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            times = s_[0]
            addrs = s_[1]
            with h5py.File(save_path_train + os.sep + 'DVS-Gesture-train' + str(i * 12 + lbls_idx + 1) + '.hdf5',
                           'w') as f:
                tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                lbl_dset = f.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)

    print('train finished')

    # Test
    print('processing test data...')
    save_path_test = os.path.join(save_path, 'test')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    fns_test = gather_aedat(path, 24, 30)
    assert len(fns_test) == 24

    for i in range(len(fns_test)):
        print(str(i + 1))
        data, labels_starttime = aedat_to_events(fns_test[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]

        for lbls_idx in range(len(lbls)):
            print(lbls_idx)
            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            times = s_[0]
            addrs = s_[1]
            with h5py.File(save_path_test + os.sep + 'DVS-Gesture-test' + str(i * 12 + lbls_idx + 1) + '.hdf5',
                           'w') as f:
                tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                lbl_dset = f.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)

        assert lbls_idx == 11

    print('test finished')


def datasets_process(path=None):
    if not os.path.exists(path):
        print(path + '   not exists')

    elif not os.path.isfile(os.path.join(path, 'DvsGesture.tar.gz')):
        print(path)
        print('DvsGesture.tar.gz  not exists')

    else:
        print('DVS-Gestures')
        untar(os.path.join(path, 'DvsGesture.tar.gz'), path)
        create_hdf5(os.path.join(path, 'DvsGesture'), path)


if __name__ == '__main__':
    path = './'
    datasets_process(path=path)
