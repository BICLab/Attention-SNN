import time, h5py, os
import numpy as np

def noise(
        data_path,
        save_path,
        T = 10000,
):

    file_list = []
    datanames = os.listdir(data_path)
    for i in datanames:
        if os.path.splitext(i)[1] == '.hdf5':
            file_list.append(i)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = 0
    rand_time = 0
    time_val = 0


    # print("file_list",file_list[0])
    for list_num in file_list:
        # print("list_num",list_num)
        data_addrs = []
        data_labels_processing = []
        data_times = []

        start = time.time()
        f = h5py.File(os.path.join(data_path,list_num), 'r')

        print(list_num)

        daddrs = f['addrs']
        dlabels = f['labels']
        dtimes = f['times']

        data_labels = dlabels[()]
        data_labels_processing.append(data_labels)

        print(data_labels_processing)
        print('data_labels:', str(data_labels))
        print('dtime', dtimes[-1])

        pic_num = 0
        t_last = dtimes[0 + rand_time]

        data = np.column_stack([dtimes, daddrs])

        for i in range(len(dtimes)):

            xaddr = daddrs[i + rand_time - num * time_val][1]
            yaddr = daddrs[i + rand_time - num * time_val][0]
            paddr = daddrs[i + rand_time - num * time_val][2]
            t = dtimes[i + rand_time - num * time_val]
            index = np.where((data[:, 0] >=t) & (data[:, 0]<=t+1000))
            result = np.where(
                            (data[index, 1] >= yaddr - 1) & (
                            data[index, 1] <= yaddr + 1) & (
                        data[index, 2] >= xaddr - 1) & (data[index, 2] <= xaddr + 1))

            if result[0].size > 1:
                data_addrs.append([yaddr, xaddr, paddr])
                data_times.append([t])


        f = h5py.File(os.path.join(save_path, list_num), "w")
        f["addrs"] = np.array(data_addrs)
        f["times"] = np.array(data_times)
        f["labels"] = np.array(data_labels_processing)
        f.close()

        print(time.time() - start)


if __name__ == '__main__':
    data_path = r"/data1/DVSgesture/test"
    save_path = r'/data1/DVSgesture_process/test'
    noise(
        data_path=data_path,
        save_path=save_path,
    )