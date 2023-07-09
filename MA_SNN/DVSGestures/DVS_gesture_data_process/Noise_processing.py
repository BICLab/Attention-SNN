import h5py
import numpy as np
import torch
import cmath
import os
import time
import matplotlib.pyplot as plt  # plt 用于显示图片
import pandas as pd
import random
from scipy.optimize import least_squares
import statsmodels.formula.api as smf
data_path=r"/data1/DVSgesture/test"
save_path=r'/data1/DVSgesture/test_pic'
# print(list(f.keys()))
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
print(device)
def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)  # 返回一个列表，该列表包含了 path 中所有文件与目录的名称。
                # 列表的内容应该是按照顺序排列，并且不包含特殊条目 '.' 和 '..'，即使它们确实在目录中存在。
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path,dir_file)  # 合理地拼接一个或多个路径部分。
        # 返回值是 path 和 *paths 所有值的连接，每个非空部分后面都紧跟一个目录分隔符 (os.sep)，除了最后一部分。
        # os.path.join函数在这里实际上是将root_path路径连接dir_file从而实现访问root_path下的子文件夹，
        # 并将路径赋值给dir_file_path
        # print(dir_file_path)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):  # 如果 path 是 现有的 目录，则返回 True
            dir_list.append(dir_file_path)  # append() 方法用于在列表末尾添加新的对象，
            # 在这里是将dir_file_path作为一个新对象添加到dir_list中
            # 递归获取所有文件和目录的路径
            # print(dir_list)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)
def func(p,x,y):
    a,b,c=p
    return a*x+b*y+c
def error(p,x,y,t):
    return func(p,x,y)-t
P0=[1,1,1]
file_list=[]
dir_list=[]
x_processing_data=[]
y_processing_data=[]
t_processing_data=[]
one_processing_data=[]
target_data=[]
get_file_path(data_path,file_list,dir_list)
image = torch.zeros(128, 128,dtype=torch.uint8)
image_one = torch.ones(128, 128,dtype=torch.uint8)
# image=image.numpy()
image_zero = torch.zeros(128, 128,dtype=torch.uint8)
num = 0
num_V=0
rand_list = random.sample(range(0, 12000), 10)
rand_time = 0
f = h5py.File(file_list[0], 'r')
dtimes = f['times']
t_last = dtimes[0 + rand_time]
time_val = 0
time_long = 10000
pic_num=0
t_title=0
space=0
aphla=0.0001
list_num=file_list[14]
f=h5py.File(list_num,'r')
daddrs=f['addrs']
dlabels=f['labels']
dtimes=f['times']
data_labels=dlabels[()]
print('data_labels:',str(data_labels))
print('dtime',dtimes[-1])
save_path_next=os.path.join(save_path,str(data_labels))
pic_num=0
t_last = dtimes[0 + rand_time]
print("len_dtimes",len(dtimes))
for i in range(len(dtimes)):
    xaddr=daddrs[i+rand_time-num*time_val][1]
    yaddr=daddrs[i+rand_time-num*time_val][0]
    paddr=daddrs[i+rand_time-num*time_val][2]
    t = dtimes[i+rand_time-num*time_val]
    t_processing=t
    space=0
    loss=0
    loss_pre_a=0
    loss_pre_b=0
    loss_pre_c=0
    a=random.randint(1,10)/10
    b=random.randint(1,10)/10
    c=random.randint(1,10)/10
    Cycle=0
    while 1:
        while t_processing-t<2000 and t_processing<len(dtimes):
            xaddr_processing=daddrs[i+space][1]
            yaddr_processing=daddrs[i+space][0]
            t_processing=dtimes[i+space]
            space+=1
            if abs(xaddr_processing-xaddr)<=3 and abs(yaddr_processing-yaddr)<=3:
                loss+=(a*xaddr_processing+b*yaddr_processing+c-t_processing)**2
                loss_pre_a+=(a*xaddr_processing+b*yaddr_processing+c-t_processing)*xaddr_processing
                loss_pre_b+=(a*xaddr_processing+b*yaddr_processing+c-t_processing)*yaddr_processing
                loss_pre_c+=a*xaddr_processing+b*yaddr_processing+c-t_processing
        loss=loss/(2*space)
        Cycle+=1
        if loss<10000 or Cycle>50:
            if abs(a)>0 and abs(b)>0:
                Vmax=(1/a**2+1/b**2)**0.5
            else:
                Vmax=1
            break
        a=a-aphla/space*loss_pre_a
        b=b-aphla/space*loss_pre_b
        c=c-aphla/space*loss_pre_c
        t_processing=t
        space=0
        # print("loss",loss)
    print("Vmax",Vmax,"t",t)
        # print("a",a,"b",b)
    if Vmax<0.01:
        if int((t - t_last) / time_long) >= 1:
            t_title = 1
        # print('t:', t)
        # print('t_title:',t_title)
        # print(image[xaddr][yaddr])
        if paddr == 1:
            image[xaddr][yaddr] = 1
        else:
            image[xaddr][yaddr] = 138
        # print(t_title)
        if t_title == 1:
            t_title = 0
            # image_show = torch.where(image > 128, image_one * 120, image)
            image_show = torch.where(image < 1, image_one * 255, image)
            # print(image_show)
            image_show_numpy = image_show.numpy()
            plt.subplot(1, 1, num + 1)
            # step_num = 'Event Frame:' + str(num+1)
            # plt.title(step_num)
            plt.imshow(image_show_numpy, cmap=plt.cm.flag)
            plt.draw()
            # time.sleep(1)
            # plt.pause(1)
            num += 1
            # t_last=t
            # rand_time=rand_list[num]
            # rand_time=random.randint(99000,100000)
            t_last = dtimes[i + rand_time - num * time_val]
            image = torch.where(image > 0.5, image_zero, image)
            image_show = torch.where(image > 0.5, image_zero, image)
            image_show_numpy = torch.where(image > 0.5, image_zero, image).numpy()
            # print('1',image)
            # print('2',image_show)
            # print('3',image_show_numpy)
        if num == 1:
            plt.savefig(save_path_next + '/' + str(list_num[38:-5] + '_' + str(pic_num)))
            num = 0
            pic_num += 1








