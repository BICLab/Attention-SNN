# [Attention Spiking Neural Networks](https://ieeexplore.ieee.org/document/10032591)

## **Requirements**

1. Python 3.7.4
2. PyTorch 1.7.1
3. tqdm 4.56.0
4. numpy 1.19.2



## **Instructions**
### 1. DVS128 Gesture

1. Download [DVS128 Gesture](https://www.research.ibm.com/dvsgesture/) and put the downloaded dataset to /MA_SNN/DVSGestures/data, then run /MA_SNN/DVSGestures/data/DVS_Gesture.py.
```
MA_SNN
├── /DVSGestures/
│  ├── /data/
│  │  ├── DVS_Gesture.py
│  │  └── DvsGesture.tar.gz
```
2. Change the values of T and dt in /MA_SNN/DVSGestures/CNN/Config.py then run the tasks in /MA_SNN/DVSGestures.

eg:
```
python Att_SNN_CNN.py
```
3. View the results in /MA_SNN/DVSGestures/CNN/Result/.



### 2. CIFAR10-DVS
1. Download [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2) and processing dataset using official matlab program, then put the result to /MA_SNN/CIFAR10DVS/data.
```
MA_SNN
├── /CIFAR10DVS/
│  ├── /data/
│  │  ├── /airplane/
│  │  |  ├──0.mat
│  │  |  ├──1.mat
│  │  |  ├──...
│  │  ├──automobile
│  │  └──...
```
2. Change the values of T and dt in /MA_SNN/CIFAR10DVS/CNN/Config.py then run the tasks in /MA_SNN/CIFAR10DVS.

eg:
```
python Att_SNN.py
```
3. View the results in /MA_SNN/CIFAR10DVS/CNN/Result/.




### 3. DVSGait Dataset
1. Download [DVSGait Dataset] and put the downloaded dataset to /MA_SNN/DVSGait/data.

2. Change the values of T and dt in /MA_SNN/DVSGait/CNN/Config.py then run the tasks in /MA_SNN/DVSGait.

eg:
```
python Att_SNN_CNN.py
```
3. View the results in /MA_SNN/DVSGait/CNN/Result/.

### 4. ImageNet Dataset

We adopt the MS-SNN (https://github.com/Ariande1/MS-ResNet) as the residual spiking neural network backbone. 

1. Download [ImageNet Dataset] and set the downloaded dataset path in utils.py.
2. then run the tasks in /Att_Res_SNN.

eg:

```
python -m torch.distributed.launch --master_port=[port] --nproc_per_node=[node_num] train_amp.py -net [model_type] -b [batchsize] -lr [learning_rate]
```

3. View the results in /checkpoint and /runs.

### 5. Extra

1. The implementation of Att-VGG-SNN in https://github.com/ridgerchu/SNN_Attention_VGG

2. /module/Attention.py defines the  Attention layer and /module/LIF.py,LIF_Module.py defines LIF module.

3. The CSA-MS-ResNet104 model is available at https://pan.baidu.com/s/1Uro7IVSerV23OKbG8Qn6pQ?pwd=54tl (Code: 54tl).

   

## **Citation**
```
@ARTICLE{10032591,
  author={Yao, Man and Zhao, Guangshe and Zhang, Hengyu and Hu, Yifan and Deng, Lei and Tian, Yonghong and Xu, Bo and Li, Guoqi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention Spiking Neural Networks}, 
  year={2023},
  volume={45},
  number={8},
  pages={9393-9410},
  doi={10.1109/TPAMI.2023.3241201}}
```
