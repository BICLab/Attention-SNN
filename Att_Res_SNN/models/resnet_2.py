import torch
import torch.nn as nn
import torch.nn.functional as F
from module.LIF_Module import *

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
thresh = 0.5  # 0.5 # neuronal threshold
lens = 0.5  # 0.5 # hyper-parameters of approximate function
decay = 0.25  # 0.25 # decay constants
num_classes = 1000
time_window = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


class mem_update(nn.Module):
    # LIF Layer
    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(
            num_features
        )  # input (N,C,D,H,W) 进行C-dimension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = (
            input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        )  # 可以使用permute实现？ # y = input.permute(1,2,0).contiguous
        # y = self.bn(y,4)
        y = self.bn(y)
        return (
            y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)
        )  # 原始输入是(T,N,C,H,W) BN处理时转变为(N,C,T,H,W)


class batch_norm_2d1(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        # y = self.bn(y,4)
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * thresh)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        marker="b",
    ):
        super(Snn_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (
            input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]
        ) // self.stride[0] + 1
        w = (
            input.size()[4] - self.kernel_size[0] + 2 * self.padding[0]
        ) // self.stride[0] + 1
        c1 = torch.zeros(
            time_window, input.size()[1], self.out_channels, h, w, device=input.device
        )
        for i in range(time_window):
            c1[i] = F.conv2d(
                input[i],
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return c1


######################################################################################################################
class BasicBlock(nn.Module):
    """correspond to 'ReLU only preactivation"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update(),
            Snn_Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            batch_norm_2d1(out_channels * BasicBlock.expansion),
            PruningCell(
                out_channels * BasicBlock.expansion,
                T=1,
                attention="CSA",
                c_ratio=8,
                t_ratio=1,
            ),
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                Snn_Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                batch_norm_2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet_origin(nn.Module):
    # Channel：
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        # batch_norm_2d = group_norm_2d
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 64 * k, kernel_size=7, padding=3, bias=False, stride=2),
            # Snn_Conv2d(3, 64 * k, kernel_size=3, padding=1, stride=2),
            # Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            # Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(
            time_window, x.size()[0], 3, x.size()[2], x.size()[3], device=device
        )
        for i in range(time_window):
            input[i] = x
        output = self.conv1(input)
        # output = mem_update(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.dropout(output)
        output = self.fc(output)
        return output


def resnet18():

    return ResNet_origin(BasicBlock, [2, 2, 2, 2])


def resnet34():
    # return ResNet_origin(BasicBlock, [3, 4, 40, 3])
    return ResNet_origin(BasicBlock, [3, 4, 6, 3])


def resnet104():
    return ResNet_origin(BasicBlock, [3, 8, 32, 8])
