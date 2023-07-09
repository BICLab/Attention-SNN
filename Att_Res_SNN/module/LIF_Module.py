import torch
from torch import nn
from einops import rearrange
from module.utils import *
import torch.nn.functional as F


class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b c f h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c f h w -> b f c h w")
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "b f c h w -> b (f c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)

        return self.sigmoid(x)


class TCSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1):
        super(TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TCA(nn.Module):
    def __init__(
        self, timeWindows, channels, stride=1, fbs=False, t_ratio=16, c_ratio=5
    ):
        super(TCA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fbs = fbs

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(timeWindows, t_ratio)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out_ = self.ta(x) * x
        out = self.ca(out_) * out_  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        if self.fbs:
            return self.ta(x), self.ca(out_)
        else:
            return out


class CSA(nn.Module):
    def __init__(
        self, timeWindows, channels, stride=1, fbs=False, c_ratio=16, t_ratio=1
    ):
        super(CSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1):
        super(TSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, fbs=False, t_ratio=16):
        super(TA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fbs = fbs

        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows, t_ratio)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        if self.fbs:
            return self.ta(x)
        else:
            return out


class CA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, fbs=False, c_ratio=5):
        super(CA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fbs = fbs

        self.ca = ChannelAttention(channels, c_ratio)
        # self.ta = TimeAttention(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        if self.fbs:
            return self.ca(x)
        else:
            return out


class SA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1):
        super(SA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(channels)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(x) * x  # 广播机制

        out = self.relu(out)
        return out


class PruningCell(nn.Module):
    def __init__(
        self,
        hiddenSize,
        attention="no",
        reduction=16,
        T=6,
        fbs=False,
        c_sparsity_ratio=1.0,
        t_sparsity_ratio=1.0,
        c_ratio=16,
        t_ratio=1,
        reserve_coefficient=True,
    ):
        super().__init__()
        self.reserve_coefficient = reserve_coefficient
        self.attention_flag = attention
        self.fbs = fbs
        self.c_sparsity_ratio = c_sparsity_ratio
        self.t_sparsity_ratio = t_sparsity_ratio

        if self.attention_flag == "T":
            self.attention = Tlayer(timeWindows=T, dimension=5, reduction=reduction)
        elif self.attention_flag == "TCSA":
            self.attention = TCSA(T, hiddenSize)
        elif self.attention_flag == "TSA":
            self.attention = TSA(T, hiddenSize)
        elif self.attention_flag == "TCA":
            self.attention = TCA(
                T, hiddenSize, fbs=fbs, c_ratio=c_ratio, t_ratio=t_ratio
            )
        elif self.attention_flag == "CSA":
            self.attention = CSA(
                T, hiddenSize, fbs=fbs, c_ratio=c_ratio, t_ratio=t_ratio
            )
        elif self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, fbs=fbs, t_ratio=t_ratio)
        elif self.attention_flag == "CA":
            self.attention = CA(T, hiddenSize, fbs=fbs, c_ratio=c_ratio)
        elif self.attention_flag == "SA":
            self.attention = SA(T, hiddenSize)
        elif self.attention_flag == "no":
            pass

        if fbs:
            self.avg_c = nn.AdaptiveAvgPool2d(1)
            self.avg_t = nn.AdaptiveAvgPool3d(1)

    def forward(self, data):
        output = data

        if self.attention_flag == "no":
            data = output.permute(1, 0, 2, 3, 4)
            pred_saliency_wta = None
        else:
            if self.fbs:
                # 是否裁剪
                if self.attention_flag == "TA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta = self.attention(data)  # attention score
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = (
                        winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )

                    data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

                elif self.attention_flag == "CA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ca = self.attention(data)  # attention score

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

                    data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

                elif self.attention_flag == "TCA":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta, ca = self.attention(data)  # attention score

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_c_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_t_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(
                        -1
                    ).unsqueeze(-1) * pred_saliency_t_wta.unsqueeze(-1).unsqueeze(
                        -1
                    ).unsqueeze(
                        -1
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(
                        -1
                    ) * winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

            else:
                data = self.attention(output.permute(1, 0, 2, 3, 4))
                pred_saliency_t_wta = 0
                pred_saliency_c_wta = 0

        data = data.permute(1, 0, 2, 3, 4)
        return data
