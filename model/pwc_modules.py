from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging
from utils.tools import tools
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
            )


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        # inputs[:, 0, :, :] *= (w / w_)
        # inputs[:, 1, :, :] *= (h / h_)
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u *= u_scale
        v *= v_scale
        res = torch.cat([u, v], dim=1)
    return res


def upsample_flow(inputs, target_size=None, target_flow=None, mode="bilinear"):
    if target_size is not None:
        h, w = target_size
    elif target_flow is not None:
        _, _, h, w = target_flow.size()
    else:
        raise ValueError('wrong input')
    _, _, h_, w_ = inputs.size()
    res = tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    res[:, 0, :, :] *= (w / w_)
    res[:, 1, :, :] *= (h / h_)
    return res


def rescale_flow(flow, div_flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im / div_flow)
        v_scale = float(flow.size(2) / height_im / div_flow)
    else:
        u_scale = float(width_im * div_flow / flow.size(3))
        v_scale = float(height_im * div_flow / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u *= u_scale
    v *= v_scale

    return torch.cat([u, v], dim=1)


class FeatureExtractor(nn.Module):

    def __init__(self, num_chs, if_end_relu=True, if_end_norm=False):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out, isReLU=if_end_relu, if_IN=if_end_norm)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    if x.is_cuda:
        grids_cuda = grid.float().requires_grad_(False).cuda()
    else:
        grids_cuda = grid.float().requires_grad_(False)  # .cuda()
    return grids_cuda


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class WarpingLayer_no_div(nn.Module):

    def __init__(self):
        super(WarpingLayer_no_div, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        x_warp = tf.grid_sample(x, vgrid, padding_mode='zeros')
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = tf.grid_sample(mask, vgrid)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class OpticalFlowEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OpticalFlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_last = conv(32, 2, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class FlowEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorDense_v2(tools.abstract_model):

    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), out_channel=2):
        super(FlowEstimatorDense_v2, self).__init__()
        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.n_channels = N
        ind += 1
        self.conv_last = conv(N, out_channel, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorDense_v3(tools.abstract_model):

    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), if_conv_cat=False):
        super(FlowEstimatorDense_v3, self).__init__()
        self.conv_ls = nn.ModuleList()
        in_channel = ch_in
        self.if_conv_cat = if_conv_cat
        for i in f_channels:
            out_channel = i
            self.conv_ls.append(conv(in_channel, out_channel))
            in_channel += out_channel

        # N = 0
        # ind = 0
        # N += ch_in
        # self.conv1 = conv(N, f_channels[ind])
        # N += f_channels[ind]
        #
        # ind += 1
        # self.conv2 = conv(N, f_channels[ind])
        # N += f_channels[ind]
        #
        # ind += 1
        # self.conv3 = conv(N, f_channels[ind])
        # N += f_channels[ind]
        #
        # ind += 1
        # self.conv4 = conv(N, f_channels[ind])
        # N += f_channels[ind]
        #
        # ind += 1
        # self.conv5 = conv(N, f_channels[ind])
        # N += f_channels[ind]
        self.n_channels = in_channel
        # ind += 1
        self.conv_last = conv(in_channel, 2, isReLU=False)

    def forward(self, x):
        for conv_layer in self.conv_ls:
            x = torch.cat([conv_layer(x), x], dim=1)
        # x1 = torch.cat([self.conv1(x), x], dim=1)
        # x2 = torch.cat([self.conv2(x1), x1], dim=1)
        # x3 = torch.cat([self.conv3(x2), x2], dim=1)
        # x4 = torch.cat([self.conv4(x3), x3], dim=1)
        # x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x)
        return x, x_out


class OcclusionEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OcclusionEstimator, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_last = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class OccEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(OccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 1, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class ContextNetwork_v2_(nn.Module):

    def __init__(self, ch_in, f_channels=(128, 128, 128, 96, 64, 32, 2)):
        super(ContextNetwork_v2_, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, f_channels[0], 3, 1, 1),
            conv(f_channels[0], f_channels[1], 3, 1, 2),
            conv(f_channels[1], f_channels[2], 3, 1, 4),
            conv(f_channels[2], f_channels[3], 3, 1, 8),
            conv(f_channels[3], f_channels[4], 3, 1, 16),
            conv(f_channels[4], f_channels[5], 3, 1, 1),
            conv(f_channels[5], f_channels[6], isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class ContextNetwork_v2(nn.Module):

    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
        super(ContextNetwork_v2, self).__init__()
        self.num_ls = num_ls
        cnt = 0
        cnt_in = num_ls[0]
        self.cov1 = conv(num_ls[0], num_ls[1], 3, 1, 1)

        cnt += 1  # 1
        cnt_in += num_ls[cnt]
        self.cov2 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 2)

        cnt += 1  # 2
        cnt_in += num_ls[cnt]
        self.cov3 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 4)

        cnt += 1  # 3
        cnt_in += num_ls[cnt]
        self.cov4 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 8)

        cnt += 1  # 4
        cnt_in += num_ls[cnt]
        self.cov5 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 16)

        cnt += 1  # 5
        cnt_in += num_ls[cnt]
        self.cov6 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 1)

        cnt += 1
        cnt_in += num_ls[cnt]
        self.final = conv(cnt_in, num_ls[cnt + 1], isReLU=False)

    def forward(self, x):
        x = torch.cat((self.cov1(x), x), dim=1)
        x = torch.cat((self.cov2(x), x), dim=1)
        x = torch.cat((self.cov3(x), x), dim=1)
        x = torch.cat((self.cov4(x), x), dim=1)
        x = torch.cat((self.cov5(x), x), dim=1)
        x = torch.cat((self.cov6(x), x), dim=1)
        x = self.final(x)
        return x


class OccContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(OccContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 1, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)

