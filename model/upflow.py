from __future__ import absolute_import, division, print_function
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.spectral_norm import spectral_norm
from model.pwc_modules import conv, initialize_msra, upsample2d_flow_as, upsample_flow, FlowEstimatorDense_v2, ContextNetwork_v2_, OccEstimatorDense, OccContextNetwork
from model.pwc_modules import WarpingLayer_no_div, FeatureExtractor
from model.correlation_package.correlation import Correlation
import numpy as np
from utils.tools import tools
from utils.loss import loss_functions
from utils.pytorch_correlation import Corr_pyTorch
import cv2
import os
import math


class network_tools():
    class sgu_model(tools.abstract_model):
        def __init__(self):
            super(network_tools.sgu_model, self).__init__()

            class FlowEstimatorDense_temp(tools.abstract_model):

                def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), ch_out=2):
                    super(FlowEstimatorDense_temp, self).__init__()
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
                    self.num_feature_channel = N
                    ind += 1
                    self.conv_last = conv(N, ch_out, isReLU=False)

                def forward(self, x):
                    x1 = torch.cat([self.conv1(x), x], dim=1)
                    x2 = torch.cat([self.conv2(x1), x1], dim=1)
                    x3 = torch.cat([self.conv3(x2), x2], dim=1)
                    x4 = torch.cat([self.conv4(x3), x3], dim=1)
                    x5 = torch.cat([self.conv5(x4), x4], dim=1)
                    x_out = self.conv_last(x5)
                    return x5, x_out

            f_channels_es = (32, 32, 32, 16, 8)
            in_C = 64
            self.warping_layer = WarpingLayer_no_div()
            self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
            self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                      conv(16, 16, stride=2),
                                                      conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                      conv(32, 32, stride=2), )

        def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
            n, c, h, w = flow_init.shape
            n_f, c_f, h_f, w_f = feature_1.shape
            if h != h_f or w != w_f:
                flow_init = upsample2d_flow_as(flow_init, feature_1, mode="bilinear", if_rate=True)
            feature_2_warp = self.warping_layer(feature_2, flow_init)
            input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
            feature, x_out = self.dense_estimator_mask(input_feature)
            inter_flow = x_out[:, :2, :, :]
            inter_mask = x_out[:, 2, :, :]
            inter_mask = torch.unsqueeze(inter_mask, 1)
            inter_mask = torch.sigmoid(inter_mask)
            n_, c_, h_, w_ = inter_flow.shape
            if output_level_flow is not None:
                inter_flow = upsample2d_flow_as(inter_flow, output_level_flow, mode="bilinear", if_rate=True)
                inter_mask = upsample2d_flow_as(inter_mask, output_level_flow, mode="bilinear")
                flow_init = output_level_flow
            flow_up = tools.torch_warp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
            return flow_init, flow_up, inter_flow, inter_mask

        def output_conv(self, x):
            return self.upsample_output_conv(x)

    @classmethod
    def normalize_features(cls, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        """Normalizes feature tensors (e.g., before computing the cost volume).
        Args:
          feature_list: list of torch tensors, each with dimensions [b, c, h, w]
          normalize: bool flag, divide features by their standard deviation
          center: bool flag, subtract feature mean
          moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
          moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True

        Returns:
          list, normalized feature_list
        """

        # Compute feature statistics.

        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            # statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
            #                       len(feature_list))
            # statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
            #                      ] * len(feature_list)
            statistics['mean'] = ([torch.mean(torch.stack(statistics['mean'], dim=0), dim=(0,))] * len(feature_list))
            statistics['var'] = ([torch.var(torch.stack(statistics['var'], dim=0), dim=(0,))] * len(feature_list))

        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

        # Center and normalize features.

        if center:
            feature_list = [
                f - mean for f, mean in zip(feature_list, statistics['mean'])
            ]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

        return feature_list

    @classmethod
    def weighted_ssim(cls, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        """Computes a weighted structured image similarity measure.
        Args:
          x: a batch of images, of shape [B, C, H, W].
          y:  a batch of images, of shape [B, C, H, W].
          weight: shape [B, 1, H, W], representing the weight of each
            pixel in both images when we come to calculate moments (means and
            correlations). values are in [0,1]
          c1: A floating point number, regularizes division by zero of the means.
          c2: A floating point number, regularizes division by zero of the second
            moments.
          weight_epsilon: A floating point number, used to regularize division by the
            weight.

        Returns:
          A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
          similarity loss per pixel per channel, and the second, of shape
          [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
          know how much to weigh each pixel in the first tensor. For example, if
          `'weight` was very small in some area of the images, the first tensor will
          still assign a loss to these pixels, but we shouldn't take the result too
          seriously.
        """

        def _avg_pool3x3(x):
            # tf kernel [b,h,w,c]
            return F.avg_pool2d(x, (3, 3), (1, 1))
            # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

        if c1 == float('inf') and c2 == float('inf'):
            raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                             'likely unintended.')
        average_pooled_weight = _avg_pool3x3(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
            return wighted_avg * inverse_average_pooled_weight

        mu_x = weighted_avg_pool3x3(x)
        mu_y = weighted_avg_pool3x3(y)
        sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
        sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
        sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight

    @classmethod
    def edge_aware_smoothness_order1(cls, img, pred):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def edge_aware_smoothness_order2(cls, img, pred):
        def gradient_x(img, stride=1):
            gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
            return gx

        def gradient_y(img, stride=1):
            gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_xx = gradient_x(pred_gradients_x)
        pred_gradients_y = gradient_y(pred)
        pred_gradients_yy = gradient_y(pred_gradients_y)

        image_gradients_x = gradient_x(img, stride=2)
        image_gradients_y = gradient_y(img, stride=2)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_xx) * weights_x
        smoothness_y = torch.abs(pred_gradients_yy) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def flow_smooth_delta(cls, flow, if_second_order=False):
        def gradient(x):
            D_dy = x[:, :, 1:] - x[:, :, :-1]
            D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        if if_second_order:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        else:
            smooth_loss = dx.abs().mean() + dy.abs().mean()
        # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
        return smooth_loss

    @classmethod
    def photo_loss_multi_type(cls, x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_delta=0.4, photo_loss_use_occ=False,
                              ):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            loss_diff, occ_weight = cls.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        if photo_loss_use_occ:
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss


class UPFlow_net(tools.abstract_model):
    class config(tools.abstract_config):
        def __init__(self):
            # occ loss choose
            self.occ_type = 'for_back_check'
            self.alpha_1 = 0.1
            self.alpha_2 = 0.5
            self.occ_check_obj_out_all = 'obj'
            self.stop_occ_gradient = False
            self.smooth_level = 'final'  # final or 1/4
            self.smooth_type = 'edge'  # edge or delta
            self.smooth_order_1_weight = 1
            # smooth loss
            self.smooth_order_2_weight = 0
            # photo loss type add SSIM
            self.photo_loss_type = 'abs_robust'  # abs_robust, charbonnier,L1, SSIM
            self.photo_loss_delta = 0.4
            self.photo_loss_use_occ = False
            self.photo_loss_census_weight = 0
            # use cost volume norm
            self.if_norm_before_cost_volume = False
            self.norm_moments_across_channels = True
            self.norm_moments_across_images = True
            self.multi_scale_distillation_weight = 0
            self.multi_scale_distillation_style = 'upup'  # down,upup,
            # 'down', 'upup', 'updown'
            self.multi_scale_distillation_occ = True  # if consider occlusion mask in multiscale distilation
            self.if_froze_pwc = False
            self.input_or_sp_input = 1  # use raw input or special input for photo loss
            self.if_use_boundary_warp = True  # if use the boundary dilated warping

            self.if_sgu_upsample = False  # if use sgu upsampling
            self.if_use_cor_pytorch = False  # use my implementation of correlation layer by pytorch. only for test model in cpu(corr layer cuda is not compiled)

        def __call__(self, ):
            # return PWCNet_unsup_irr_bi_v5_4(self)
            return UPFlow_net(self)

    def __init__(self, conf: config):
        super(UPFlow_net, self).__init__()
        # === get config file
        self.conf = conf

        # === build the network
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        self.estimator_f_channels = (128, 128, 96, 64, 32)
        self.context_f_channels = (128, 128, 128, 96, 64, 32, 2)
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense_v2(self.num_ch_in, f_channels=self.estimator_f_channels)
        self.context_networks = ContextNetwork_v2_(self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels)
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])
        self.occ_check_model_ls = []
        self.correlation_pytorch = Corr_pyTorch(pad_size=self.search_range, kernel_size=1,
                                                max_displacement=self.search_range, stride1=1, stride2=1)  # correlation layer using pytorch
        # === build sgu upsampling
        if self.conf.if_sgu_upsample:
            self.sgi_model = network_tools.sgu_model()
        else:
            self.sgi_model = None

        # === build loss function
        self.occ_check_model = tools.occ_check_model(occ_type=self.conf.occ_type, occ_alpha_1=self.conf.alpha_1, occ_alpha_2=self.conf.alpha_2,
                                                     obj_out_all=self.conf.occ_check_obj_out_all)
        initialize_msra(self.modules())
        if self.conf.if_froze_pwc:
            self.froze_PWC()

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start, if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1_ori, im2_ori = input_dict['im1'], input_dict['im2']  # in training: the cropped image; in testing: the input image
        if input_dict['if_loss']:
            if self.conf.input_or_sp_input == 1:
                im1, im2 = im1_ori, im2_ori
            else:
                im1, im2 = input_dict['im1_sp'], input_dict['im2_sp']  # change the input image to special input image
        else:
            im1, im2 = im1_ori, im2_ori

        output_dict = {}
        flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v3(im1, im2, if_loss=input_dict['if_loss'])  # forward estimation
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_pwc_out, flow_b=flow_b_pwc_out)  # 0 in occ area, 1 in others

        '''  ======================================    =====================================  '''
        output_dict['flow_f_out'] = flow_f_pwc_out
        output_dict['flow_b_out'] = flow_b_pwc_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw

        if input_dict['if_loss']:
            # === smooth loss
            if self.conf.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_pwc_out, flow_b_pwc_out
                s_im1, s_im2 = im1_ori, im2_ori
            elif self.conf.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]  # flow in 1/4 scale
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            # 1 order smooth loss
            if self.conf.smooth_order_1_weight > 0:
                if self.conf.smooth_type == 'edge':
                    smooth_loss += self.conf.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.conf.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.conf.smooth_type == 'delta':
                    smooth_loss += self.conf.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.conf.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.conf.smooth_type)

            # 2 order smooth loss
            if self.conf.smooth_order_2_weight > 0:
                if self.conf.smooth_type == 'edge':
                    smooth_loss += self.conf.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.conf.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.conf.smooth_type == 'delta':
                    smooth_loss += self.conf.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.conf.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.conf.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # === photo loss
            if self.conf.if_use_boundary_warp:
                im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']  # the image before cropping
                im1_warp = tools.boundary_dilated_warp.warp_im(im2_s, flow_f_pwc_out, start_s)  # warped im1 by forward flow and im2
                im2_warp = tools.boundary_dilated_warp.warp_im(im1_s, flow_b_pwc_out, start_s)
            else:
                im1_warp = tools.torch_warp(im2_ori, flow_f_pwc_out)  # warped im1 by forward flow and im2
                im2_warp = tools.torch_warp(im1_ori, flow_b_pwc_out)
            # photo loss
            if self.conf.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.conf.photo_loss_type,
                                                             photo_loss_delta=self.conf.photo_loss_delta, photo_loss_use_occ=self.conf.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.conf.photo_loss_type,
                                                              photo_loss_delta=self.conf.photo_loss_delta, photo_loss_use_occ=self.conf.photo_loss_use_occ)
            output_dict['photo_loss'] = photo_loss
            output_dict['im1_warp'] = im1_warp
            output_dict['im2_warp'] = im2_warp

            # === census loss
            if self.conf.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.conf.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.conf.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.conf.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.conf.photo_loss_use_occ, averge=True)
                census_loss *= self.conf.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss

            # === multi scale distillation loss
            if self.conf.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_pwc_out.clone().detach()
                flow_bw_label = flow_b_pwc_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.conf.multi_scale_distillation_style == 'down':
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.conf.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.conf.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.conf.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.conf.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.conf.multi_scale_distillation_weight * msd_loss
            else:
                msd_loss = None

            output_dict['msd_loss'] = msd_loss
        return output_dict

    def forward_2_frame_v3(self, x1_raw, x2_raw, if_loss=False):
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        flows = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self.decode_level_res(level=level, flow_1=flow_f, flow_2=flow_b,
                                                                           feature_1=x1, feature_1_1x1=x1_1by1,
                                                                           feature_2=x2, feature_2_1x1=x2_1by1,
                                                                           img_ori_1=x1_raw, img_ori_2=x2_raw)
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)

        # === upsample to full resolution
        if self.conf.if_sgu_upsample:
            feature_1_1x1 = self.sgi_model.output_conv(x1_raw)
            feature_2_1x1 = self.sgi_model.output_conv(x2_raw)
            flow_f_out = self.self_guided_upsample(flow_up_bilinear=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, output_level_flow=flow_f_out)
            flow_b_out = self.self_guided_upsample(flow_up_bilinear=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, output_level_flow=flow_b_out)
        else:
            pass
        return flow_f_out, flow_b_out, flows[::-1]

    def decode_level_res(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1, img_ori_1, img_ori_2):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.conf.if_sgu_upsample:
                flow_1_up_bilinear = self.self_guided_upsample(flow_up_bilinear=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1)
                flow_2_up_bilinear = self.self_guided_upsample(flow_up_bilinear=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.conf.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True,
                                                                         moments_across_channels=self.conf.norm_moments_across_channels,
                                                                         moments_across_images=self.conf.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True,
                                                                         moments_across_channels=self.conf.norm_moments_across_channels,
                                                                         moments_across_images=self.conf.norm_moments_across_images)
        # correlation
        if self.conf.if_use_cor_pytorch:
            out_corr_1 = self.correlation_pytorch(feature_1, feature_2_warp)
            out_corr_2 = self.correlation_pytorch(feature_2, feature_1_warp)
        else:
            out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
            out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear_ = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear_ = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear_], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear_], dim=1))
        flow_1_res = flow_res_1 + flow_fine_1
        flow_2_res = flow_res_2 + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear, flow_1_res, flow_2_res

    def froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    def self_guided_upsample(self, flow_up_bilinear, feature_1, feature_2, output_level_flow=None):
        flow_up_bilinear_, out_flow, inter_flow, inter_mask = self.sgi_model(flow_up_bilinear, feature_1, feature_2, output_level_flow=output_level_flow)
        return out_flow

    @classmethod
    def demo(cls):
        param_dict = {
            'occ_type': 'for_back_check',
            'alpha_1': 0.1,
            'alpha_2': 0.5,
            'occ_check_obj_out_all': 'obj',
            'stop_occ_gradient': False,
            'smooth_level': 'final',  # final or 1/4
            'smooth_type': 'edge',  # edge or delta
            'smooth_order_1_weight': 1,
            # smooth loss
            'smooth_order_2_weight': 0,
            # photo loss type add SSIM
            'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1, SSIM
            'photo_loss_delta': 0.4,
            'photo_loss_use_occ': False,
            'photo_loss_census_weight': 1,
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'multi_scale_distillation_weight': 1,
            'multi_scale_distillation_style': 'upup',
            'multi_scale_photo_weight': 1,  # 'down', 'upup', 'updown'
            'multi_scale_distillation_occ': True,  # if consider occlusion mask in multiscale distilation
            'if_froze_pwc': False,
            'input_or_sp_input': 1,
            'if_use_boundary_warp': True,
            'if_use_cor_pytorch': True,
        }
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net_conf.get_name(print_now=True)
        net = net_conf()  # .cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float()  # .cuda()
        im_torch = torch.from_numpy(im).float()  # .cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'im1_sp': im_torch, 'im2_sp': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'])
        for i in output_dict.keys():
            if output_dict[i] is None:
                print(i, output_dict[i])
            else:
                tools.check_tensor(output_dict[i], i)


if __name__ == '__main__':
    UPFlow_net.demo()
