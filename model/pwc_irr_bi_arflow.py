from __future__ import absolute_import, division, print_function
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.spectral_norm import spectral_norm
from model.pwc_modules import conv, rescale_flow, upsample2d_as, initialize_msra, upsample2d_flow_as, upsample_flow, FlowEstimatorDense_v2, ContextNetwork_v2_
from model.pwc_modules import WarpingLayer_no_div, FeatureExtractor, ContextNetwork, FlowEstimatorDense, App_model_level_select, WarpingLayer, Appearance_flow_net_for_disdiilation
from model.correlation_package.correlation import Correlation
import numpy as np
from utils_luo.tools import tools
from utils_luo.loss import loss_functions
import cv2
import os


class network_tools():
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

    @classmethod
    def compute_inpaint_photo_loss_mask(cls, img_raw, img_restore, mask, q=0.4, if_l1=False):
        # img = upsample2d_as(img_raw, app_flow)
        # input_im = img * mask
        # img_restore = tools.torch_warp(input_im, app_flow)
        diff = img_raw - img_restore
        loss_mask = 1 - mask
        # print(' ')
        if if_l1:
            diff = torch.abs(diff).mean()
            diff = diff * loss_mask
            loss_mean = diff.mean() / (loss_mask.mean() * 2 + 1e-6)
        else:
            # loss_mean=cls.photo_loss_function(diff=diff,mask=mask,charbonnier_or_abs_robust=False,if_use_occ=True,q=q)
            # loss_mean = (torch.abs(diff * mask) + 0.01).pow(q).mean() / (mask.sum() * 2 + 1e-6)
            diff = (torch.abs(diff) + 0.01).pow(q)
            diff = diff * loss_mask
            diff_sum = torch.sum(diff)
            loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    @classmethod
    def compute_inpaint_photo_loss_mask_multi_type(cls, img_raw, img_restore, mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                                                   q=0.4):
        # img = upsample2d_as(img_raw, app_flow)
        # input_im = img * mask
        # img_restore = tools.torch_warp(input_im, app_flow)
        # diff = img_raw - img_restore
        loss_mask = 1 - mask
        # print(' ')
        # if if_l1:
        #     diff = torch.abs(diff).mean()
        #     diff = diff * loss_mask
        #     loss_mean = diff.mean() / (loss_mask.mean() * 2 + 1e-6)
        # else:
        #     # loss_mean=cls.photo_loss_function(diff=diff,mask=mask,charbonnier_or_abs_robust=False,if_use_occ=True,q=q)
        #     # loss_mean = (torch.abs(diff * mask) + 0.01).pow(q).mean() / (mask.sum() * 2 + 1e-6)
        #     diff = (torch.abs(diff) + 0.01).pow(q)
        #     diff = diff * loss_mask
        #     diff_sum = torch.sum(diff)
        #     loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)

        occ_weight = loss_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = img_raw - img_restore
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(q)
        elif photo_loss_type == 'charbonnier':
            photo_diff = img_raw - img_restore
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(q)
        elif photo_loss_type == 'L1':
            photo_diff = img_raw - img_restore
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            loss_diff, occ_weight = cls.weighted_ssim(img_raw, img_restore, loss_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        diff = loss_diff * occ_weight
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(occ_weight) * 2 + 1e-6)
        return loss_mean


# 这一版改掉了很多原有的细节,因此收敛稍微慢一些了
class PWCNet_unsup_irr_bi_v2(tools.abstract_model):

    def __init__(self, occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj',
                 photo_loss_use_occ=False, photo_loss_delta=0.4,
                 flow_resize_conf='up_flow', multi_scale_weight=(1, 1, 1, 1)):
        super(PWCNet_unsup_irr_bi_v2, self).__init__()
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.flow_resize_conf = flow_resize_conf  # how to calculate photo loss in multiple sacel
        self.multi_scale_weight = multi_scale_weight  # photo loss weight of every scale , multi_scale_weight(1,1,1,1, 4 scales, big to small)

        self._div_flow = 0.05
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)
        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])
        initialize_msra(self.modules())

    def forward_2_frame(self, x1_raw, x2_raw):

        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f_out, flow_b_out = None, None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)
                # flow_f = F.interpolate(flow_f * 2, scale_factor=2, mode="bilinear", align_corners=True)
                # flow_b = F.interpolate(flow_b * 2, scale_factor=2, mode="bilinear", align_corners=True)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b

            # upsampling or post-processing
            if l == self.output_level:
                flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
                flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
                # flows.append([flow_f, flow_b])
                break
            else:
                # flow_f = F.interpolate(flow_f * 4, scale_factor=4, mode="bilinear", align_corners=True)
                # flow_b = F.interpolate(flow_b * 4, scale_factor=4, mode="bilinear", align_corners=True)
                flows.append([flow_f, flow_b])
        if flow_f_out is None or flow_b_out is None:
            raise ValueError('flow_f_out is None or flow_b_out is None:')
        return flow_f_out, flow_b_out, flows[::-1]

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1, im2 = input_dict['im1'], input_dict['im2']
        output_dict = {}
        flow_f_out, flow_b_out, flows = self.forward_2_frame(im1, im2)
        output_dict['flows'] = flows
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)
        if input_dict['if_loss']:
            _, _, h_c, w_c = im1.size()
            photo_loss = None
            for i, (flow_f, flow_b) in enumerate(flows):
                scale_weight = self.multi_scale_weight[i]
                if scale_weight <= 0:
                    pass
                else:
                    b, _, h, w = flow_f.size()  # flow size of current scale
                    if self.flow_resize_conf == 'down_img':  # resize images to match the size of layer
                        # im1_scaled = F.interpolate(im1, (h, w), mode='area')
                        # im2_scaled = F.interpolate(im2, (h, w), mode='area')
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
                    elif self.flow_resize_conf == 'down_img_up_flow4':
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
                    elif self.flow_resize_conf == 'up_flow':
                        im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
                        flow_f_up = upsample2d_flow_as(flow_f, im1, mode="bilinear", if_rate=True)
                        flow_b_up = upsample2d_flow_as(flow_b, im1, mode="bilinear", if_rate=True)

                        im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_up, start_s)  # warped im1 by forward flow and im2
                        im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_up, start_s)
                        im_diff_fw = im1 - im1_warp
                        im_diff_bw = im2 - im2_warp
                        # photo loss
                        photo_loss_sclae = loss_functions.photo_loss_function(diff=im_diff_fw, mask=occ_fw, q=self.photo_loss_delta,
                                                                              charbonnier_or_abs_robust=False,
                                                                              if_use_occ=self.photo_loss_use_occ, averge=True) + \
                                           loss_functions.photo_loss_function(diff=im_diff_bw, mask=occ_bw, q=self.photo_loss_delta,
                                                                              charbonnier_or_abs_robust=False,
                                                                              if_use_occ=self.photo_loss_use_occ, averge=True)
                        if photo_loss is None:
                            photo_loss = scale_weight * photo_loss_sclae
                        else:
                            photo_loss += scale_weight * photo_loss_sclae
                    else:
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
            output_dict['photo_loss'] = photo_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', photo_loss_use_occ=False).cuda()
        net.eval()
        im = np.zeros((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        flows = output_dict['flows']
        for cnt, (out_f, out_b) in enumerate(flows):
            tools.check_tensor(out_f, 'out_f %s' % cnt)
            tools.check_tensor(out_b, 'out_b %s' % cnt)
            print(' ')


# 这一版保留原版的一些细节,然后在forward过程中计算几个尺度的photo loss
class PWCNet_unsup_irr_bi_v3(tools.abstract_model):

    def __init__(self, occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj',
                 photo_loss_use_occ=False, photo_loss_delta=0.4,
                 flow_resize_conf='up_flow', multi_scale_weight=(1, 1, 1, 1),
                 # 我决定,就在这个版本的基础上加入新的功能啦
                 ):
        super(PWCNet_unsup_irr_bi_v3, self).__init__()
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.flow_resize_conf = flow_resize_conf  # how to calculate photo loss in multiple sacel
        self.multi_scale_weight = multi_scale_weight  # photo loss weight of every scale , multi_scale_weight(1,1,1,1, 4 scales, big to small)

        self._div_flow = 0.05
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)
        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward_2_frame(self, x1_raw, x2_raw):

        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)
            del out_corr_f
            del out_corr_b

            # concat and estimate flow
            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            del x1_1by1
            del x2_1by1
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            del x_intm_f
            del x_intm_b
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b

            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)  # u_scale = float(width_im * div_flow / flow.size(3))
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

            # upsampling or post-processing
            if l == self.output_level:
                break
            else:
                # flows.append([flow_f/self._div_flow, flow_b/self._div_flow])
                flows.append([rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True), rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)])

        # output_dict['flow'] = flows

        # if self.training:
        #     return output_dict
        # else:
        #     output_dict_eval = {}
        #     output_dict_eval['flow'] = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        #     return output_dict_eval
        out_flow_f = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        out_flow_b = upsample2d_as(flow_b, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        return out_flow_f, out_flow_b, flows[::-1]

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1, im2 = input_dict['im1'], input_dict['im2']
        output_dict = {}
        flow_f_out, flow_b_out, flows = self.forward_2_frame(im1, im2)
        # output_dict['flows'] = flows#暂时不能用这个，save显存
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)
        if input_dict['if_loss']:
            _, _, h_c, w_c = im1.size()
            photo_loss = None
            for i, (flow_f, flow_b) in enumerate(flows):
                scale_weight = self.multi_scale_weight[i]
                if scale_weight <= 0:
                    pass
                else:
                    b, _, h, w = flow_f.size()  # flow size of current scale
                    if self.flow_resize_conf == 'down_img':  # resize images to match the size of layer
                        # im1_scaled = F.interpolate(im1, (h, w), mode='area')
                        # im2_scaled = F.interpolate(im2, (h, w), mode='area')
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
                    elif self.flow_resize_conf == 'down_img_up_flow4':
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
                    elif self.flow_resize_conf == 'up_flow':
                        im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
                        flow_f_up = upsample2d_flow_as(flow_f, im1, mode="bilinear", if_rate=True)
                        flow_b_up = upsample2d_flow_as(flow_b, im1, mode="bilinear", if_rate=True)

                        im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_up, start_s)  # warped im1 by forward flow and im2
                        im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_up, start_s)
                        im_diff_fw = im1 - im1_warp
                        im_diff_bw = im2 - im2_warp
                        # photo loss
                        photo_loss_sclae = loss_functions.photo_loss_function(diff=im_diff_fw, mask=occ_fw, q=self.photo_loss_delta,
                                                                              charbonnier_or_abs_robust=False,
                                                                              if_use_occ=self.photo_loss_use_occ, averge=True) + \
                                           loss_functions.photo_loss_function(diff=im_diff_bw, mask=occ_bw, q=self.photo_loss_delta,
                                                                              charbonnier_or_abs_robust=False,
                                                                              if_use_occ=self.photo_loss_use_occ, averge=True)
                        if photo_loss is None:
                            photo_loss = scale_weight * photo_loss_sclae
                        else:
                            photo_loss += scale_weight * photo_loss_sclae
                    else:
                        raise ValueError('not implemented, flow_resize_conf: %s' % self.flow_resize_conf)
            output_dict['photo_loss'] = photo_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v3(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', photo_loss_use_occ=False).cuda()
        net.eval()
        im = np.zeros((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        flows = output_dict['flows']
        for cnt, (out_f, out_b) in enumerate(flows):
            tools.check_tensor(out_f, 'out_f %s' % cnt)
            tools.check_tensor(out_b, 'out_b %s' % cnt)
            print(' ')


# 这一版保留原版的一些细节,不打算再求多尺度的photo loss了, smooth loss的位置选择，因此smooth loss的所有参数都在这里选择了，photo loss也在这里求吧，ssim选择，cost volume norm
class PWCNet_unsup_irr_bi_v4(tools.abstract_model):

    def __init__(self,
                 # smooth loss 类型及位置选择
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,  # 一阶还是二阶，还是高阶(暂时不知道怎么实现), 这俩weight如果小于等于0就表示莫得
                 smooth_order_2_weight=0,
                 # photo loss类型选择，新加入SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,  # 是不是使用census loss呢
                 # 是否cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',  # 几种选择，把大尺度光流下采样'down', 把小尺度的光流上采样到原尺寸上'upup', 只把小尺度光流上采样四倍，大尺度光流下采样'updown'
                 multi_scale_distillation_occ=True,  # 计算多尺度损失的时候是否考虑遮挡区域
                 ):
        super(PWCNet_unsup_irr_bi_v4, self).__init__()
        self.if_test = if_test  # 用来判断是不是在测试的时候多传一些东西来展示
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume  # 是否做cost volume norm
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self._div_flow = 0.05
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)
        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward_2_frame(self, x1_raw, x2_raw):

        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)
            del out_corr_f
            del out_corr_b

            # concat and estimate flow
            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            del x1_1by1
            del x2_1by1
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            del x_intm_f
            del x_intm_b
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b

            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)  # u_scale = float(width_im * div_flow / flow.size(3))
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)
            # 不知道为什么只要改flows，输出就会被改变，因此新建一个flows_v2
            temp_flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
            tem_flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)
            flows.append([temp_flow_f, tem_flow_b])  # 不知道为什么，一定要有这个操作才行， 这样输出的光流是正确的

            # def get_flow_scale(_flow_, ori_h, ori_w):#这一块, 主要是为了测试中间scale的输出光流的值域是否正确
            #     u_scale = float(_flow_.size(3) / ori_w)
            #     v_scale = float(_flow_.size(2) / ori_h )
            #     u, v = _flow_.chunk(2, dim=1)
            #     u *= u_scale
            #     v *= v_scale
            #     return torch.cat([u, v], dim=1)
            #
            # flows_v2.append([flow_f * (1.0 / self._div_flow), flow_b * (1.0 / self._div_flow)])
            # upsampling or post-processing
            if l == self.output_level:
                break
            # else:
            #     # flows.append([flow_f/self._div_flow, flow_b/self._div_flow])
            #     flows.append([rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True), rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)])

        # output_dict['flow'] = flows

        # if self.training:
        #     return output_dict
        # else:
        #     output_dict_eval = {}
        #     output_dict_eval['flow'] = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        #     return output_dict_eval
        out_flow_f = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        out_flow_b = upsample2d_as(flow_b, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        return out_flow_f, out_flow_b, flows[::-1]

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1, im2 = input_dict['im1'], input_dict['im2']
        output_dict = {}
        flow_f_out, flow_b_out, flows = self.forward_2_frame(im1, im2)
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)
        # output_dict['flows'] = flows#暂时不能用这个，save显存
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # _, _, h_c, w_c = im1.size()
            # 先计算smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1, im2
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # 接下来计算photo loss
            im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
            im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
            im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                             photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                              photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)

            output_dict['photo_loss'] = photo_loss
            # census loss
            if self.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':  # 几种选择，把大尺度光流下采样'down', 把小尺度的光流上采样到原尺寸上'upup', 只把小尺度光流上采样四倍，大尺度光流下采样'updown'
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  # 上采样4倍
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  # 标签下采样下来
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ 也下采样下来
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                msd_loss = None
            output_dict['msd_loss'] = msd_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v4(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                     occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                     smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                     photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                     if_norm_before_cost_volume=False, norm_moments_across_channels=True, norm_moments_across_images=True,
                                     ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'])


# 这一版比v4版有些改动，主要是光流的scale上面的事情，想去掉div flow这个东西，但代价是模型得重新训练，跟v4以前的模型参数不兼容
class PWCNet_unsup_irr_bi_v5(tools.abstract_model):

    def __init__(self,
                 # smooth loss 类型及位置选择
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,  # 一阶还是二阶，还是高阶(暂时不知道怎么实现), 这俩weight如果小于等于0就表示莫得
                 smooth_order_2_weight=0,
                 # photo loss类型选择，新加入SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,  # 是不是使用census loss呢
                 # 是否cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',  # 几种选择，把大尺度光流下采样'down', 把小尺度的光流上采样到原尺寸上'upup', 只把小尺度光流上采样四倍，大尺度光流下采样'updown'
                 multi_scale_distillation_occ=True,  # 计算多尺度损失的时候是否考虑遮挡区域
                 ):
        super(PWCNet_unsup_irr_bi_v5, self).__init__()
        self.if_test = if_test  # 用来判断是不是在测试的时候多传一些东西来展示
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume  # 是否做cost volume norm
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        # self.warping_layer = WarpingLayer()
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)
        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward_2_frame(self, x1_raw, x2_raw):

        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                # x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)
            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b
            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b
            flows.append([flow_f, flow_b])  # 不知道为什么，一定要有这个操作才行， 这样输出的光流是正确的
            if l == self.output_level:
                break
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        return flow_f_out, flow_b_out, flows[::-1]

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1, im2 = input_dict['im1'], input_dict['im2']
        output_dict = {}
        flow_f_out, flow_b_out, flows = self.forward_2_frame(im1, im2)
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)
        # output_dict['flows'] = flows#暂时不能用这个，save显存
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # _, _, h_c, w_c = im1.size()
            # 先计算smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1, im2
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # 接下来计算photo loss
            im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
            im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
            im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                             photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                              photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)

            output_dict['photo_loss'] = photo_loss
            # census loss
            if self.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':  # 几种选择，把大尺度光流下采样'down', 把小尺度的光流上采样到原尺寸上'upup', 只把小尺度光流上采样四倍，大尺度光流下采样'updown'
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  # 上采样4倍
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  # 标签下采样下来
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ 也下采样下来
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                msd_loss = None
            output_dict['msd_loss'] = msd_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v4(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                     occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                     smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                     photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                     if_norm_before_cost_volume=False, norm_moments_across_channels=True, norm_moments_across_images=True,
                                     ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'])


# add appearance flow for distilation
class PWCNet_unsup_irr_bi_v5_1(tools.abstract_model):

    def __init__(self,
                 # smooth loss choose
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,
                 # smooth loss
                 smooth_order_2_weight=0,
                 # photo loss type add SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,
                 # use cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',
                 multi_scale_photo_weight=0,
                 # 'down', 'upup', 'updown'
                 multi_scale_distillation_occ=True,  # if consider occlusion mask in multiscale distilation
                 # appearance flow params
                 if_froze_pwc=False,
                 app_occ_stop_gradient=True,
                 app_loss_weight=0,
                 app_distilation_weight=0,
                 if_upsample_flow=False,
                 if_upsample_flow_mask=False,
                 if_upsample_flow_output=False,
                 if_concat_multi_scale_feature=False,
                 input_or_sp_input=1,
                 ):
        super(PWCNet_unsup_irr_bi_v5_1, self).__init__()
        self.input_or_sp_input = input_or_sp_input  # 控制用sp crop来forward，然后用原图计算photo loss
        self.if_save_running_process = False
        self.save_running_process_dir = ''
        self.if_test = if_test
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_photo_weight = multi_scale_photo_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        # self.warping_layer = WarpingLayer()
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)
        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)
        self.if_concat_multi_scale_feature = if_concat_multi_scale_feature
        if if_concat_multi_scale_feature:
            self.conv_1x1_cmsf = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(128 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(96 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(64 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(32 + 32, 32, kernel_size=1, stride=1, dilation=1)])
        else:
            self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        # flow upsample module
        # flow upsample module
        class _Upsample_flow(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow, self).__init__()
                ch_in = 32
                k = ch_in
                ch_out = 64
                self.conv1 = conv(ch_in, ch_out)
                k += ch_out

                ch_out = 64
                self.conv2 = conv(k, ch_out)
                k += ch_out

                ch_out = 32
                self.conv3 = conv(k, ch_out)
                k += ch_out

                ch_out = 16
                self.conv4 = conv(k, ch_out)
                k += ch_out

                # ch_out = 64
                # self.conv5 = conv(k, ch_out)
                # k += ch_out
                self.conv_last = conv(k, 2, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                # x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x4)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.1f G, params: %.1f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.1f G, params: %.1f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v2(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v2, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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

                        ind += 1
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                _, x_out = self.dense_estimator(x)
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow_v2()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow_v2()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v3(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v3, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 64, 64, 64, 32, 32, 16, 2))
                # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                feature, x_out = self.dense_estimator(x)
                flow = flow_pre + x_out
                flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                x_out = flow + flow_fine_f
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v3()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 2, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 2, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v4(tools.abstract_model):
            def __init__(self, if_mask):
                super(_Upsample_flow_v4, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16), ch_out=3)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=(self.dense_estimator_mask.num_feature_channel + 3, 64, 64, 64, 32, 32, 16, 3))
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                    self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 64, 64, 64, 32, 32, 16, 2))
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                if self.if_mask:
                    feature, x_out = self.dense_estimator_mask(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out[:, :2, :, :]
                    mask_out = x_out[:, 2, :, :]
                    mask_out = torch.unsqueeze(mask_out, 1)
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                        mask_out = upsample2d_flow_as(mask_out, x_raw, mode="bilinear")
                    mask_out = torch.sigmoid(mask_out)
                    return x_out, flow_out, mask_out
                else:
                    feature, x_out = self.dense_estimator(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                    return flow_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        self.if_upsample_flow = if_upsample_flow
        self.if_upsample_flow_output = if_upsample_flow_output
        self.if_upsample_flow_mask = if_upsample_flow_mask
        if self.if_upsample_flow or self.if_upsample_flow_output:
            self.upsample_model = _Upsample_flow_v4(self.if_upsample_flow_mask)
        else:
            self.upsample_model = None
            self.upsample_output_conv = None

        # app flow module
        self.app_occ_stop_gradient = app_occ_stop_gradient  # stop gradient of the occ mask when inpaint
        self.app_distilation_weight = app_distilation_weight
        if app_loss_weight > 0:
            self.appflow_model = Appearance_flow_net_for_disdiilation.App_model(input_channel=7, if_share_decoder=False)
        self.app_loss_weight = app_loss_weight

        class _WarpingLayer(tools.abstract_model):

            def __init__(self):
                super(_WarpingLayer, self).__init__()

            def forward(self, x, flo):
                """
                warp an image/tensor (im2) back to im1, according to the optical flow

                x: [B, C, H, W] (im2)
                flo: [B, 2, H, W] flow

                """

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
                vgrid = grid + flo

                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
                output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
                if x.is_cuda:
                    mask = torch.ones(x.size(), requires_grad=False).cuda()
                else:
                    mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
                mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                mask = (mask >= 1.0).float()
                # mask = torch.autograd.Variable(torch.ones(x.size()))
                # if x.is_cuda:
                #     mask = mask.cuda()
                # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                #
                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
                output = output * mask
                # # nchw->>>nhwc
                # if x.is_cuda:
                #     output = output.cpu()
                # output_im = output.numpy()
                # output_im = np.transpose(output_im, (0, 2, 3, 1))
                # output_im = np.squeeze(output_im)
                return output

        self.warping_layer_inpaint = _WarpingLayer()

        initialize_msra(self.modules())
        self.if_froze_pwc = if_froze_pwc
        if self.if_froze_pwc:
            self.froze_PWC()

    def froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    def save_image(self, image_tensor, name='image'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join('/data/Optical_Flow_all/training_v3', name + '.png'), tools.Show_GIF.im_norm(image_tensor_np)[:, :, ::-1])

    def save_flow(self, flow_tensor, name='flow'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        flow_tensor_np = tensor_to_np_for_save(flow_tensor)
        cv2.imwrite(os.path.join(self.save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(flow_tensor_np)[:, :, ::-1]))

    def save_mask(self, image_tensor, name='mask'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(self.save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np))

    def decode_level(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1,
                     up_flow_1=None, up_flow_mask_1=None, up_flow_mask_2=None, up_flow_2=None):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_save_running_process:
                self.save_flow(flow_1_up_bilinear, '%s_flow_f_up2d' % level)
                self.save_flow(flow_2_up_bilinear, '%s_flow_b_up2d' % level)
            if self.if_upsample_flow or self.if_upsample_flow_output:
                up_flow_1 = upsample2d_flow_as(up_flow_1, feature_1, mode="bilinear", if_rate=True)
                up_flow_2 = upsample2d_flow_as(up_flow_2, feature_2, mode="bilinear", if_rate=True)
                if self.if_upsample_flow_mask:
                    up_flow_mask_1 = upsample2d_flow_as(up_flow_mask_1, feature_1, mode="bilinear")
                    up_flow_mask_2 = upsample2d_flow_as(up_flow_mask_2, feature_2, mode="bilinear")
                    _, up_flow_1, up_flow_mask_1 = self.upsample_model(torch.cat((up_flow_1, up_flow_mask_1), dim=1), feature_1_1x1)
                    _, up_flow_2, up_flow_mask_2 = self.upsample_model(torch.cat((up_flow_2, up_flow_mask_2), dim=1), feature_2_1x1)
                    if self.if_upsample_flow:
                        flow_1_up_bilinear = flow_1_up_bilinear * up_flow_mask_1 + tools.torch_warp(flow_1_up_bilinear, up_flow_1) * (1 - up_flow_mask_1)
                        flow_2_up_bilinear = flow_2_up_bilinear * up_flow_mask_2 + tools.torch_warp(flow_2_up_bilinear, up_flow_2) * (1 - up_flow_mask_2)
                        if self.if_save_running_process:
                            self.save_flow(flow_1_up_bilinear, '%s_flow_f_upbyflow' % level)
                            self.save_flow(flow_2_up_bilinear, '%s_flow_b_upbyflow' % level)
                            self.save_flow(up_flow_1, '%s_flow_f_upflow' % level)
                            self.save_flow(up_flow_2, '%s_flow_b_upflow' % level)
                            self.save_mask(up_flow_mask_1, '%s_flow_f_upmask' % level)
                            self.save_mask(up_flow_mask_2, '%s_flow_b_upmask' % level)
                else:
                    up_flow_1 = self.upsample_model(up_flow_1, feature_1_1x1)
                    up_flow_2 = self.upsample_model(up_flow_2, feature_2_1x1)
                    if self.if_upsample_flow:
                        # flow_f = self.warping_layer(flow_f, up_flow_f)
                        # flow_b = self.warping_layer(flow_b, up_flow_b)
                        flow_1_up_bilinear = tools.torch_warp(flow_1_up_bilinear, up_flow_1)
                        flow_2_up_bilinear = tools.torch_warp(flow_2_up_bilinear, up_flow_2)
                        if self.if_save_running_process:
                            self.save_flow(flow_1_up_bilinear, '%s_flow_f_upbyflow' % level)
                            self.save_flow(flow_2_up_bilinear, '%s_flow_b_upbyflow' % level)
                            self.save_flow(up_flow_1, '%s_flow_f_upflow' % level)
                            self.save_flow(up_flow_2, '%s_flow_b_upflow' % level)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
        out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
        out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_fine_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear, up_flow_1, up_flow_mask_1, up_flow_mask_2, up_flow_2

    def forward_2_frame_v2(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        up_flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_f_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        level_iter_ls = (1, 1, 1, 1, 1)
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            level_iter = level_iter_ls[level]
            for _ in range(level_iter):
                flow_f, flow_b, up_flow_f, up_flow_f_mask, up_flow_b_mask, up_flow_b = \
                    self.decode_level(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2, feature_2_1x1=x2_1by1,
                                      up_flow_1=up_flow_f, up_flow_mask_1=up_flow_f_mask, up_flow_mask_2=up_flow_b_mask, up_flow_2=up_flow_b)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(flow_f_out, 'out_flow_f_up2d')
            self.save_flow(flow_b_out, 'out_flow_b_up2d')
            self.save_image(x1_raw, 'image1')
            self.save_image(x2_raw, 'image2')
        if self.if_upsample_flow_output:
            if self.if_upsample_flow_mask:
                _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_raw, if_output_level=True)
                _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_raw, if_output_level=True)
                # flow_f_out = flow_f_out * up_flow_f_mask + self.warping_layer(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                # flow_b_out = flow_b_out * up_flow_b_mask + self.warping_layer(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                flow_f_out = flow_f_out * up_flow_f_mask + tools.torch_warp(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                flow_b_out = flow_b_out * up_flow_b_mask + tools.torch_warp(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                if self.if_save_running_process:
                    self.save_flow(flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(up_flow_b, '%s_flow_b_upflow' % 'out')
                    self.save_mask(up_flow_f_mask, '%s_flow_f_upmask' % 'out')
                    self.save_mask(up_flow_b_mask, '%s_flow_b_upmask' % 'out')
            else:
                up_flow_f = self.upsample_model(up_flow_f, x1_raw, if_output_level=True)
                up_flow_b = self.upsample_model(up_flow_b, x2_raw, if_output_level=True)
                # flow_f_out = self.warping_layer(flow_f_out, up_flow_f)
                # flow_b_out = self.warping_layer(flow_b_out, up_flow_b)
                flow_f_out = tools.torch_warp(flow_f_out, up_flow_f)
                flow_b_out = tools.torch_warp(flow_b_out, up_flow_b)
                if self.if_save_running_process:
                    self.save_flow(flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(up_flow_b, '%s_flow_b_upflow' % 'out')
        return flow_f_out, flow_b_out, flows[::-1]

    def forward_2_frame(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        up_flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_f_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                # x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                if self.if_save_running_process:
                    self.save_flow(flow_f, '%s_flow_f_up2d' % l)
                    self.save_flow(flow_b, '%s_flow_b_up2d' % l)
                if self.if_upsample_flow or self.if_upsample_flow_output:
                    up_flow_f = upsample2d_flow_as(up_flow_f, x1, mode="bilinear", if_rate=True)
                    up_flow_b = upsample2d_flow_as(up_flow_b, x1, mode="bilinear", if_rate=True)
                    if self.if_upsample_flow_mask:
                        up_flow_f_mask = upsample2d_flow_as(up_flow_f_mask, x1, mode="bilinear")
                        up_flow_b_mask = upsample2d_flow_as(up_flow_b_mask, x1, mode="bilinear")
                        _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_1by1)
                        _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = flow_f * up_flow_f_mask + self.warping_layer(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            # flow_b = flow_b * up_flow_b_mask + self.warping_layer(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            flow_f = flow_f * up_flow_f_mask + tools.torch_warp(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            flow_b = flow_b * up_flow_b_mask + tools.torch_warp(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            if self.if_save_running_process:
                                self.save_flow(flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(up_flow_b, '%s_flow_b_upflow' % l)
                                self.save_mask(up_flow_f_mask, '%s_flow_f_upmask' % l)
                                self.save_mask(up_flow_b_mask, '%s_flow_b_upmask' % l)
                    else:
                        up_flow_f = self.upsample_model(up_flow_f, x1_1by1)
                        up_flow_b = self.upsample_model(up_flow_b, x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = self.warping_layer(flow_f, up_flow_f)
                            # flow_b = self.warping_layer(flow_b, up_flow_b)
                            flow_f = tools.torch_warp(flow_f, up_flow_f)
                            flow_b = tools.torch_warp(flow_b, up_flow_b)
                            if self.if_save_running_process:
                                self.save_flow(flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(up_flow_b, '%s_flow_b_upflow' % l)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b
            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b
            flows.append([flow_f, flow_b])
            if l == self.output_level:
                break
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(flow_f_out, 'out_flow_f_up2d')
            self.save_flow(flow_b_out, 'out_flow_b_up2d')
            self.save_image(x1_raw, 'image1')
            self.save_image(x2_raw, 'image2')
        if self.if_upsample_flow_output:
            if self.if_upsample_flow_mask:
                _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_raw, if_output_level=True)
                _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_raw, if_output_level=True)
                # flow_f_out = flow_f_out * up_flow_f_mask + self.warping_layer(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                # flow_b_out = flow_b_out * up_flow_b_mask + self.warping_layer(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                flow_f_out = flow_f_out * up_flow_f_mask + tools.torch_warp(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                flow_b_out = flow_b_out * up_flow_b_mask + tools.torch_warp(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                if self.if_save_running_process:
                    self.save_flow(flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(up_flow_b, '%s_flow_b_upflow' % 'out')
                    self.save_mask(up_flow_f_mask, '%s_flow_f_upmask' % 'out')
                    self.save_mask(up_flow_b_mask, '%s_flow_b_upmask' % 'out')
            else:
                up_flow_f = self.upsample_model(up_flow_f, x1_raw, if_output_level=True)
                up_flow_b = self.upsample_model(up_flow_b, x2_raw, if_output_level=True)
                # flow_f_out = self.warping_layer(flow_f_out, up_flow_f)
                # flow_b_out = self.warping_layer(flow_b_out, up_flow_b)
                flow_f_out = tools.torch_warp(flow_f_out, up_flow_f)
                flow_b_out = tools.torch_warp(flow_b_out, up_flow_b)
                if self.if_save_running_process:
                    self.save_flow(flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(up_flow_b, '%s_flow_b_upflow' % 'out')
        return flow_f_out, flow_b_out, flows[::-1]

    def app_refine(self, img, flow, mask):
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img * mask
        app_flow = self.appflow_model(torch.cat((input_im, img, mask), dim=1))
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        flow_restore = self.warping_layer_inpaint(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img_restore

    def app_loss(self, img_ori, img_restore, occ_mask):
        diff = img_ori - img_restore
        loss_mask = 1 - occ_mask  # only take care about the inpainting area
        diff = (torch.abs(diff) + 0.01).pow(0.4)
        diff = diff * loss_mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1_ori, im2_ori = input_dict['im1'], input_dict['im2']
        if input_dict['if_loss']:
            sp_im1_ori, sp_im2_ori = input_dict['im1_sp'], input_dict['im2_sp']
            if self.input_or_sp_input >= 1:
                im1, im2 = im1_ori, im2_ori
            elif self.input_or_sp_input > 0:
                if tools.random_flag(threshold_0_1=self.input_or_sp_input):
                    im1, im2 = im1_ori, im2_ori
                else:
                    im1, im2 = sp_im1_ori, sp_im2_ori
            else:
                im1, im2 = sp_im1_ori, sp_im2_ori
        else:
            im1, im2 = im1_ori, im2_ori

        # 是否测试
        if 'if_test' in input_dict.keys():
            if_test = input_dict['if_test']
        else:
            if_test = False
        # 是否传回一些结果用于展示
        if 'save_running_process' in input_dict.keys():
            self.if_save_running_process = input_dict['save_running_process']
        else:
            self.if_save_running_process = False
        if self.if_save_running_process:
            if 'process_dir' in input_dict.keys():
                self.save_running_process_dir = input_dict['process_dir']
            else:
                self.if_save_running_process = False  # 如果保存文件夹没指定的话，就还是False，不保存为好
                self.save_running_process_dir = None
        # 是否在测试或者验证的时候保存中间结果
        if 'if_show' in input_dict.keys():
            if_show = input_dict['if_show']
        else:
            if_show = False
        output_dict = {}
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame(im1, im2)
        flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v2(im1, im2)
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_pwc_out, flow_b=flow_b_pwc_out)
        if self.app_loss_weight > 0:
            if self.app_occ_stop_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            # tools.check_tensor(occ_fw, '%s' % (torch.sum(occ_fw == 1) / torch.sum(occ_fw)))
            flow_f, app_flow_1, masked_im1, im1_restore = self.app_refine(img=im1, flow=flow_f_pwc_out, mask=occ_fw)
            # tools.check_tensor(app_flow_1, 'app_flow_1')
            flow_b, app_flow_2, masked_im2, im2_restore = self.app_refine(img=im2, flow=flow_b_pwc_out, mask=occ_bw)
            app_loss = self.app_loss(im1, im1_restore, occ_fw)
            app_loss += self.app_loss(im2, im2_restore, occ_bw)
            app_loss *= self.app_loss_weight
            # tools.check_tensor(app_loss, 'app_loss')
            # print(' ')
            if input_dict['if_loss']:
                output_dict['app_loss'] = app_loss
            if self.app_distilation_weight > 0:
                flow_fw_label = flow_f.clone().detach()
                flow_bw_label = flow_b.clone().detach()
                appd_loss = network_tools.photo_loss_multi_type(x=flow_fw_label, y=flow_f_pwc_out, occ_mask=1 - occ_fw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss += network_tools.photo_loss_multi_type(x=flow_bw_label, y=flow_b_pwc_out, occ_mask=1 - occ_bw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss *= self.app_distilation_weight
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = appd_loss
                if if_test:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                else:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                    # flow_f_out = flow_f  # use app refine output
                    # flow_b_out = flow_b
            else:
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = None
                flow_f_out = flow_f
                flow_b_out = flow_b
            if if_show:
                output_dict['app_flow_1'] = app_flow_1
                output_dict['masked_im1'] = masked_im1
                output_dict['im1_restore'] = im1_restore
        else:
            if input_dict['if_loss']:
                output_dict['app_loss'] = None
            flow_f_out = flow_f_pwc_out
            flow_b_out = flow_b_pwc_out
            if if_show:
                output_dict['app_flow_1'] = None
                output_dict['masked_im1'] = None
                output_dict['im1_restore'] = None

        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # 计算 smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1_ori, im2_ori
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            # 计算 二阶 smooth loss
            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # 计算 photo loss
            im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
            im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
            im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                             photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                              photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            output_dict['photo_loss'] = photo_loss
            output_dict['im1_warp'] = im1_warp
            output_dict['im2_warp'] = im2_warp

            # 计算 census loss
            if self.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss

            # 计算多尺度蒸馏msd loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  #
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                # 换成计算多尺度photo loss： multi_scale_photo_weight
                if self.multi_scale_photo_weight > 0:
                    _, _, h_raw, w_raw = im1_s.size()
                    _, _, h_temp_crop, h_temp_crop = im1_ori.size()
                    msd_loss_ls = []
                    for i, (scale_fw, scale_bw) in enumerate(flows):
                        if self.multi_scale_distillation_style == 'down':  # 原图resize小计算photo loss
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        elif self.multi_scale_distillation_style == 'upup':  # 多尺度flow resize大了和原图计算photo loss
                            occ_f_resize = occ_fw
                            occ_b_resize = occ_bw
                            scale_fw = upsample_flow(scale_fw, target_flow=im1_ori)
                            scale_bw = upsample_flow(scale_bw, target_flow=im2_ori)
                            im1_crop_resize = im1
                            im2_crop_resize = im2
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_s, scale_fw, start_s)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_s, scale_bw, start_s)
                        elif self.multi_scale_distillation_style == 'updown':
                            scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                            scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        else:
                            raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)

                        temp_mds_fw = network_tools.photo_loss_multi_type(im1_crop_resize, im1_resize_warp, occ_f_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_fw)
                        temp_mds_bw = network_tools.photo_loss_multi_type(im2_crop_resize, im2_resize_warp, occ_b_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_bw)
                    msd_loss = sum(msd_loss_ls)
                    msd_loss = self.multi_scale_photo_weight * msd_loss
                else:
                    msd_loss = None

            output_dict['msd_loss'] = msd_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v5_1(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,  # \u8ba1\u7b97\u591a\u5c3a\u5ea6\u635f\u5931\u7684\u65f6\u5019\u662f\u5426\u8003\u8651\u906e\u6321\u533a\u57df
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=1,
                                       app_distilation_weight=1,
                                       ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])


# add appearance flow for distilation
class PWCNet_unsup_irr_bi_v5_2(tools.abstract_model):

    def __init__(self,
                 # smooth loss choose
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,
                 # smooth loss
                 smooth_order_2_weight=0,
                 # photo loss type add SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,
                 # use cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',
                 multi_scale_photo_weight=0,
                 # 'down', 'upup', 'updown'
                 multi_scale_distillation_occ=True,  # if consider occlusion mask in multiscale distilation
                 # appearance flow params
                 if_froze_pwc=False,
                 app_occ_stop_gradient=True,
                 app_loss_weight=0,
                 app_distilation_weight=0,
                 app_v2_if_app=False,  # if use app flow in each scale
                 app_v2_if_app_level=(0, 0, 0, 0, 0, 0),  # if use app flow in each level,(1/64,1/32,1/16,1/8,1/4,output)
                 app_v2_app_loss_weight=0,  # app loss weight
                 app_v2_app_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 app_v2_if_app_small_level=0,

                 if_upsample_flow=False,
                 if_upsample_flow_mask=False,
                 if_upsample_flow_output=False,
                 if_upsample_small=False,
                 if_upsample_cost_volume=False,
                 if_upsample_mask_inpainting=False,
                 if_concat_multi_scale_feature=False,
                 input_or_sp_input=1,
                 if_dense_decode=False,  # dense decoder
                 if_decoder_small=False,  # small decoder for dense connection
                 if_use_boundary_warp=True,
                 featureExtractor_if_end_relu=True,
                 featureExtractor_if_end_norm=False
                 ):
        super(PWCNet_unsup_irr_bi_v5_2, self).__init__()
        self.input_or_sp_input = input_or_sp_input  # ???sp crop?forward????????photo loss
        self.if_save_running_process = False
        self.save_running_process_dir = ''
        self.if_test = if_test
        self.if_use_boundary_warp = if_use_boundary_warp
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_photo_weight = multi_scale_photo_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self.if_decoder_small = if_decoder_small
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        if self.if_decoder_small:
            self.estimator_f_channels = (96, 64, 64, 32, 32)
            self.context_f_channels = (96, 96, 96, 64, 64, 32, 2)
        else:
            self.estimator_f_channels = (128, 128, 96, 64, 32)
            self.context_f_channels = (128, 128, 128, 96, 64, 32, 2)
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.if_end_relu = featureExtractor_if_end_relu
        self.if_end_norm = featureExtractor_if_end_norm
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs, if_end_relu=self.if_end_relu, if_end_norm=self.if_end_norm)
        # self.warping_layer = WarpingLayer()
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense_v2(self.num_ch_in, f_channels=self.estimator_f_channels)
        self.context_networks = ContextNetwork_v2_(self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels)
        self.if_concat_multi_scale_feature = if_concat_multi_scale_feature
        if if_concat_multi_scale_feature:
            self.conv_1x1_cmsf = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(128 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(96 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(64 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(32 + 32, 32, kernel_size=1, stride=1, dilation=1)])
        else:
            self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        # flow upsample module
        # flow upsample module
        class _Upsample_flow(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow, self).__init__()
                ch_in = 32
                k = ch_in
                ch_out = 64
                self.conv1 = conv(ch_in, ch_out)
                k += ch_out

                ch_out = 64
                self.conv2 = conv(k, ch_out)
                k += ch_out

                ch_out = 32
                self.conv3 = conv(k, ch_out)
                k += ch_out

                ch_out = 16
                self.conv4 = conv(k, ch_out)
                k += ch_out

                # ch_out = 64
                # self.conv5 = conv(k, ch_out)
                # k += ch_out
                self.conv_last = conv(k, 2, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                # x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x4)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.1f G, params: %.1f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.1f G, params: %.1f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v2(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v2, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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

                        ind += 1
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                _, x_out = self.dense_estimator(x)
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow_v2()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow_v2()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v3(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v3, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 64, 64, 64, 32, 32, 16, 2))
                # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                feature, x_out = self.dense_estimator(x)
                flow = flow_pre + x_out
                flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                x_out = flow + flow_fine_f
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v3()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 2, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 2, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v4(tools.abstract_model):
            def __init__(self, if_mask, if_small=False):
                super(_Upsample_flow_v4, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_small = if_small
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                if self.if_mask:
                    feature, x_out = self.dense_estimator_mask(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out[:, :2, :, :]
                    mask_out = x_out[:, 2, :, :]
                    mask_out = torch.unsqueeze(mask_out, 1)
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                        mask_out = upsample2d_flow_as(mask_out, x_raw, mode="bilinear")
                    mask_out = torch.sigmoid(mask_out)
                    return x_out, flow_out, mask_out
                else:
                    feature, x_out = self.dense_estimator(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                    return flow_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v5(tools.abstract_model):
            def __init__(self, if_mask, if_small=False, if_cost_volume=False, if_norm_before_cost_volume=True, if_mask_inpainting=False):
                super(_Upsample_flow_v5, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_mask_inpainting = if_mask_inpainting
                self.if_small = if_small
                self.if_cost_volume = if_cost_volume
                self.if_norm_before_cost_volume = if_norm_before_cost_volume
                self.warping_layer = WarpingLayer_no_div()
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if self.if_cost_volume:
                    in_C = 81
                else:
                    in_C = 64
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))

                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow, feature_1, feature_2, if_save_running_process=None, output_level_flow=None, save_running_process_dir=''):
                feature_2_warp = self.warping_layer(feature_2, flow)
                # print('v5 upsample')
                if self.if_cost_volume:
                    # if norm feature
                    if self.if_norm_before_cost_volume:
                        feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True,
                                                                                     moments_across_channels=False,
                                                                                     moments_across_images=False)
                    # correlation
                    input_feature = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
                    # tools.check_tensor(input_feature, 'input_feature')
                else:
                    input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
                    # tools.check_tensor(input_feature, 'input_feature')
                if self.if_mask:
                    # print('v5 upsample if_mask %s' % self.if_mask)
                    feature, x_out = self.dense_estimator_mask(input_feature)
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out[:, :2, :, :]
                    meta_mask = x_out[:, 2, :, :]
                    meta_mask = torch.unsqueeze(meta_mask, 1)
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        meta_mask = upsample2d_flow_as(meta_mask, output_level_flow, mode="bilinear")
                        flow = output_level_flow
                    meta_mask = torch.sigmoid(meta_mask)
                    if self.if_mask_inpainting:
                        # flow_up = tools.torch_warp(meta_mask * flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                        flow_up = tools.torch_warp(meta_mask * flow, meta_flow * (1 - meta_mask))  # + flow * meta_mask
                    else:
                        flow_up = tools.torch_warp(flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                    # print('v5 upsample if_mask %s  save_flow' % self.if_mask)
                    # self.save_flow(flow_up, '%s_flow_upbyflow' % if_save_running_process)
                    if if_save_running_process is not None:
                        # print('save results', if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_mask(save_running_process_dir, meta_mask, '%s_meta_mask' % if_save_running_process)
                else:
                    feature, x_out = self.dense_estimator(input_feature)
                    flow_fine_f = self.context_estimator(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        flow = output_level_flow
                    flow_up = tools.torch_warp(flow, meta_flow)
                    if if_save_running_process is not None:
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                return flow_up

            def output_feature(self, x):
                x = self.upsample_output_conv(x)
                return x

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        self.if_upsample_flow = if_upsample_flow
        self.if_upsample_flow_output = if_upsample_flow_output
        self.if_upsample_flow_mask = if_upsample_flow_mask
        self.if_upsample_small = if_upsample_small
        self.if_upsample_cost_volume = if_upsample_cost_volume
        self.if_upsample_mask_inpainting = if_upsample_mask_inpainting
        self.if_dense_decode = if_dense_decode
        if self.if_upsample_flow or self.if_upsample_flow_output:
            self.upsample_model_v5 = _Upsample_flow_v5(if_mask=self.if_upsample_flow_mask, if_small=self.if_upsample_small, if_cost_volume=self.if_upsample_cost_volume,
                                                       if_norm_before_cost_volume=self.if_norm_before_cost_volume, if_mask_inpainting=self.if_upsample_mask_inpainting)
        else:
            self.upsample_model_v5 = None
            self.upsample_output_conv = None

        # app flow module
        self.app_occ_stop_gradient = app_occ_stop_gradient  # stop gradient of the occ mask when inpaint
        self.app_distilation_weight = app_distilation_weight
        if app_loss_weight > 0:
            self.appflow_model = Appearance_flow_net_for_disdiilation.App_model(input_channel=7, if_share_decoder=False)
        self.app_loss_weight = app_loss_weight

        class _WarpingLayer(tools.abstract_model):

            def __init__(self):
                super(_WarpingLayer, self).__init__()

            def forward(self, x, flo):
                """
                warp an image/tensor (im2) back to im1, according to the optical flow

                x: [B, C, H, W] (im2)
                flo: [B, 2, H, W] flow

                """

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
                vgrid = grid + flo

                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
                output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
                if x.is_cuda:
                    mask = torch.ones(x.size(), requires_grad=False).cuda()
                else:
                    mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
                mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                mask = (mask >= 1.0).float()
                # mask = torch.autograd.Variable(torch.ones(x.size()))
                # if x.is_cuda:
                #     mask = mask.cuda()
                # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                #
                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
                output = output * mask
                # # nchw->>>nhwc
                # if x.is_cuda:
                #     output = output.cpu()
                # output_im = output.numpy()
                # output_im = np.transpose(output_im, (0, 2, 3, 1))
                # output_im = np.squeeze(output_im)
                return output

        self.warping_layer_inpaint = _WarpingLayer()

        initialize_msra(self.modules())
        self.if_froze_pwc = if_froze_pwc
        if self.if_froze_pwc:
            self.froze_PWC()

    def froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    @classmethod
    def save_image(cls, save_running_process_dir, image_tensor, name='image'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np)[:, :, ::-1])

    @classmethod
    def save_flow(cls, save_running_process_dir, flow_tensor, name='flow'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        # print(self.save_running_process_dir, 'save flow %s' % name)
        flow_tensor_np = tensor_to_np_for_save(flow_tensor)
        save_path = os.path.join(save_running_process_dir, name + '.png')
        # save_path = os.path.join(self.save_running_process_dir, name + '.png')
        # print(type(flow_tensor_np), flow_tensor_np.shape)
        # print(save_path)
        cv2.imwrite(save_path, tools.Show_GIF.im_norm(tools.flow_to_image(flow_tensor_np)[:, :, ::-1]))

    @classmethod
    def save_mask(cls, save_running_process_dir, image_tensor, name='mask'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np))

    def decode_level(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_up2d' % level)
                self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_up2d' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
        out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
        out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_fine_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear

    def decode_level_res(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_up2d' % level)
                self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_up2d' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
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

    def forward_2_frame_v2(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        level_iter_ls = (1, 1, 1, 1, 1)
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            level_iter = level_iter_ls[level]
            for _ in range(level_iter):
                flow_f, flow_b = self.decode_level(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2, feature_2_1x1=x2_1by1)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        return flow_f_out, flow_b_out, flows[::-1]

    def forward_2_frame_v3_dense(self, x1_raw, x2_raw):
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
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self.decode_level_res(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                           feature_2_1x1=x2_1by1)
            if level != 0 and self.if_dense_decode:
                for i_, (temp_f, temp_b) in enumerate(flows[:-1]):
                    _, _, temp_f_res, temp_b_res = self.decode_level_res(level='%s.%s' % (level, i_), flow_1=temp_f, flow_2=temp_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                         feature_2_1x1=x2_1by1)
                    flow_f_res = temp_f_res + flow_f_res
                    flow_b_res = temp_b_res + flow_b_res
            # tools.check_tensor(flow_f_res, 'flow_f_res')
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_bw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        return flow_f_out, flow_b_out, flows[::-1]

    def forward_2_frame(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        up_flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_f_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                # x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_up2d' % l)
                    self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_up2d' % l)
                if self.if_upsample_flow or self.if_upsample_flow_output:
                    up_flow_f = upsample2d_flow_as(up_flow_f, x1, mode="bilinear", if_rate=True)
                    up_flow_b = upsample2d_flow_as(up_flow_b, x1, mode="bilinear", if_rate=True)
                    if self.if_upsample_flow_mask:
                        up_flow_f_mask = upsample2d_flow_as(up_flow_f_mask, x1, mode="bilinear")
                        up_flow_b_mask = upsample2d_flow_as(up_flow_b_mask, x1, mode="bilinear")
                        _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_1by1)
                        _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = flow_f * up_flow_f_mask + self.warping_layer(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            # flow_b = flow_b * up_flow_b_mask + self.warping_layer(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            flow_f = flow_f * up_flow_f_mask + tools.torch_warp(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            flow_b = flow_b * up_flow_b_mask + tools.torch_warp(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % l)
                    else:
                        up_flow_f = self.upsample_model(up_flow_f, x1_1by1)
                        up_flow_b = self.upsample_model(up_flow_b, x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = self.warping_layer(flow_f, up_flow_f)
                            # flow_b = self.warping_layer(flow_b, up_flow_b)
                            flow_f = tools.torch_warp(flow_f, up_flow_f)
                            flow_b = tools.torch_warp(flow_b, up_flow_b)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b
            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b
            flows.append([flow_f, flow_b])
            if l == self.output_level:
                break
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            if self.if_upsample_flow_mask:
                _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_raw, if_output_level=True)
                _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_raw, if_output_level=True)
                # flow_f_out = flow_f_out * up_flow_f_mask + self.warping_layer(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                # flow_b_out = flow_b_out * up_flow_b_mask + self.warping_layer(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                flow_f_out = flow_f_out * up_flow_f_mask + tools.torch_warp(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                flow_b_out = flow_b_out * up_flow_b_mask + tools.torch_warp(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % 'out')
            else:
                up_flow_f = self.upsample_model(up_flow_f, x1_raw, if_output_level=True)
                up_flow_b = self.upsample_model(up_flow_b, x2_raw, if_output_level=True)
                # flow_f_out = self.warping_layer(flow_f_out, up_flow_f)
                # flow_b_out = self.warping_layer(flow_b_out, up_flow_b)
                flow_f_out = tools.torch_warp(flow_f_out, up_flow_f)
                flow_b_out = tools.torch_warp(flow_b_out, up_flow_b)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
        return flow_f_out, flow_b_out, flows[::-1]

    def app_refine(self, img, flow, mask):
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img * mask
        app_flow = self.appflow_model(torch.cat((input_im, img, mask), dim=1))
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        flow_restore = self.warping_layer_inpaint(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img_restore

    def app_loss(self, img_ori, img_restore, occ_mask):
        diff = img_ori - img_restore
        loss_mask = 1 - occ_mask  # only take care about the inpainting area
        diff = (torch.abs(diff) + 0.01).pow(0.4)
        diff = diff * loss_mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1_ori, im2_ori = input_dict['im1'], input_dict['im2']
        if input_dict['if_loss']:
            sp_im1_ori, sp_im2_ori = input_dict['im1_sp'], input_dict['im2_sp']
            if self.input_or_sp_input >= 1:
                im1, im2 = im1_ori, im2_ori
            elif self.input_or_sp_input > 0:
                if tools.random_flag(threshold_0_1=self.input_or_sp_input):
                    im1, im2 = im1_ori, im2_ori
                else:
                    im1, im2 = sp_im1_ori, sp_im2_ori
            else:
                im1, im2 = sp_im1_ori, sp_im2_ori
        else:
            im1, im2 = im1_ori, im2_ori

        #
        if 'if_test' in input_dict.keys():
            if_test = input_dict['if_test']
        else:
            if_test = False
        # check if save results
        if 'save_running_process' in input_dict.keys():
            self.if_save_running_process = input_dict['save_running_process']
        else:
            self.if_save_running_process = False
        if self.if_save_running_process:
            if 'process_dir' in input_dict.keys():
                self.save_running_process_dir = input_dict['process_dir']
            else:
                self.if_save_running_process = False
                self.save_running_process_dir = None
        # if show some results
        if 'if_show' in input_dict.keys():
            if_show = input_dict['if_show']
        else:
            if_show = False
        output_dict = {}
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame(im1, im2)
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v2(im1, im2)
        flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v3_dense(im1, im2)
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_pwc_out, flow_b=flow_b_pwc_out)
        if self.app_loss_weight > 0:
            if self.app_occ_stop_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            # tools.check_tensor(occ_fw, '%s' % (torch.sum(occ_fw == 1) / torch.sum(occ_fw)))
            flow_f, app_flow_1, masked_im1, im1_restore = self.app_refine(img=im1, flow=flow_f_pwc_out, mask=occ_fw)
            # tools.check_tensor(app_flow_1, 'app_flow_1')
            flow_b, app_flow_2, masked_im2, im2_restore = self.app_refine(img=im2, flow=flow_b_pwc_out, mask=occ_bw)
            app_loss = self.app_loss(im1, im1_restore, occ_fw)
            app_loss += self.app_loss(im2, im2_restore, occ_bw)
            app_loss *= self.app_loss_weight
            # tools.check_tensor(app_loss, 'app_loss')
            # print(' ')
            if input_dict['if_loss']:
                output_dict['app_loss'] = app_loss
            if self.app_distilation_weight > 0:
                flow_fw_label = flow_f.clone().detach()
                flow_bw_label = flow_b.clone().detach()
                appd_loss = network_tools.photo_loss_multi_type(x=flow_fw_label, y=flow_f_pwc_out, occ_mask=1 - occ_fw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss += network_tools.photo_loss_multi_type(x=flow_bw_label, y=flow_b_pwc_out, occ_mask=1 - occ_bw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss *= self.app_distilation_weight
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = appd_loss
                if if_test:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                else:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                    # flow_f_out = flow_f  # use app refine output
                    # flow_b_out = flow_b
            else:
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = None
                flow_f_out = flow_f
                flow_b_out = flow_b
            if if_show:
                output_dict['app_flow_1'] = app_flow_1
                output_dict['masked_im1'] = masked_im1
                output_dict['im1_restore'] = im1_restore
        else:
            if input_dict['if_loss']:
                output_dict['app_loss'] = None
            flow_f_out = flow_f_pwc_out
            flow_b_out = flow_b_pwc_out
            if if_show:
                output_dict['app_flow_1'] = None
                output_dict['masked_im1'] = None
                output_dict['im1_restore'] = None

        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # ?? smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1_ori, im2_ori
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            # ?? ?? smooth loss
            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # ?? photo loss
            if self.if_use_boundary_warp:
                # im1_warp = tools.nianjin_warp.warp_im(im2, flow_fw, start)  # warped im1 by forward flow and im2
                # im2_warp = tools.nianjin_warp.warp_im(im1, flow_bw, start)
                im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
                im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
                im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            else:
                im1_warp = tools.torch_warp(im2_ori, flow_f_out)  # warped im1 by forward flow and im2
                im2_warp = tools.torch_warp(im1_ori, flow_b_out)

            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                             photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                              photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            output_dict['photo_loss'] = photo_loss
            output_dict['im1_warp'] = im1_warp
            output_dict['im2_warp'] = im2_warp

            # ?? census loss
            if self.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss

            # ???????msd loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  #
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                # ???????photo loss? multi_scale_photo_weight
                if self.multi_scale_photo_weight > 0:
                    _, _, h_raw, w_raw = im1_s.size()
                    _, _, h_temp_crop, h_temp_crop = im1_ori.size()
                    msd_loss_ls = []
                    for i, (scale_fw, scale_bw) in enumerate(flows):
                        if self.multi_scale_distillation_style == 'down':  # ??resize???photo loss
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        elif self.multi_scale_distillation_style == 'upup':  # ???flow resize???????photo loss
                            occ_f_resize = occ_fw
                            occ_b_resize = occ_bw
                            scale_fw = upsample_flow(scale_fw, target_flow=im1_ori)
                            scale_bw = upsample_flow(scale_bw, target_flow=im2_ori)
                            im1_crop_resize = im1
                            im2_crop_resize = im2
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_s, scale_fw, start_s)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_s, scale_bw, start_s)
                        elif self.multi_scale_distillation_style == 'updown':
                            scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                            scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        else:
                            raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)

                        temp_mds_fw = network_tools.photo_loss_multi_type(im1_crop_resize, im1_resize_warp, occ_f_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_fw)
                        temp_mds_bw = network_tools.photo_loss_multi_type(im2_crop_resize, im2_resize_warp, occ_b_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_bw)
                    msd_loss = sum(msd_loss_ls)
                    msd_loss = self.multi_scale_photo_weight * msd_loss
                else:
                    msd_loss = None

            output_dict['msd_loss'] = msd_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=1,
                                       app_distilation_weight=1,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])

    @classmethod
    def demo_model_size(cls):
        from thop import profile
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=0,
                                       app_distilation_weight=0,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       if_upsample_small=False,
                                       if_upsample_cost_volume=False,
                                       if_upsample_mask_inpainting=False,
                                       if_dense_decode=True,
                                       if_decoder_small=True,
                                       ).cuda()
        # without upsample: flops: 39.893 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 50.525 G, params: 3.996 M
        |    True   |   False   |   False   |   False   |   True    |   False   |flops: 51.321 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   True    |   False   |flops: 60.498 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   False   |   False   |flops: 59.103 G, params: 3.996 M
        |    True   |   False   |   True    |   True    |   True    |   False   |flops: 47.208 G, params: 3.597 M
        |    True   |   False   |   True    |   True    |   False   |   False   |flops: 46.506 G, params: 3.573 M
        |    True   |   False   |   False   |   True    |   False   |   False   |flops: 43.339 G, params: 3.573 M
        |    True   |   True    |   True    |   True    |   True    |   False   |flops: 47.273 G, params: 3.599 M
        '''
        # dense decode=True   without upsample:flops: 144.675 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 183.825 G, params: 3.996 M
        '''
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': False}
        flops, params = profile(net, inputs=(input_dict,), verbose=False)
        print('temp(%s): flops: %.3f G, params: %.3f M' % (' ', flops / 1000 / 1000 / 1000, params / 1000 / 1000))
        # output_dict = net(input_dict)
        # print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])


# add appearance flow v2
class PWCNet_unsup_irr_bi_v5_3(tools.abstract_model):

    def __init__(self,
                 # smooth loss choose
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,
                 # smooth loss
                 smooth_order_2_weight=0,
                 # photo loss type add SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,
                 # use cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',
                 multi_scale_photo_weight=0,
                 # 'down', 'upup', 'updown'
                 multi_scale_distillation_occ=True,  # if consider occlusion mask in multiscale distilation
                 # appearance flow params
                 if_froze_pwc=False,
                 app_occ_stop_gradient=True,
                 app_loss_weight=0,
                 app_distilation_weight=0,
                 app_v2_if_app=False,  # if use app flow in each scale
                 app_v2_if_app_level=(0, 0, 0, 0, 0, 0),  # if use app flow in each level,(1/64,1/32,1/16,1/8,1/4,output)
                 app_v2_if_app_level_alpha=((0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5)),
                 app_v2_app_loss_weight=0,  # app loss weight
                 app_v2_app_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 app_v2_if_app_small_level=0,
                 app_v2_iter_num=1,  # 默认只迭代一次，但可迭代多次

                 if_upsample_flow=False,
                 if_upsample_flow_mask=False,
                 if_upsample_flow_output=False,
                 if_upsample_small=False,
                 if_upsample_cost_volume=False,
                 if_upsample_mask_inpainting=False,
                 if_concat_multi_scale_feature=False,
                 input_or_sp_input=1,
                 if_dense_decode=False,  # dense decoder
                 if_decoder_small=False,  # small decoder for dense connection
                 if_use_boundary_warp=True,
                 featureExtractor_if_end_relu=True,
                 featureExtractor_if_end_norm=False,
                 ):
        super(PWCNet_unsup_irr_bi_v5_3, self).__init__()
        self.input_or_sp_input = input_or_sp_input  # ???sp crop?forward????????photo loss
        self.if_save_running_process = False
        self.save_running_process_dir = ''
        self.if_test = if_test
        self.if_use_boundary_warp = if_use_boundary_warp
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_photo_weight = multi_scale_photo_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self.if_decoder_small = if_decoder_small
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        if self.if_decoder_small:
            self.estimator_f_channels = (96, 64, 64, 32, 32)
            self.context_f_channels = (96, 96, 96, 64, 64, 32, 2)
        else:
            self.estimator_f_channels = (128, 128, 96, 64, 32)
            self.context_f_channels = (128, 128, 128, 96, 64, 32, 2)
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.if_end_relu = featureExtractor_if_end_relu
        self.if_end_norm = featureExtractor_if_end_norm
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs, if_end_relu=self.if_end_relu, if_end_norm=self.if_end_norm)
        # self.warping_layer = WarpingLayer()
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense_v2(self.num_ch_in, f_channels=self.estimator_f_channels)
        self.context_networks = ContextNetwork_v2_(self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels)
        self.if_concat_multi_scale_feature = if_concat_multi_scale_feature
        if if_concat_multi_scale_feature:
            self.conv_1x1_cmsf = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(128 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(96 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(64 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(32 + 32, 32, kernel_size=1, stride=1, dilation=1)])
        else:
            self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        # flow upsample module
        # flow upsample module
        class _Upsample_flow(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow, self).__init__()
                ch_in = 32
                k = ch_in
                ch_out = 64
                self.conv1 = conv(ch_in, ch_out)
                k += ch_out

                ch_out = 64
                self.conv2 = conv(k, ch_out)
                k += ch_out

                ch_out = 32
                self.conv3 = conv(k, ch_out)
                k += ch_out

                ch_out = 16
                self.conv4 = conv(k, ch_out)
                k += ch_out

                # ch_out = 64
                # self.conv5 = conv(k, ch_out)
                # k += ch_out
                self.conv_last = conv(k, 2, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                # x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x4)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.1f G, params: %.1f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.1f G, params: %.1f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v2(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v2, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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

                        ind += 1
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                _, x_out = self.dense_estimator(x)
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow_v2()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow_v2()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v3(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v3, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 64, 64, 64, 32, 32, 16, 2))
                # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                feature, x_out = self.dense_estimator(x)
                flow = flow_pre + x_out
                flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                x_out = flow + flow_fine_f
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v3()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 2, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 2, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v4(tools.abstract_model):
            def __init__(self, if_mask, if_small=False):
                super(_Upsample_flow_v4, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_small = if_small
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                if self.if_mask:
                    feature, x_out = self.dense_estimator_mask(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out[:, :2, :, :]
                    mask_out = x_out[:, 2, :, :]
                    mask_out = torch.unsqueeze(mask_out, 1)
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                        mask_out = upsample2d_flow_as(mask_out, x_raw, mode="bilinear")
                    mask_out = torch.sigmoid(mask_out)
                    return x_out, flow_out, mask_out
                else:
                    feature, x_out = self.dense_estimator(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                    return flow_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v5(tools.abstract_model):
            def __init__(self, if_mask, if_small=False, if_cost_volume=False, if_norm_before_cost_volume=True, if_mask_inpainting=False):
                super(_Upsample_flow_v5, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_mask_inpainting = if_mask_inpainting
                self.if_small = if_small
                self.if_cost_volume = if_cost_volume
                self.if_norm_before_cost_volume = if_norm_before_cost_volume
                self.warping_layer = WarpingLayer_no_div()
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if self.if_cost_volume:
                    in_C = 81
                else:
                    in_C = 64
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))

                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow, feature_1, feature_2, if_save_running_process=None, output_level_flow=None, save_running_process_dir=''):
                feature_2_warp = self.warping_layer(feature_2, flow)
                # print('v5 upsample')
                if self.if_cost_volume:
                    # if norm feature
                    if self.if_norm_before_cost_volume:
                        feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True,
                                                                                     moments_across_channels=False,
                                                                                     moments_across_images=False)
                    # correlation
                    input_feature = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
                    # tools.check_tensor(input_feature, 'input_feature')
                else:
                    input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
                    # tools.check_tensor(input_feature, 'input_feature')
                if self.if_mask:
                    # print('v5 upsample if_mask %s' % self.if_mask)
                    feature, x_out = self.dense_estimator_mask(input_feature)
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out[:, :2, :, :]
                    meta_mask = x_out[:, 2, :, :]
                    meta_mask = torch.unsqueeze(meta_mask, 1)
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        meta_mask = upsample2d_flow_as(meta_mask, output_level_flow, mode="bilinear")
                        flow = output_level_flow
                    meta_mask = torch.sigmoid(meta_mask)
                    if self.if_mask_inpainting:
                        # flow_up = tools.torch_warp(meta_mask * flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                        flow_up = tools.torch_warp(meta_mask * flow, meta_flow * (1 - meta_mask))  # + flow * meta_mask
                    else:
                        flow_up = tools.torch_warp(flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                    # print('v5 upsample if_mask %s  save_flow' % self.if_mask)
                    # self.save_flow(flow_up, '%s_flow_upbyflow' % if_save_running_process)
                    if if_save_running_process is not None:
                        # print('save results', if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_mask(save_running_process_dir, meta_mask, '%s_meta_mask' % if_save_running_process)
                else:
                    feature, x_out = self.dense_estimator(input_feature)
                    flow_fine_f = self.context_estimator(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        flow = output_level_flow
                    flow_up = tools.torch_warp(flow, meta_flow)
                    if if_save_running_process is not None:
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                return flow_up

            def output_feature(self, x):
                x = self.upsample_output_conv(x)
                return x

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        self.if_upsample_flow = if_upsample_flow
        self.if_upsample_flow_output = if_upsample_flow_output
        self.if_upsample_flow_mask = if_upsample_flow_mask
        self.if_upsample_small = if_upsample_small
        self.if_upsample_cost_volume = if_upsample_cost_volume
        self.if_upsample_mask_inpainting = if_upsample_mask_inpainting
        self.if_dense_decode = if_dense_decode
        if self.if_upsample_flow or self.if_upsample_flow_output:
            self.upsample_model_v5 = _Upsample_flow_v5(if_mask=self.if_upsample_flow_mask, if_small=self.if_upsample_small, if_cost_volume=self.if_upsample_cost_volume,
                                                       if_norm_before_cost_volume=self.if_norm_before_cost_volume, if_mask_inpainting=self.if_upsample_mask_inpainting)
        else:
            self.upsample_model_v5 = None
            self.upsample_output_conv = None

        # app flow module
        self.app_occ_stop_gradient = app_occ_stop_gradient  # stop gradient of the occ mask when inpaint
        self.app_distilation_weight = app_distilation_weight  # fangqi, i will do not use this part
        if app_loss_weight > 0:
            self.appflow_model = Appearance_flow_net_for_disdiilation.App_model(input_channel=7, if_share_decoder=False)
        self.app_loss_weight = app_loss_weight

        self.app_v2_if_app = app_v2_if_app,  # if use app flow
        self.app_v2_if_app_small_level = app_v2_if_app_small_level
        self.app_v2_iter_num = app_v2_iter_num
        if self.app_v2_if_app:
            self.app_v2_flow_model = Appearance_flow_net_for_disdiilation.App_model_small(input_channel=39, if_share_decoder=True, small_level=self.app_v2_if_app_small_level)
        self.app_v2_if_app_level = app_v2_if_app_level  # if use app flow in each level, (6 ge True/Flase)
        self.app_v2_if_app_level_alpha = app_v2_if_app_level_alpha
        # build occ models
        self.occ_check_model_ls = []
        for i in range(len(self.app_v2_if_app_level_alpha)):
            occ_alpha_1 = self.app_v2_if_app_level_alpha[i][0]
            occ_alpha_2 = self.app_v2_if_app_level_alpha[i][1]
            model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2, obj_out_all=occ_check_obj_out_all)
            self.occ_check_model_ls.append(model)
        self.app_v2_app_loss_weight = app_v2_app_loss_weight  # app loss weight
        self.app_v2_app_loss_type = app_v2_app_loss_type  # abs_robust, charbonnier,L1, SSIM

        class _WarpingLayer(tools.abstract_model):

            def __init__(self):
                super(_WarpingLayer, self).__init__()

            def forward(self, x, flo):
                """
                warp an image/tensor (im2) back to im1, according to the optical flow

                x: [B, C, H, W] (im2)
                flo: [B, 2, H, W] flow

                """

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
                vgrid = grid + flo

                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
                output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
                if x.is_cuda:
                    mask = torch.ones(x.size(), requires_grad=False).cuda()
                else:
                    mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
                mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                mask = (mask >= 1.0).float()
                # mask = torch.autograd.Variable(torch.ones(x.size()))
                # if x.is_cuda:
                #     mask = mask.cuda()
                # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                #
                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
                output = output * mask
                # # nchw->>>nhwc
                # if x.is_cuda:
                #     output = output.cpu()
                # output_im = output.numpy()
                # output_im = np.transpose(output_im, (0, 2, 3, 1))
                # output_im = np.squeeze(output_im)
                return output

        self.warping_layer_inpaint = _WarpingLayer()

        initialize_msra(self.modules())
        self.if_froze_pwc = if_froze_pwc
        if self.if_froze_pwc:
            self.froze_PWC()

    def froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    @classmethod
    def save_image(cls, save_running_process_dir, image_tensor, name='image'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np)[:, :, ::-1])

    @classmethod
    def save_flow(cls, save_running_process_dir, flow_tensor, name='flow'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        # print(self.save_running_process_dir, 'save flow %s' % name)
        flow_tensor_np = tensor_to_np_for_save(flow_tensor)
        save_path = os.path.join(save_running_process_dir, name + '.png')
        # save_path = os.path.join(self.save_running_process_dir, name + '.png')
        # print(type(flow_tensor_np), flow_tensor_np.shape)
        # print(save_path)
        # cv2.imwrite(save_path, tools.Show_GIF.im_norm(tools.flow_to_image(flow_tensor_np)[:, :, ::-1]))
        cv2.imwrite(save_path, tools.flow_to_image(flow_tensor_np)[:, :, ::-1])

    @classmethod
    def save_mask(cls, save_running_process_dir, image_tensor, name='mask'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np))

    def decode_level(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_up2d' % level)
                self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_up2d' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
        out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
        out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_fine_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear

    def decode_level_res(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            # if self.if_save_running_process:
            #     self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_linear_up' % level)
            #     self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_linear_up' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
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

    def forward_2_frame_v2(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        level_iter_ls = (1, 1, 1, 1, 1)
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            level_iter = level_iter_ls[level]
            for _ in range(level_iter):
                flow_f, flow_b = self.decode_level(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2, feature_2_1x1=x2_1by1)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        return flow_f_out, flow_b_out, flows[::-1]

    def forward_2_frame_v3_dense(self, x1_raw, x2_raw, if_loss=False):
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
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        app_loss = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self.decode_level_res(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                           feature_2_1x1=x2_1by1)
            # do not use dense_decoder, no use
            if level != 0 and self.if_dense_decode:
                for i_, (temp_f, temp_b) in enumerate(flows[:-1]):
                    _, _, temp_f_res, temp_b_res = self.decode_level_res(level='%s.%s' % (level, i_), flow_1=temp_f, flow_2=temp_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                         feature_2_1x1=x2_1by1)
                    flow_f_res = temp_f_res + flow_f_res
                    flow_b_res = temp_b_res + flow_b_res
            # tools.check_tensor(flow_f_res, 'flow_f_res')
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            # app refine
            # print(level, 'level ')
            if self.app_v2_if_app and self.app_v2_if_app_level[level] > 0:
                # occ_1, occ_2 = self.occ_check_model_ls[level](flow_f=flow_f, flow_b=flow_b)
                # occ_1, occ_2 = self.occ_check_model(flow_f=flow_f, flow_b=flow_b)  # 0 in occ area, 1 in others
                # occ_1, occ_2 = self.occ_check_model.forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, alpha1=self.app_v2_if_app_level_alpha[level][0],
                #                                                                alpha2=self.app_v2_if_app_level_alpha[level][1])
                occ_1, occ_2 = self.occ_check_model_ls[level](flow_f=flow_f, flow_b=flow_b)
                flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f, occ_1, x1_1by1, refine_level=level)
                flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b, occ_2, x2_1by1, refine_level=level)
                if self.app_v2_iter_num > 1:
                    for ind in range(self.app_v2_iter_num - 1):
                        occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_restore, flow_b=flow_b_restore)
                        flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_restore, occ_1, x1_1by1, refine_level=-1)
                        flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_restore, occ_2, x2_1by1, refine_level=-1)
                if self.if_save_running_process:
                    im_1_restore = tools.torch_warp(im_1_masked, app_f_flow)
                    im_2_restore = tools.torch_warp(im_2_masked, app_b_flow)
                    self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_decode' % level)
                    self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_decode' % level)
                    self.save_flow(self.save_running_process_dir, flow_f_restore, '%s_flow_f_appref' % level)
                    self.save_flow(self.save_running_process_dir, flow_b_restore, '%s_flow_b_appref' % level)
                    self.save_flow(self.save_running_process_dir, app_f_flow, '%s_flow_f_appflow' % level)
                    self.save_flow(self.save_running_process_dir, app_b_flow, '%s_flow_b_appflow' % level)
                    self.save_mask(self.save_running_process_dir, occ_1, '%s_occ_1' % level)
                    self.save_mask(self.save_running_process_dir, occ_2, '%s_occ_2' % level)
                    self.save_image(self.save_running_process_dir, im_1_restore, '%s_im_1_restore' % level)
                    self.save_image(self.save_running_process_dir, im_1_masked, '%s_im_1_masked' % level)
                    self.save_image(self.save_running_process_dir, im_2_restore, '%s_im_2_restore' % level)
                    self.save_image(self.save_running_process_dir, im_2_masked, '%s_im_2_masked' % level)
                if if_loss:
                    temp_app_loss = self.app_v2_loss(im_1_resize, app_f_flow, occ_1) + self.app_v2_loss(im_2_resize, app_b_flow, occ_2)
                    if app_loss is None:
                        app_loss = temp_app_loss * self.app_v2_if_app_level[level]
                    else:
                        app_loss += temp_app_loss * self.app_v2_if_app_level[level]

                flow_f = flow_f_restore
                flow_b = flow_b_restore
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_f, '%s_scale_output_flow_f' % level)
                self.save_flow(self.save_running_process_dir, flow_b, '%s_scale_output_flow_b' % level)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_linear_up')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_linear_up')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_bw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        if self.app_v2_if_app and self.app_v2_if_app_level[-1] > 0:
            app_feature_1_1x1 = self.app_v2_flow_model.output_feature(x1_raw)
            app_feature_2_1x1 = self.app_v2_flow_model.output_feature(x2_raw)
            # occ_1, occ_2 = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)  # 0 in occ area, 1 in others
            # occ_1, occ_2 = self.occ_check_model.forward_backward_occ_check(flow_fw=flow_f_out, flow_bw=flow_b_out, alpha1=self.app_v2_if_app_level_alpha[-1][0],
            #                                                                alpha2=self.app_v2_if_app_level_alpha[-1][1])
            occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_out, flow_b=flow_b_out)
            flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_out, occ_1, app_feature_1_1x1, refine_level=-1)
            # tools.check_tensor(app_f_flow,'app_f_flow  forward')
            flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_out, occ_2, app_feature_2_1x1, refine_level=-1)
            if self.app_v2_iter_num > 1:
                for ind in range(self.app_v2_iter_num - 1):
                    occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_restore, flow_b=flow_b_restore)
                    flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_restore, occ_1, app_feature_1_1x1, refine_level=-1)
                    flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_restore, occ_2, app_feature_2_1x1, refine_level=-1)
            flow_f_out = flow_f_restore
            flow_b_out = flow_b_restore
            im_1_restore = tools.torch_warp(im_1_masked, app_f_flow)
            im_2_restore = tools.torch_warp(im_2_masked, app_b_flow)
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_f_restore, '%s_flow_f_appref' % 'out')
                self.save_flow(self.save_running_process_dir, flow_b_restore, '%s_flow_b_appref' % 'out')
                self.save_flow(self.save_running_process_dir, app_f_flow, '%s_flow_f_appflow' % 'out')
                self.save_mask(self.save_running_process_dir, occ_1, '%s_occ_1' % 'out')
                self.save_mask(self.save_running_process_dir, occ_2, '%s_occ_2' % 'out')
                self.save_image(self.save_running_process_dir, im_1_restore, '%s_im_1_restore' % 'out')
                self.save_image(self.save_running_process_dir, im_1_masked, '%s_im_1_masked' % 'out')
                self.save_image(self.save_running_process_dir, im_2_restore, '%s_im_2_restore' % 'out')
                self.save_image(self.save_running_process_dir, im_2_masked, '%s_im_2_masked' % 'out')
            if if_loss:
                temp_app_loss = self.app_v2_loss(im_1_resize, app_f_flow, occ_1) + self.app_v2_loss(im_2_resize, app_b_flow, occ_2)
                if app_loss is None:
                    app_loss = temp_app_loss * self.app_v2_if_app_level[-1]
                else:
                    app_loss += temp_app_loss * self.app_v2_if_app_level[-1]
            return flow_f_out, flow_b_out, flows[::-1], app_loss, app_f_flow, im_1_masked, im_1_restore
        else:
            return flow_f_out, flow_b_out, flows[::-1], app_loss, None, None, None

    def forward_2_frame(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        up_flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_f_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                # x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_up2d' % l)
                    self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_up2d' % l)
                if self.if_upsample_flow or self.if_upsample_flow_output:
                    up_flow_f = upsample2d_flow_as(up_flow_f, x1, mode="bilinear", if_rate=True)
                    up_flow_b = upsample2d_flow_as(up_flow_b, x1, mode="bilinear", if_rate=True)
                    if self.if_upsample_flow_mask:
                        up_flow_f_mask = upsample2d_flow_as(up_flow_f_mask, x1, mode="bilinear")
                        up_flow_b_mask = upsample2d_flow_as(up_flow_b_mask, x1, mode="bilinear")
                        _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_1by1)
                        _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = flow_f * up_flow_f_mask + self.warping_layer(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            # flow_b = flow_b * up_flow_b_mask + self.warping_layer(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            flow_f = flow_f * up_flow_f_mask + tools.torch_warp(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            flow_b = flow_b * up_flow_b_mask + tools.torch_warp(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % l)
                    else:
                        up_flow_f = self.upsample_model(up_flow_f, x1_1by1)
                        up_flow_b = self.upsample_model(up_flow_b, x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = self.warping_layer(flow_f, up_flow_f)
                            # flow_b = self.warping_layer(flow_b, up_flow_b)
                            flow_f = tools.torch_warp(flow_f, up_flow_f)
                            flow_b = tools.torch_warp(flow_b, up_flow_b)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b
            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b
            flows.append([flow_f, flow_b])
            if l == self.output_level:
                break
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            if self.if_upsample_flow_mask:
                _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_raw, if_output_level=True)
                _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_raw, if_output_level=True)
                # flow_f_out = flow_f_out * up_flow_f_mask + self.warping_layer(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                # flow_b_out = flow_b_out * up_flow_b_mask + self.warping_layer(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                flow_f_out = flow_f_out * up_flow_f_mask + tools.torch_warp(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                flow_b_out = flow_b_out * up_flow_b_mask + tools.torch_warp(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % 'out')
            else:
                up_flow_f = self.upsample_model(up_flow_f, x1_raw, if_output_level=True)
                up_flow_b = self.upsample_model(up_flow_b, x2_raw, if_output_level=True)
                # flow_f_out = self.warping_layer(flow_f_out, up_flow_f)
                # flow_b_out = self.warping_layer(flow_b_out, up_flow_b)
                flow_f_out = tools.torch_warp(flow_f_out, up_flow_f)
                flow_b_out = tools.torch_warp(flow_b_out, up_flow_b)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
        return flow_f_out, flow_b_out, flows[::-1]

    def app_v2_refine(self, img_raw, flow, mask, feature_1x1, refine_level=0):
        if img_raw.shape[-2:] != flow.shape[-2:]:
            # _, _, h_raw, w_raw = im1_s.size()
            img1_resize = upsample2d_as(img_raw, flow, mode="bilinear")
        else:
            img1_resize = img_raw
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img1_resize * mask
        # app_flow = self.app_v2_flow_model(torch.cat((feature_1x1, input_im, img1_resize, mask), dim=1), refine_level=refine_level)
        app_flow = self.app_v2_flow_model(torch.cat((input_im, img1_resize, feature_1x1, mask), dim=1), refine_level=refine_level)
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = self.warping_layer_inpaint(flow, app_flow)
        flow_restore = tools.torch_warp(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        # img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img1_resize

    def app_v2_loss(self, img_ori, app_flow, occ_mask):
        img_label = img_ori.clone().detach()

        mask_im = img_label * occ_mask
        img_restore = tools.torch_warp(mask_im, app_flow)
        # diff = img_label - img_restore
        # loss_mask = 1 - occ_mask
        # diff = img_ori - img_restore
        # loss_mask = 1 - occ_mask  # only take care about the inpainting area

        # loss_mean = network_tools.photo_loss_multi_type(img_ori, img_restore, loss_mask, photo_loss_type=self.app_v2_app_loss_type,
        #                                                 photo_loss_delta=0.4, photo_loss_use_occ=True)
        # loss_mean = network_tools.compute_inpaint_photo_loss_mask(img_raw=img_label, img_restore=img_restore,
        #                                                           mask=occ_mask, if_l1=False)
        loss_mean = network_tools.compute_inpaint_photo_loss_mask_multi_type(img_raw=img_label, img_restore=img_restore, mask=occ_mask,
                                                                             photo_loss_type=self.app_v2_app_loss_type,
                                                                             )
        #
        # diff = (torch.abs(diff) + 0.01).pow(0.4)
        # diff = diff * loss_mask
        # diff_sum = torch.sum(diff)
        # loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def app_refine(self, img, flow, mask):
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img * mask
        app_flow = self.appflow_model(torch.cat((input_im, img, mask), dim=1))
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        flow_restore = self.warping_layer_inpaint(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img_restore

    def app_loss(self, img_ori, img_restore, occ_mask):
        diff = img_ori - img_restore
        loss_mask = 1 - occ_mask  # only take care about the inpainting area
        diff = (torch.abs(diff) + 0.01).pow(0.4)
        diff = diff * loss_mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1_ori, im2_ori = input_dict['im1'], input_dict['im2']
        if input_dict['if_loss']:
            sp_im1_ori, sp_im2_ori = input_dict['im1_sp'], input_dict['im2_sp']
            if self.input_or_sp_input >= 1:
                im1, im2 = im1_ori, im2_ori
            elif self.input_or_sp_input > 0:
                if tools.random_flag(threshold_0_1=self.input_or_sp_input):
                    im1, im2 = im1_ori, im2_ori
                else:
                    im1, im2 = sp_im1_ori, sp_im2_ori
            else:
                im1, im2 = sp_im1_ori, sp_im2_ori
        else:
            im1, im2 = im1_ori, im2_ori

        #
        if 'if_test' in input_dict.keys():
            if_test = input_dict['if_test']
        else:
            if_test = False
        # check if save results
        if 'save_running_process' in input_dict.keys():
            self.if_save_running_process = input_dict['save_running_process']
        else:
            self.if_save_running_process = False
        if self.if_save_running_process:
            if 'process_dir' in input_dict.keys():
                self.save_running_process_dir = input_dict['process_dir']
            else:
                self.if_save_running_process = False
                self.save_running_process_dir = None
        # if show some results
        if 'if_show' in input_dict.keys():
            if_show = input_dict['if_show']
        else:
            if_show = False
        output_dict = {}
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame(im1, im2)
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v2(im1, im2)
        flow_f_pwc_out, flow_b_pwc_out, flows, app_loss, app_flow_1, masked_im1, im1_restore = self.forward_2_frame_v3_dense(im1, im2, if_loss=input_dict['if_loss'])
        if if_show:
            output_dict['app_flow_1'] = app_flow_1
            output_dict['masked_im1'] = masked_im1
            output_dict['im1_restore'] = im1_restore
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_pwc_out, flow_b=flow_b_pwc_out)  # 0 in occ area, 1 in others
        if self.app_loss_weight > 0:
            if self.app_occ_stop_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            # tools.check_tensor(occ_fw, '%s' % (torch.sum(occ_fw == 1) / torch.sum(occ_fw)))
            flow_f, app_flow_1, masked_im1, im1_restore = self.app_refine(img=im1, flow=flow_f_pwc_out, mask=occ_fw)
            # tools.check_tensor(app_flow_1, 'app_flow_1')
            flow_b, app_flow_2, masked_im2, im2_restore = self.app_refine(img=im2, flow=flow_b_pwc_out, mask=occ_bw)
            app_loss = self.app_loss(im1, im1_restore, occ_fw)
            app_loss += self.app_loss(im2, im2_restore, occ_bw)
            app_loss *= self.app_loss_weight
            # tools.check_tensor(app_loss, 'app_loss')
            # print(' ')
            if input_dict['if_loss']:
                output_dict['app_loss'] = app_loss
            if self.app_distilation_weight > 0:
                flow_fw_label = flow_f.clone().detach()
                flow_bw_label = flow_b.clone().detach()
                appd_loss = network_tools.photo_loss_multi_type(x=flow_fw_label, y=flow_f_pwc_out, occ_mask=1 - occ_fw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss += network_tools.photo_loss_multi_type(x=flow_bw_label, y=flow_b_pwc_out, occ_mask=1 - occ_bw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss *= self.app_distilation_weight
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = appd_loss
                if if_test:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                else:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                    # flow_f_out = flow_f  # use app refine output
                    # flow_b_out = flow_b
            else:
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = None
                flow_f_out = flow_f
                flow_b_out = flow_b
            if if_show:
                output_dict['app_flow_1'] = app_flow_1
                output_dict['masked_im1'] = masked_im1
                output_dict['im1_restore'] = im1_restore
        else:
            if input_dict['if_loss']:
                output_dict['app_loss'] = None
            flow_f_out = flow_f_pwc_out
            flow_b_out = flow_b_pwc_out
            # if if_show:
            #     output_dict['app_flow_1'] = None
            #     output_dict['masked_im1'] = None
            #     output_dict['im1_restore'] = None
        if_reverse_occ = True  # False
        if if_reverse_occ:
            if_do_reverse = False
            if if_do_reverse:
                f_b_w = tools.torch_warp(flow_b_out, flow_f_out)  # 用f 把b光流 warp过来了
                f_f_w = tools.torch_warp(flow_f_out, flow_b_out)  # 用b 把f光流 warp过来了
                f_b_w = -f_b_w  # 方向取反
                f_f_w = -f_f_w
            else:
                f_b_w = tools.torch_warp(flow_f_out, flow_f_out)  # 用f 把b光流 warp过来了
                f_f_w = tools.torch_warp(flow_b_out, flow_b_out)  # 用b 把f光流 warp过来了
            # 按照occlusion mask进行融合
            temp_f = occ_fw * flow_f_out + (1 - occ_fw) * f_b_w
            temp_b = occ_bw * flow_b_out + (1 - occ_bw) * f_f_w
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, f_b_w, 'warpflow_b_by_f')
                self.save_flow(self.save_running_process_dir, f_f_w, 'warpflow_f_by_b')
                self.save_flow(self.save_running_process_dir, temp_f, 'fuse_occ_flow_f')
                self.save_flow(self.save_running_process_dir, temp_b, 'fuse_occ_flow_b')
            flow_f_out = temp_f
            flow_b_out = temp_b
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # ?? smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1_ori, im2_ori
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            # ?? ?? smooth loss
            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                elif self.smooth_type == 'delta':
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # ?? photo loss
            if self.if_use_boundary_warp:
                # im1_warp = tools.nianjin_warp.warp_im(im2, flow_fw, start)  # warped im1 by forward flow and im2
                # im2_warp = tools.nianjin_warp.warp_im(im1, flow_bw, start)
                im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
                im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
                im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            else:
                im1_warp = tools.torch_warp(im2_ori, flow_f_out)  # warped im1 by forward flow and im2
                im2_warp = tools.torch_warp(im1_ori, flow_b_out)

            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = network_tools.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                             photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            photo_loss += network_tools.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                              photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            output_dict['photo_loss'] = photo_loss
            output_dict['im1_warp'] = im1_warp
            output_dict['im2_warp'] = im2_warp

            # ?? census loss
            if self.photo_loss_census_weight > 0:
                census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                              loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                               charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss

            # ???????msd loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  #
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                # ???????photo loss? multi_scale_photo_weight
                if self.multi_scale_photo_weight > 0:
                    _, _, h_raw, w_raw = im1_s.size()
                    _, _, h_temp_crop, h_temp_crop = im1_ori.size()
                    msd_loss_ls = []
                    for i, (scale_fw, scale_bw) in enumerate(flows):
                        if self.multi_scale_distillation_style == 'down':  # ??resize???photo loss
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        elif self.multi_scale_distillation_style == 'upup':  # ???flow resize???????photo loss
                            occ_f_resize = occ_fw
                            occ_b_resize = occ_bw
                            scale_fw = upsample_flow(scale_fw, target_flow=im1_ori)
                            scale_bw = upsample_flow(scale_bw, target_flow=im2_ori)
                            im1_crop_resize = im1
                            im2_crop_resize = im2
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_s, scale_fw, start_s)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_s, scale_bw, start_s)
                        elif self.multi_scale_distillation_style == 'updown':
                            scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                            scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        else:
                            raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)

                        temp_mds_fw = network_tools.photo_loss_multi_type(im1_crop_resize, im1_resize_warp, occ_f_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_fw)
                        temp_mds_bw = network_tools.photo_loss_multi_type(im2_crop_resize, im2_resize_warp, occ_b_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_bw)
                    msd_loss = sum(msd_loss_ls)
                    msd_loss = self.multi_scale_photo_weight * msd_loss
                else:
                    msd_loss = None

            output_dict['msd_loss'] = msd_loss

            # appearance flow restore loss
            if app_loss is None:
                pass
            else:
                app_loss = app_loss * self.app_v2_app_loss_weight
                output_dict['app_loss'] = app_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=1,
                                       app_distilation_weight=1,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])

    @classmethod
    def demo_model_size_app_module(cls):
        from thop import profile
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,

                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_v2_if_app=True,  # if use app flow in each scale
                                       app_v2_if_app_level=(False, True, True, True, True, True),  # if use app flow in each level,(1/64,1/32,1/16,1/8,1/4,output)
                                       app_v2_app_loss_weight=1,  # app loss weight
                                       app_v2_app_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                                       app_v2_if_app_small_level=1,

                                       multi_scale_distillation_weight=0,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_photo_weight=0,

                                       featureExtractor_if_end_relu=True,
                                       featureExtractor_if_end_norm=True
                                       ).cuda()
        # without upsample: flops: 39.893 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 50.525 G, params: 3.996 M

        '''

        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch, 'im1_sp': im_torch, 'im2_sp': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        # out=net(input_dict)
        flops, params = profile(net, inputs=(input_dict,), verbose=False)
        print('temp(%s): flops: %.3f G, params: %.3f M' % (' ', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def demo_model_size_upsample(cls):
        from thop import profile
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=0,
                                       app_distilation_weight=1,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       if_upsample_small=False,
                                       if_upsample_cost_volume=False,
                                       if_upsample_mask_inpainting=False,
                                       if_dense_decode=True,
                                       if_decoder_small=True,
                                       ).cuda()
        # without upsample: flops: 39.893 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 50.525 G, params: 3.996 M
        |    True   |   False   |   False   |   False   |   True    |   False   |flops: 51.321 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   True    |   False   |flops: 60.498 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   False   |   False   |flops: 59.103 G, params: 3.996 M
        |    True   |   False   |   True    |   True    |   True    |   False   |flops: 47.208 G, params: 3.597 M
        |    True   |   False   |   True    |   True    |   False   |   False   |flops: 46.506 G, params: 3.573 M
        |    True   |   False   |   False   |   True    |   False   |   False   |flops: 43.339 G, params: 3.573 M
        |    True   |   True    |   True    |   True    |   True    |   False   |flops: 47.273 G, params: 3.599 M
        '''
        # dense decode=True   without upsample:flops: 144.675 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 183.825 G, params: 3.996 M
        '''
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch, 'im1_sp': im_torch, 'im2_sp': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': False}
        flops, params = profile(net, inputs=(input_dict,), verbose=False)
        print('temp(%s): flops: %.3f G, params: %.3f M' % (' ', flops / 1000 / 1000 / 1000, params / 1000 / 1000))
        # output_dict = net(input_dict)
        # print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])


class PWCNet_unsup_irr_bi_v5_3_single_forward(tools.abstract_model):

    def __init__(self,
                 # smooth loss choose
                 occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True, occ_check_obj_out_all='obj', stop_occ_gradient=False,
                 smooth_level='final',  # final or 1/4
                 smooth_type='edge',  # edge or delta
                 smooth_order_1_weight=1,
                 # smooth loss
                 smooth_order_2_weight=0,
                 # photo loss type add SSIM
                 photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 photo_loss_delta=0.4,
                 photo_loss_use_occ=False,
                 photo_loss_census_weight=0,
                 # use cost volume norm
                 if_norm_before_cost_volume=False,
                 norm_moments_across_channels=True,
                 norm_moments_across_images=True,
                 if_test=False,
                 multi_scale_distillation_weight=0,
                 multi_scale_distillation_style='upup',
                 multi_scale_photo_weight=0,
                 # 'down', 'upup', 'updown'
                 multi_scale_distillation_occ=True,  # if consider occlusion mask in multiscale distilation
                 # appearance flow params
                 if_froze_pwc=False,
                 app_occ_stop_gradient=True,
                 app_loss_weight=0,
                 app_distilation_weight=0,
                 app_v2_if_app=False,  # if use app flow in each scale
                 app_v2_if_app_level=(0, 0, 0, 0, 0, 0),  # if use app flow in each level,(1/64,1/32,1/16,1/8,1/4,output)
                 app_v2_if_app_level_alpha=((0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5)),
                 app_v2_app_loss_weight=0,  # app loss weight
                 app_v2_app_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                 app_v2_if_app_small_level=0,
                 app_v2_iter_num=1,  # 默认只迭代一次，但可迭代多次

                 if_upsample_flow=False,
                 if_upsample_flow_mask=False,
                 if_upsample_flow_output=False,
                 if_upsample_small=False,
                 if_upsample_cost_volume=False,
                 if_upsample_mask_inpainting=False,
                 if_concat_multi_scale_feature=False,
                 input_or_sp_input=1,
                 if_dense_decode=False,  # dense decoder
                 if_decoder_small=False,  # small decoder for dense connection
                 if_use_boundary_warp=True,
                 featureExtractor_if_end_relu=True,
                 featureExtractor_if_end_norm=False,
                 learn_direction='fw',
                 ):
        super(PWCNet_unsup_irr_bi_v5_3_single_forward, self).__init__()
        self.learn_direction = learn_direction
        assert self.learn_direction in ['fw', 'bw']
        self.input_or_sp_input = input_or_sp_input  # ???sp crop?forward????????photo loss
        self.if_save_running_process = False
        self.save_running_process_dir = ''
        self.if_test = if_test
        self.if_use_boundary_warp = if_use_boundary_warp
        self.multi_scale_distillation_weight = multi_scale_distillation_weight
        self.multi_scale_photo_weight = multi_scale_photo_weight
        self.multi_scale_distillation_style = multi_scale_distillation_style
        self.multi_scale_distillation_occ = multi_scale_distillation_occ
        # smooth
        self.occ_check_model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2,
                                                     sum_abs_or_squar=occ_check_sum_abs_or_squar, obj_out_all=occ_check_obj_out_all)
        self.smooth_level = smooth_level
        self.smooth_type = smooth_type
        self.smooth_order_1_weight = smooth_order_1_weight
        self.smooth_order_2_weight = smooth_order_2_weight

        # photo loss
        self.photo_loss_type = photo_loss_type
        self.photo_loss_census_weight = photo_loss_census_weight
        self.photo_loss_use_occ = photo_loss_use_occ  # if use occ mask in photo loss
        self.photo_loss_delta = photo_loss_delta  # delta in photo loss function
        self.stop_occ_gradient = stop_occ_gradient

        self.if_norm_before_cost_volume = if_norm_before_cost_volume
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images

        self.if_decoder_small = if_decoder_small
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        if self.if_decoder_small:
            self.estimator_f_channels = (96, 64, 64, 32, 32)
            self.context_f_channels = (96, 96, 96, 64, 64, 32, 2)
        else:
            self.estimator_f_channels = (128, 128, 96, 64, 32)
            self.context_f_channels = (128, 128, 128, 96, 64, 32, 2)
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.if_end_relu = featureExtractor_if_end_relu
        self.if_end_norm = featureExtractor_if_end_norm
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs, if_end_relu=self.if_end_relu, if_end_norm=self.if_end_norm)
        # self.warping_layer = WarpingLayer()
        self.warping_layer = WarpingLayer_no_div()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.flow_estimators = FlowEstimatorDense_v2(self.num_ch_in, f_channels=self.estimator_f_channels)
        self.context_networks = ContextNetwork_v2_(self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels)
        self.if_concat_multi_scale_feature = if_concat_multi_scale_feature
        if if_concat_multi_scale_feature:
            self.conv_1x1_cmsf = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(128 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(96 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(64 + 32, 32, kernel_size=1, stride=1, dilation=1),
                                                conv(32 + 32, 32, kernel_size=1, stride=1, dilation=1)])
        else:
            self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                           conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        # flow upsample module
        # flow upsample module
        class _Upsample_flow(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow, self).__init__()
                ch_in = 32
                k = ch_in
                ch_out = 64
                self.conv1 = conv(ch_in, ch_out)
                k += ch_out

                ch_out = 64
                self.conv2 = conv(k, ch_out)
                k += ch_out

                ch_out = 32
                self.conv3 = conv(k, ch_out)
                k += ch_out

                ch_out = 16
                self.conv4 = conv(k, ch_out)
                k += ch_out

                # ch_out = 64
                # self.conv5 = conv(k, ch_out)
                # k += ch_out
                self.conv_last = conv(k, 2, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                # x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x4)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.1f G, params: %.1f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.1f G, params: %.1f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v2(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v2, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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

                        ind += 1
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                _, x_out = self.dense_estimator(x)
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo(cls):
                from thop import profile
                a = _Upsample_flow_v2()
                feature = np.zeros((1, 32, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_,), verbose=False)
                print('PWCNet_unsup_irr_bi_appflow_v8: flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                320 : flops: 15.5 G, params: 0.2 M
                160 : flops: 3.9 G, params: 0.2 M
                80 : flops: 1.0 G, params: 0.2 M
                40 : flops: 0.2 G, params: 0.2 M
                20 : flops: 0.1 G, params: 0.2 M
                10 : flops: 0.0 G, params: 0.2 M
                5 : flops: 0.0 G, params: 0.2 M
                '''
                a = _Upsample_flow_v2()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flops, params = profile(a, inputs=(feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flops, params = profile(a, inputs=(feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v3(tools.abstract_model):
            def __init__(self):
                super(_Upsample_flow_v3, self).__init__()

                class FlowEstimatorDense_temp(tools.abstract_model):

                    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32)):
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
                        self.conv_last = conv(N, 2, isReLU=False)

                    def forward(self, x):
                        x1 = torch.cat([self.conv1(x), x], dim=1)
                        x2 = torch.cat([self.conv2(x1), x1], dim=1)
                        x3 = torch.cat([self.conv3(x2), x2], dim=1)
                        x4 = torch.cat([self.conv4(x3), x3], dim=1)
                        x5 = torch.cat([self.conv5(x4), x4], dim=1)
                        x_out = self.conv_last(x5)
                        return x5, x_out

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.dense_estimator = FlowEstimatorDense_temp(32, (64, 64, 64, 32, 16))
                self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 64, 64, 64, 32, 32, 16, 2))
                # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                feature, x_out = self.dense_estimator(x)
                flow = flow_pre + x_out
                flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                x_out = flow + flow_fine_f
                if if_output_level:
                    x_out = upsample2d_flow_as(x_out, x_raw, mode="bilinear", if_rate=True)
                return x_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v3()
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 2, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 2, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v4(tools.abstract_model):
            def __init__(self, if_mask, if_small=False):
                super(_Upsample_flow_v4, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_small = if_small
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(32, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                    self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                              conv(16, 16, stride=2),
                                                              conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                              conv(32, 32, stride=2), )

            def forward(self, flow_pre, x_raw, if_output_level=False):
                if if_output_level:
                    x = self.upsample_output_conv(x_raw)
                else:
                    x = x_raw
                if self.if_mask:
                    feature, x_out = self.dense_estimator_mask(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out[:, :2, :, :]
                    mask_out = x_out[:, 2, :, :]
                    mask_out = torch.unsqueeze(mask_out, 1)
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                        mask_out = upsample2d_flow_as(mask_out, x_raw, mode="bilinear")
                    mask_out = torch.sigmoid(mask_out)
                    return x_out, flow_out, mask_out
                else:
                    feature, x_out = self.dense_estimator(x)
                    flow = flow_pre + x_out
                    flow_fine_f = self.context_estimator(torch.cat([feature, flow], dim=1))
                    x_out = flow + flow_fine_f
                    flow_out = x_out
                    if if_output_level:
                        flow_out = upsample2d_flow_as(flow_out, x_raw, mode="bilinear", if_rate=True)
                    return flow_out

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        class _Upsample_flow_v5(tools.abstract_model):
            def __init__(self, if_mask, if_small=False, if_cost_volume=False, if_norm_before_cost_volume=True, if_mask_inpainting=False):
                super(_Upsample_flow_v5, self).__init__()

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

                class ContextNetwork_temp(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp, self).__init__()
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

                class ContextNetwork_temp_2(nn.Module):

                    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
                        super(ContextNetwork_temp_2, self).__init__()

                        self.convs = nn.Sequential(
                            conv(num_ls[0], num_ls[1], 3, 1, 1),
                            conv(num_ls[1], num_ls[2], 3, 1, 2),
                            conv(num_ls[2], num_ls[3], 3, 1, 4),
                            conv(num_ls[3], num_ls[4], 3, 1, 8),
                            conv(num_ls[4], num_ls[5], 3, 1, 16),
                            conv(num_ls[5], num_ls[6], 3, 1, 1),
                            conv(num_ls[6], num_ls[7], isReLU=False)
                        )

                    def forward(self, x):
                        return self.convs(x)

                self.if_mask = if_mask
                self.if_mask_inpainting = if_mask_inpainting
                self.if_small = if_small
                self.if_cost_volume = if_cost_volume
                self.if_norm_before_cost_volume = if_norm_before_cost_volume
                self.warping_layer = WarpingLayer_no_div()
                if self.if_small:
                    f_channels_es = (32, 32, 32, 16, 8)
                    f_channels_ct = (32, 32, 32, 16, 16, 8)
                else:
                    f_channels_es = (64, 64, 64, 32, 16)
                    f_channels_ct = (64, 64, 64, 32, 32, 16)
                if self.if_cost_volume:
                    in_C = 81
                else:
                    in_C = 64
                if if_mask:
                    self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
                    num_ls = (self.dense_estimator_mask.num_feature_channel + 3,) + f_channels_ct + (3,)
                    self.context_estimator_mask = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))
                else:
                    self.dense_estimator = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=2)
                    num_ls = (self.dense_estimator.num_feature_channel + 2,) + f_channels_ct + (2,)
                    self.context_estimator = ContextNetwork_temp_2(num_ls=num_ls)
                    # self.dense_estimator = FlowEstimatorDense_temp(32, (128, 128, 96, 64, 32))
                    # self.context_estimator = ContextNetwork_temp_2(num_ls=(self.dense_estimator.num_feature_channel + 2, 128, 128, 128, 96, 64, 32, 2))

                self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                          conv(16, 16, stride=2),
                                                          conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                          conv(32, 32, stride=2), )

            def forward(self, flow, feature_1, feature_2, if_save_running_process=None, output_level_flow=None, save_running_process_dir=''):
                feature_2_warp = self.warping_layer(feature_2, flow)
                # print('v5 upsample')
                if self.if_cost_volume:
                    # if norm feature
                    if self.if_norm_before_cost_volume:
                        feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True,
                                                                                     moments_across_channels=False,
                                                                                     moments_across_images=False)
                    # correlation
                    input_feature = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
                    # tools.check_tensor(input_feature, 'input_feature')
                else:
                    input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
                    # tools.check_tensor(input_feature, 'input_feature')
                if self.if_mask:
                    # print('v5 upsample if_mask %s' % self.if_mask)
                    feature, x_out = self.dense_estimator_mask(input_feature)
                    flow_fine_f = self.context_estimator_mask(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out[:, :2, :, :]
                    meta_mask = x_out[:, 2, :, :]
                    meta_mask = torch.unsqueeze(meta_mask, 1)
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        meta_mask = upsample2d_flow_as(meta_mask, output_level_flow, mode="bilinear")
                        flow = output_level_flow
                    meta_mask = torch.sigmoid(meta_mask)
                    if self.if_mask_inpainting:
                        # flow_up = tools.torch_warp(meta_mask * flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                        flow_up = tools.torch_warp(meta_mask * flow, meta_flow * (1 - meta_mask))  # + flow * meta_mask
                    else:
                        flow_up = tools.torch_warp(flow, meta_flow) * (1 - meta_mask) + flow * meta_mask
                    # print('v5 upsample if_mask %s  save_flow' % self.if_mask)
                    # self.save_flow(flow_up, '%s_flow_upbyflow' % if_save_running_process)
                    if if_save_running_process is not None:
                        # print('save results', if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_mask(save_running_process_dir, meta_mask, '%s_meta_mask' % if_save_running_process)
                else:
                    feature, x_out = self.dense_estimator(input_feature)
                    flow_fine_f = self.context_estimator(torch.cat([feature, x_out], dim=1))
                    x_out = x_out + flow_fine_f
                    meta_flow = x_out
                    if output_level_flow is not None:
                        meta_flow = upsample2d_flow_as(meta_flow, output_level_flow, mode="bilinear", if_rate=True)
                        flow = output_level_flow
                    flow_up = tools.torch_warp(flow, meta_flow)
                    if if_save_running_process is not None:
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, flow_up, '%s_flow_upbyflow' % if_save_running_process)
                        PWCNet_unsup_irr_bi_v5_2.save_flow(save_running_process_dir, meta_flow, '%s_meta_flow' % if_save_running_process)
                return flow_up

            def output_feature(self, x):
                x = self.upsample_output_conv(x)
                return x

            @classmethod
            def demo_mscale(cls):
                from thop import profile
                '''
                    320 : flops: 55.018 G, params: 0.537 M
                    160 : flops: 13.754 G, params: 0.537 M
                    80 : flops: 3.439 G, params: 0.537 M
                    40 : flops: 0.860 G, params: 0.537 M
                    20 : flops: 0.215 G, params: 0.537 M
                    10 : flops: 0.054 G, params: 0.537 M
                    5 : flops: 0.013 G, params: 0.537 M
                    output level : flops: 3.725 G, params: 0.553 M
                '''
                a = _Upsample_flow_v4(if_mask=True)
                for i in [320, 160, 80, 40, 20, 10, 5]:
                    feature = np.zeros((1, 32, i, i))
                    flow_pre = np.zeros((1, 3, i, i))
                    feature_ = torch.from_numpy(feature).float()
                    flow_pre_ = torch.from_numpy(flow_pre).float()
                    flops, params = profile(a, inputs=(flow_pre_, feature_,), verbose=False)
                    print('%s : flops: %.3f G, params: %.3f M' % (i, flops / 1000 / 1000 / 1000, params / 1000 / 1000))
                feature = np.zeros((1, 3, 320, 320))
                feature_ = torch.from_numpy(feature).float()
                flow_pre = np.zeros((1, 3, 80, 80))
                flow_pre_ = torch.from_numpy(flow_pre).float()
                flops, params = profile(a, inputs=(flow_pre_, feature_, True), verbose=False)
                print('%s : flops: %.3f G, params: %.3f M' % ('output level', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

        self.if_upsample_flow = if_upsample_flow
        self.if_upsample_flow_output = if_upsample_flow_output
        self.if_upsample_flow_mask = if_upsample_flow_mask
        self.if_upsample_small = if_upsample_small
        self.if_upsample_cost_volume = if_upsample_cost_volume
        self.if_upsample_mask_inpainting = if_upsample_mask_inpainting
        self.if_dense_decode = if_dense_decode
        if self.if_upsample_flow or self.if_upsample_flow_output:
            self.upsample_model_v5 = _Upsample_flow_v5(if_mask=self.if_upsample_flow_mask, if_small=self.if_upsample_small, if_cost_volume=self.if_upsample_cost_volume,
                                                       if_norm_before_cost_volume=self.if_norm_before_cost_volume, if_mask_inpainting=self.if_upsample_mask_inpainting)
        else:
            self.upsample_model_v5 = None
            self.upsample_output_conv = None

        # app flow module
        self.app_occ_stop_gradient = app_occ_stop_gradient  # stop gradient of the occ mask when inpaint
        self.app_distilation_weight = app_distilation_weight  # fangqi, i will do not use this part
        if app_loss_weight > 0:
            self.appflow_model = Appearance_flow_net_for_disdiilation.App_model(input_channel=7, if_share_decoder=False)
        self.app_loss_weight = app_loss_weight

        self.app_v2_if_app = app_v2_if_app,  # if use app flow
        self.app_v2_if_app_small_level = app_v2_if_app_small_level
        self.app_v2_iter_num = app_v2_iter_num
        if self.app_v2_if_app:
            self.app_v2_flow_model = Appearance_flow_net_for_disdiilation.App_model_small(input_channel=39, if_share_decoder=True, small_level=self.app_v2_if_app_small_level)
        self.app_v2_if_app_level = app_v2_if_app_level  # if use app flow in each level, (6 ge True/Flase)
        self.app_v2_if_app_level_alpha = app_v2_if_app_level_alpha
        # build occ models
        self.occ_check_model_ls = []
        for i in range(len(self.app_v2_if_app_level_alpha)):
            occ_alpha_1 = self.app_v2_if_app_level_alpha[i][0]
            occ_alpha_2 = self.app_v2_if_app_level_alpha[i][1]
            model = tools.occ_check_model(occ_type=occ_type, occ_alpha_1=occ_alpha_1, occ_alpha_2=occ_alpha_2, obj_out_all=occ_check_obj_out_all)
            self.occ_check_model_ls.append(model)
        self.app_v2_app_loss_weight = app_v2_app_loss_weight  # app loss weight
        self.app_v2_app_loss_type = app_v2_app_loss_type  # abs_robust, charbonnier,L1, SSIM

        class _WarpingLayer(tools.abstract_model):

            def __init__(self):
                super(_WarpingLayer, self).__init__()

            def forward(self, x, flo):
                """
                warp an image/tensor (im2) back to im1, according to the optical flow

                x: [B, C, H, W] (im2)
                flo: [B, 2, H, W] flow

                """

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
                vgrid = grid + flo

                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
                output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
                if x.is_cuda:
                    mask = torch.ones(x.size(), requires_grad=False).cuda()
                else:
                    mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
                mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                mask = (mask >= 1.0).float()
                # mask = torch.autograd.Variable(torch.ones(x.size()))
                # if x.is_cuda:
                #     mask = mask.cuda()
                # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
                #
                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
                output = output * mask
                # # nchw->>>nhwc
                # if x.is_cuda:
                #     output = output.cpu()
                # output_im = output.numpy()
                # output_im = np.transpose(output_im, (0, 2, 3, 1))
                # output_im = np.squeeze(output_im)
                return output

        self.warping_layer_inpaint = _WarpingLayer()

        initialize_msra(self.modules())
        self.if_froze_pwc = if_froze_pwc
        if self.if_froze_pwc:
            self.froze_PWC()

    def froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    @classmethod
    def save_image(cls, save_running_process_dir, image_tensor, name='image'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np)[:, :, ::-1])

    @classmethod
    def save_flow(cls, save_running_process_dir, flow_tensor, name='flow'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        # print(self.save_running_process_dir, 'save flow %s' % name)
        flow_tensor_np = tensor_to_np_for_save(flow_tensor)
        save_path = os.path.join(save_running_process_dir, name + '.png')
        # save_path = os.path.join(self.save_running_process_dir, name + '.png')
        # print(type(flow_tensor_np), flow_tensor_np.shape)
        # print(save_path)
        # cv2.imwrite(save_path, tools.Show_GIF.im_norm(tools.flow_to_image(flow_tensor_np)[:, :, ::-1]))
        cv2.imwrite(save_path, tools.flow_to_image(flow_tensor_np)[:, :, ::-1])

    @classmethod
    def save_mask(cls, save_running_process_dir, image_tensor, name='mask'):
        def tensor_to_np_for_save(a):
            # gt_flow_np = tensor_to_np_for_save(gt_flow)
            # cv2.imwrite(os.path.join(save_dir_dir, 'gt_flow_np' + '.png'), tools.Show_GIF.im_norm(tools.flow_to_image(gt_flow_np)[:, :, ::-1]))
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        image_tensor_np = tensor_to_np_for_save(image_tensor)
        cv2.imwrite(os.path.join(save_running_process_dir, name + '.png'), tools.Show_GIF.im_norm(image_tensor_np))

    def decode_level(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_up2d' % level)
                self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_up2d' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
        out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
        out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear = flow_1_up_bilinear + flow_fine_1
        flow_2_up_bilinear = flow_2_up_bilinear + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear

    def decode_level_res(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            # if self.if_save_running_process:
            #     self.save_flow(self.save_running_process_dir, flow_1_up_bilinear, '%s_flow_f_linear_up' % level)
            #     self.save_flow(self.save_running_process_dir, flow_2_up_bilinear, '%s_flow_b_linear_up' % level)
            if self.if_upsample_flow:
                flow_1_up_bilinear = self.upsample_model_v5(flow=flow_1_up_bilinear, feature_1=feature_1_1x1, feature_2=feature_2_1x1,
                                                            if_save_running_process='%s_fw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
                flow_2_up_bilinear = self.upsample_model_v5(flow=flow_2_up_bilinear, feature_1=feature_2_1x1, feature_2=feature_1_1x1,
                                                            if_save_running_process='%s_bw' % level if self.if_save_running_process else None, save_running_process_dir=self.save_running_process_dir)
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = network_tools.normalize_features((feature_1, feature_2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
            feature_2, feature_1_warp = network_tools.normalize_features((feature_2, feature_1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                                         moments_across_images=self.norm_moments_across_images)
        # correlation
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

    def forward_2_frame_v2(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        level_iter_ls = (1, 1, 1, 1, 1)
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            level_iter = level_iter_ls[level]
            for _ in range(level_iter):
                flow_f, flow_b = self.decode_level(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2, feature_2_1x1=x2_1by1)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        return flow_f_out, flow_b_out, flows[::-1]

    def forward_2_frame_v3_dense(self, x1_raw, x2_raw, if_loss=False):
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
        x1_m = None
        x2_m = None
        # build pyramid
        feature_level_ls = []
        app_loss = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self.decode_level_res(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                           feature_2_1x1=x2_1by1)
            # do not use dense_decoder, no use
            if level != 0 and self.if_dense_decode:
                for i_, (temp_f, temp_b) in enumerate(flows[:-1]):
                    _, _, temp_f_res, temp_b_res = self.decode_level_res(level='%s.%s' % (level, i_), flow_1=temp_f, flow_2=temp_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                         feature_2_1x1=x2_1by1)
                    flow_f_res = temp_f_res + flow_f_res
                    flow_b_res = temp_b_res + flow_b_res
            # tools.check_tensor(flow_f_res, 'flow_f_res')
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            # app refine
            # print(level, 'level ')
            if self.app_v2_if_app and self.app_v2_if_app_level[level] > 0:
                # occ_1, occ_2 = self.occ_check_model_ls[level](flow_f=flow_f, flow_b=flow_b)
                # occ_1, occ_2 = self.occ_check_model(flow_f=flow_f, flow_b=flow_b)  # 0 in occ area, 1 in others
                # occ_1, occ_2 = self.occ_check_model.forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, alpha1=self.app_v2_if_app_level_alpha[level][0],
                #                                                                alpha2=self.app_v2_if_app_level_alpha[level][1])
                occ_1, occ_2 = self.occ_check_model_ls[level](flow_f=flow_f, flow_b=flow_b)
                flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f, occ_1, x1_1by1, refine_level=level)
                flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b, occ_2, x2_1by1, refine_level=level)
                if self.app_v2_iter_num > 1:
                    for ind in range(self.app_v2_iter_num - 1):
                        occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_restore, flow_b=flow_b_restore)
                        flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_restore, occ_1, x1_1by1, refine_level=-1)
                        flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_restore, occ_2, x2_1by1, refine_level=-1)
                if self.if_save_running_process:
                    im_1_restore = tools.torch_warp(im_1_masked, app_f_flow)
                    im_2_restore = tools.torch_warp(im_2_masked, app_b_flow)
                    self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_decode' % level)
                    self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_decode' % level)
                    self.save_flow(self.save_running_process_dir, flow_f_restore, '%s_flow_f_appref' % level)
                    self.save_flow(self.save_running_process_dir, flow_b_restore, '%s_flow_b_appref' % level)
                    self.save_flow(self.save_running_process_dir, app_f_flow, '%s_flow_f_appflow' % level)
                    self.save_flow(self.save_running_process_dir, app_b_flow, '%s_flow_b_appflow' % level)
                    self.save_mask(self.save_running_process_dir, occ_1, '%s_occ_1' % level)
                    self.save_mask(self.save_running_process_dir, occ_2, '%s_occ_2' % level)
                    self.save_image(self.save_running_process_dir, im_1_restore, '%s_im_1_restore' % level)
                    self.save_image(self.save_running_process_dir, im_1_masked, '%s_im_1_masked' % level)
                    self.save_image(self.save_running_process_dir, im_2_restore, '%s_im_2_restore' % level)
                    self.save_image(self.save_running_process_dir, im_2_masked, '%s_im_2_masked' % level)
                if if_loss:
                    temp_app_loss = self.app_v2_loss(im_1_resize, app_f_flow, occ_1) + self.app_v2_loss(im_2_resize, app_b_flow, occ_2)
                    if app_loss is None:
                        app_loss = temp_app_loss * self.app_v2_if_app_level[level]
                    else:
                        app_loss += temp_app_loss * self.app_v2_if_app_level[level]

                flow_f = flow_f_restore
                flow_b = flow_b_restore
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_f, '%s_scale_output_flow_f' % level)
                self.save_flow(self.save_running_process_dir, flow_b, '%s_scale_output_flow_b' % level)
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_linear_up')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_linear_up')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            feature_1_1x1 = self.upsample_model_v5.output_feature(x1_raw)
            feature_2_1x1 = self.upsample_model_v5.output_feature(x2_raw)
            flow_f_out = self.upsample_model_v5(flow=flow_f, feature_1=feature_1_1x1, feature_2=feature_2_1x1, if_save_running_process='%s_fw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_f_out, save_running_process_dir=self.save_running_process_dir)
            flow_b_out = self.upsample_model_v5(flow=flow_b, feature_1=feature_2_1x1, feature_2=feature_1_1x1, if_save_running_process='%s_bw' % 'out' if self.if_save_running_process else None,
                                                output_level_flow=flow_b_out, save_running_process_dir=self.save_running_process_dir)
        if self.app_v2_if_app and self.app_v2_if_app_level[-1] > 0:
            app_feature_1_1x1 = self.app_v2_flow_model.output_feature(x1_raw)
            app_feature_2_1x1 = self.app_v2_flow_model.output_feature(x2_raw)
            # occ_1, occ_2 = self.occ_check_model(flow_f=flow_f_out, flow_b=flow_b_out)  # 0 in occ area, 1 in others
            # occ_1, occ_2 = self.occ_check_model.forward_backward_occ_check(flow_fw=flow_f_out, flow_bw=flow_b_out, alpha1=self.app_v2_if_app_level_alpha[-1][0],
            #                                                                alpha2=self.app_v2_if_app_level_alpha[-1][1])
            occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_out, flow_b=flow_b_out)
            flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_out, occ_1, app_feature_1_1x1, refine_level=-1)
            # tools.check_tensor(app_f_flow,'app_f_flow  forward')
            flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_out, occ_2, app_feature_2_1x1, refine_level=-1)
            if self.app_v2_iter_num > 1:
                for ind in range(self.app_v2_iter_num - 1):
                    occ_1, occ_2 = self.occ_check_model_ls[-1](flow_f=flow_f_restore, flow_b=flow_b_restore)
                    flow_f_restore, app_f_flow, im_1_masked, im_1_resize = self.app_v2_refine(x1_raw, flow_f_restore, occ_1, app_feature_1_1x1, refine_level=-1)
                    flow_b_restore, app_b_flow, im_2_masked, im_2_resize = self.app_v2_refine(x2_raw, flow_b_restore, occ_2, app_feature_2_1x1, refine_level=-1)
            flow_f_out = flow_f_restore
            flow_b_out = flow_b_restore
            im_1_restore = tools.torch_warp(im_1_masked, app_f_flow)
            im_2_restore = tools.torch_warp(im_2_masked, app_b_flow)
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, flow_f_restore, '%s_flow_f_appref' % 'out')
                self.save_flow(self.save_running_process_dir, flow_b_restore, '%s_flow_b_appref' % 'out')
                self.save_flow(self.save_running_process_dir, app_f_flow, '%s_flow_f_appflow' % 'out')
                self.save_mask(self.save_running_process_dir, occ_1, '%s_occ_1' % 'out')
                self.save_mask(self.save_running_process_dir, occ_2, '%s_occ_2' % 'out')
                self.save_image(self.save_running_process_dir, im_1_restore, '%s_im_1_restore' % 'out')
                self.save_image(self.save_running_process_dir, im_1_masked, '%s_im_1_masked' % 'out')
                self.save_image(self.save_running_process_dir, im_2_restore, '%s_im_2_restore' % 'out')
                self.save_image(self.save_running_process_dir, im_2_masked, '%s_im_2_masked' % 'out')
            if if_loss:
                temp_app_loss = self.app_v2_loss(im_1_resize, app_f_flow, occ_1) + self.app_v2_loss(im_2_resize, app_b_flow, occ_2)
                if app_loss is None:
                    app_loss = temp_app_loss * self.app_v2_if_app_level[-1]
                else:
                    app_loss += temp_app_loss * self.app_v2_if_app_level[-1]
            return flow_f_out, flow_b_out, flows[::-1], app_loss, app_f_flow, im_1_masked, im_1_restore
        else:
            return flow_f_out, flow_b_out, flows[::-1], app_loss, None, None, None

    def forward_2_frame(self, x1_raw, x2_raw):
        # x1_raw = input_dict['input1']
        # x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # outputs
        # output_dict = {}
        flows = []
        flows_v2 = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        up_flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_f_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        up_flow_b_mask = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        x1_m = None
        x2_m = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # concat and estimate flow
            if self.if_concat_multi_scale_feature:
                if x1_m is None or x2_m is None:
                    x1_1by1 = self.conv_1x1_cmsf[l](x1)
                    x2_1by1 = self.conv_1x1_cmsf[l](x2)
                    x1_m = x1_1by1
                    x2_m = x2_1by1
                else:
                    x1_m = upsample2d_as(x1_m, x1, mode="bilinear")
                    x2_m = upsample2d_as(x2_m, x1, mode="bilinear")
                    x1_1by1 = self.conv_1x1_cmsf[l](torch.cat((x1, x1_m), dim=1))
                    x2_1by1 = self.conv_1x1_cmsf[l](torch.cat((x2, x2_m), dim=1))
                    x1_m = x1_1by1
                    x2_m = x2_1by1
            else:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                # flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                # flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                # x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                # x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_f = upsample2d_flow_as(flow_f, x1, mode="bilinear", if_rate=True)
                flow_b = upsample2d_flow_as(flow_b, x1, mode="bilinear", if_rate=True)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_up2d' % l)
                    self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_up2d' % l)
                if self.if_upsample_flow or self.if_upsample_flow_output:
                    up_flow_f = upsample2d_flow_as(up_flow_f, x1, mode="bilinear", if_rate=True)
                    up_flow_b = upsample2d_flow_as(up_flow_b, x1, mode="bilinear", if_rate=True)
                    if self.if_upsample_flow_mask:
                        up_flow_f_mask = upsample2d_flow_as(up_flow_f_mask, x1, mode="bilinear")
                        up_flow_b_mask = upsample2d_flow_as(up_flow_b_mask, x1, mode="bilinear")
                        _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_1by1)
                        _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = flow_f * up_flow_f_mask + self.warping_layer(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            # flow_b = flow_b * up_flow_b_mask + self.warping_layer(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            flow_f = flow_f * up_flow_f_mask + tools.torch_warp(flow_f, up_flow_f) * (1 - up_flow_f_mask)
                            flow_b = flow_b * up_flow_b_mask + tools.torch_warp(flow_b, up_flow_b) * (1 - up_flow_b_mask)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % l)
                                self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % l)
                    else:
                        up_flow_f = self.upsample_model(up_flow_f, x1_1by1)
                        up_flow_b = self.upsample_model(up_flow_b, x2_1by1)
                        if self.if_upsample_flow:
                            # flow_f = self.warping_layer(flow_f, up_flow_f)
                            # flow_b = self.warping_layer(flow_b, up_flow_b)
                            flow_f = tools.torch_warp(flow_f, up_flow_f)
                            flow_b = tools.torch_warp(flow_b, up_flow_b)
                            if self.if_save_running_process:
                                self.save_flow(self.save_running_process_dir, flow_f, '%s_flow_f_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, flow_b, '%s_flow_b_upbyflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % l)
                                self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % l)
                x2_warp = self.warping_layer(x2, flow_f)
                x1_warp = self.warping_layer(x1, flow_b)
            # if norm feature
            if self.if_norm_before_cost_volume:
                x1, x2_warp = network_tools.normalize_features((x1, x2_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)
                x2, x1_warp = network_tools.normalize_features((x2, x1_warp), normalize=True, center=True, moments_across_channels=self.norm_moments_across_channels,
                                                               moments_across_images=self.norm_moments_across_images)

            # correlation
            out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b
            flow_fine_f = self.context_networks(torch.cat([x_intm_f, flow_f], dim=1))
            flow_fine_b = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
            flow_f = flow_f + flow_fine_f
            flow_b = flow_b + flow_fine_b
            flows.append([flow_f, flow_b])
            if l == self.output_level:
                break
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)
        if self.if_save_running_process:
            self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_up2d')
            self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_up2d')
            self.save_image(self.save_running_process_dir, x1_raw, 'image1')
            self.save_image(self.save_running_process_dir, x2_raw, 'image2')
        if self.if_upsample_flow_output:
            if self.if_upsample_flow_mask:
                _, up_flow_f, up_flow_f_mask = self.upsample_model(torch.cat((up_flow_f, up_flow_f_mask), dim=1), x1_raw, if_output_level=True)
                _, up_flow_b, up_flow_b_mask = self.upsample_model(torch.cat((up_flow_b, up_flow_b_mask), dim=1), x2_raw, if_output_level=True)
                # flow_f_out = flow_f_out * up_flow_f_mask + self.warping_layer(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                # flow_b_out = flow_b_out * up_flow_b_mask + self.warping_layer(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                flow_f_out = flow_f_out * up_flow_f_mask + tools.torch_warp(flow_f_out, up_flow_f) * (1 - up_flow_f_mask)
                flow_b_out = flow_b_out * up_flow_b_mask + tools.torch_warp(flow_b_out, up_flow_b) * (1 - up_flow_b_mask)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_f_mask, '%s_flow_f_upmask' % 'out')
                    self.save_mask(self.save_running_process_dir, up_flow_b_mask, '%s_flow_b_upmask' % 'out')
            else:
                up_flow_f = self.upsample_model(up_flow_f, x1_raw, if_output_level=True)
                up_flow_b = self.upsample_model(up_flow_b, x2_raw, if_output_level=True)
                # flow_f_out = self.warping_layer(flow_f_out, up_flow_f)
                # flow_b_out = self.warping_layer(flow_b_out, up_flow_b)
                flow_f_out = tools.torch_warp(flow_f_out, up_flow_f)
                flow_b_out = tools.torch_warp(flow_b_out, up_flow_b)
                if self.if_save_running_process:
                    self.save_flow(self.save_running_process_dir, flow_f_out, 'out_flow_f_upbyflow')
                    self.save_flow(self.save_running_process_dir, flow_b_out, 'out_flow_b_upbyflow')
                    self.save_flow(self.save_running_process_dir, up_flow_f, '%s_flow_f_upflow' % 'out')
                    self.save_flow(self.save_running_process_dir, up_flow_b, '%s_flow_b_upflow' % 'out')
        return flow_f_out, flow_b_out, flows[::-1]

    def app_v2_refine(self, img_raw, flow, mask, feature_1x1, refine_level=0):
        if img_raw.shape[-2:] != flow.shape[-2:]:
            # _, _, h_raw, w_raw = im1_s.size()
            img1_resize = upsample2d_as(img_raw, flow, mode="bilinear")
        else:
            img1_resize = img_raw
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img1_resize * mask
        # app_flow = self.app_v2_flow_model(torch.cat((feature_1x1, input_im, img1_resize, mask), dim=1), refine_level=refine_level)
        app_flow = self.app_v2_flow_model(torch.cat((input_im, img1_resize, feature_1x1, mask), dim=1), refine_level=refine_level)
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = self.warping_layer_inpaint(flow, app_flow)
        flow_restore = tools.torch_warp(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        # img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img1_resize

    def app_v2_loss(self, img_ori, app_flow, occ_mask):
        img_label = img_ori.clone().detach()

        mask_im = img_label * occ_mask
        img_restore = tools.torch_warp(mask_im, app_flow)
        # diff = img_label - img_restore
        # loss_mask = 1 - occ_mask
        # diff = img_ori - img_restore
        # loss_mask = 1 - occ_mask  # only take care about the inpainting area

        # loss_mean = network_tools.photo_loss_multi_type(img_ori, img_restore, loss_mask, photo_loss_type=self.app_v2_app_loss_type,
        #                                                 photo_loss_delta=0.4, photo_loss_use_occ=True)
        # loss_mean = network_tools.compute_inpaint_photo_loss_mask(img_raw=img_label, img_restore=img_restore,
        #                                                           mask=occ_mask, if_l1=False)
        loss_mean = network_tools.compute_inpaint_photo_loss_mask_multi_type(img_raw=img_label, img_restore=img_restore, mask=occ_mask,
                                                                             photo_loss_type=self.app_v2_app_loss_type,
                                                                             )
        #
        # diff = (torch.abs(diff) + 0.01).pow(0.4)
        # diff = diff * loss_mask
        # diff_sum = torch.sum(diff)
        # loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def app_refine(self, img, flow, mask):
        # occlusion mask: 0-1, where occlusion area is 0
        input_im = img * mask
        app_flow = self.appflow_model(torch.cat((input_im, img, mask), dim=1))
        # app_flow = upsample2d_as(app_flow, input_im, mode="bilinear") * (1.0 / self._div_flow)
        app_flow = app_flow * (1 - mask)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        flow_restore = self.warping_layer_inpaint(flow, app_flow)
        # img_restore = self.warping_layer_inpaint(input_im, app_flow)
        # flow_restore = tools.torch_warp(flow, app_flow)
        img_restore = tools.torch_warp(input_im, app_flow)
        return flow_restore, app_flow, input_im, img_restore

    def app_loss(self, img_ori, img_restore, occ_mask):
        diff = img_ori - img_restore
        loss_mask = 1 - occ_mask  # only take care about the inpainting area
        diff = (torch.abs(diff) + 0.01).pow(0.4)
        diff = diff * loss_mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(loss_mask) * 2 + 1e-6)
        return loss_mean

    def forward(self, input_dict: dict):
        '''
        :param input_dict:     im1, im2, im1_raw, im2_raw, start,if_loss
        :return: output_dict:  flows, flow_f_out, flow_b_out, photo_loss
        '''
        im1_ori, im2_ori = input_dict['im1'], input_dict['im2']
        if input_dict['if_loss']:
            sp_im1_ori, sp_im2_ori = input_dict['im1_sp'], input_dict['im2_sp']
            if self.input_or_sp_input >= 1:
                im1, im2 = im1_ori, im2_ori
            elif self.input_or_sp_input > 0:
                if tools.random_flag(threshold_0_1=self.input_or_sp_input):
                    im1, im2 = im1_ori, im2_ori
                else:
                    im1, im2 = sp_im1_ori, sp_im2_ori
            else:
                im1, im2 = sp_im1_ori, sp_im2_ori
        else:
            im1, im2 = im1_ori, im2_ori

        #
        if 'if_test' in input_dict.keys():
            if_test = input_dict['if_test']
        else:
            if_test = False
        # check if save results
        if 'save_running_process' in input_dict.keys():
            self.if_save_running_process = input_dict['save_running_process']
        else:
            self.if_save_running_process = False
        if self.if_save_running_process:
            if 'process_dir' in input_dict.keys():
                self.save_running_process_dir = input_dict['process_dir']
            else:
                self.if_save_running_process = False
                self.save_running_process_dir = None
        # if show some results
        if 'if_show' in input_dict.keys():
            if_show = input_dict['if_show']
        else:
            if_show = False
        output_dict = {}
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame(im1, im2)
        # flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v2(im1, im2)
        flow_f_pwc_out, flow_b_pwc_out, flows, app_loss, app_flow_1, masked_im1, im1_restore = self.forward_2_frame_v3_dense(im1, im2, if_loss=input_dict['if_loss'])
        if if_show:
            output_dict['app_flow_1'] = app_flow_1
            output_dict['masked_im1'] = masked_im1
            output_dict['im1_restore'] = im1_restore
        occ_fw, occ_bw = self.occ_check_model(flow_f=flow_f_pwc_out, flow_b=flow_b_pwc_out)  # 0 in occ area, 1 in others
        if self.app_loss_weight > 0:
            if self.app_occ_stop_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            # tools.check_tensor(occ_fw, '%s' % (torch.sum(occ_fw == 1) / torch.sum(occ_fw)))
            flow_f, app_flow_1, masked_im1, im1_restore = self.app_refine(img=im1, flow=flow_f_pwc_out, mask=occ_fw)
            # tools.check_tensor(app_flow_1, 'app_flow_1')
            flow_b, app_flow_2, masked_im2, im2_restore = self.app_refine(img=im2, flow=flow_b_pwc_out, mask=occ_bw)
            app_loss = self.app_loss(im1, im1_restore, occ_fw)
            app_loss += self.app_loss(im2, im2_restore, occ_bw)
            app_loss *= self.app_loss_weight
            # tools.check_tensor(app_loss, 'app_loss')
            # print(' ')
            if input_dict['if_loss']:
                output_dict['app_loss'] = app_loss
            if self.app_distilation_weight > 0:
                flow_fw_label = flow_f.clone().detach()
                flow_bw_label = flow_b.clone().detach()
                appd_loss = network_tools.photo_loss_multi_type(x=flow_fw_label, y=flow_f_pwc_out, occ_mask=1 - occ_fw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss += network_tools.photo_loss_multi_type(x=flow_bw_label, y=flow_b_pwc_out, occ_mask=1 - occ_bw, photo_loss_type='abs_robust', photo_loss_use_occ=True)
                appd_loss *= self.app_distilation_weight
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = appd_loss
                if if_test:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                else:
                    flow_f_out = flow_f_pwc_out  # use pwc output
                    flow_b_out = flow_b_pwc_out
                    # flow_f_out = flow_f  # use app refine output
                    # flow_b_out = flow_b
            else:
                if input_dict['if_loss']:
                    output_dict['appd_loss'] = None
                flow_f_out = flow_f
                flow_b_out = flow_b
            if if_show:
                output_dict['app_flow_1'] = app_flow_1
                output_dict['masked_im1'] = masked_im1
                output_dict['im1_restore'] = im1_restore
        else:
            if input_dict['if_loss']:
                output_dict['app_loss'] = None
            flow_f_out = flow_f_pwc_out
            flow_b_out = flow_b_pwc_out
            # if if_show:
            #     output_dict['app_flow_1'] = None
            #     output_dict['masked_im1'] = None
            #     output_dict['im1_restore'] = None
        if_reverse_occ = True  # False
        if if_reverse_occ:
            if_do_reverse = False
            if if_do_reverse:
                f_b_w = tools.torch_warp(flow_b_out, flow_f_out)  # 用f 把b光流 warp过来了
                f_f_w = tools.torch_warp(flow_f_out, flow_b_out)  # 用b 把f光流 warp过来了
                f_b_w = -f_b_w  # 方向取反
                f_f_w = -f_f_w
            else:
                f_b_w = tools.torch_warp(flow_f_out, flow_f_out)  # 用f 把b光流 warp过来了
                f_f_w = tools.torch_warp(flow_b_out, flow_b_out)  # 用b 把f光流 warp过来了
            # 按照occlusion mask进行融合
            temp_f = occ_fw * flow_f_out + (1 - occ_fw) * f_b_w
            temp_b = occ_bw * flow_b_out + (1 - occ_bw) * f_f_w
            if self.if_save_running_process:
                self.save_flow(self.save_running_process_dir, f_b_w, 'warpflow_b_by_f')
                self.save_flow(self.save_running_process_dir, f_f_w, 'warpflow_f_by_b')
                self.save_flow(self.save_running_process_dir, temp_f, 'fuse_occ_flow_f')
                self.save_flow(self.save_running_process_dir, temp_b, 'fuse_occ_flow_b')
            flow_f_out = temp_f
            flow_b_out = temp_b
        output_dict['flow_f_out'] = flow_f_out
        output_dict['flow_b_out'] = flow_b_out
        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw
        if self.if_test:
            output_dict['flows'] = flows
        if input_dict['if_loss']:
            # ?? smooth loss
            if self.smooth_level == 'final':
                s_flow_f, s_flow_b = flow_f_out, flow_b_out
                s_im1, s_im2 = im1_ori, im2_ori
            elif self.smooth_level == '1/4':
                s_flow_f, s_flow_b = flows[0]
                _, _, temp_h, temp_w = s_flow_f.size()
                s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
                s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
                # tools.check_tensor(s_im1, 's_im1')  # TODO
            else:
                raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
            smooth_loss = 0
            if self.smooth_order_1_weight > 0:
                if self.smooth_type == 'edge':
                    # if self.learn_direction=='fw':
                    #     pass
                    # elif self.learn_direction=='bw':
                    #     pass
                    # else:
                    #     raise ValueError('wrong')
                    if self.learn_direction == 'fw':
                        smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                    elif self.learn_direction == 'bw':
                        smooth_loss += self.smooth_order_1_weight * network_tools.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
                    else:
                        raise ValueError('wrong')
                elif self.smooth_type == 'delta':
                    if self.learn_direction == 'fw':
                        smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                    elif self.learn_direction == 'bw':
                        smooth_loss += self.smooth_order_1_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
                    else:
                        raise ValueError('wrong')
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)

            # ?? ?? smooth loss
            if self.smooth_order_2_weight > 0:
                if self.smooth_type == 'edge':
                    if self.learn_direction == 'fw':
                        smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                    elif self.learn_direction == 'bw':
                        smooth_loss += self.smooth_order_2_weight * network_tools.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
                    else:
                        raise ValueError('wrong')
                elif self.smooth_type == 'delta':
                    if self.learn_direction == 'fw':
                        smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                    elif self.learn_direction == 'bw':
                        smooth_loss += self.smooth_order_2_weight * network_tools.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
                    else:
                        raise ValueError('wrong')
                else:
                    raise ValueError('wrong smooth_type: %s' % self.smooth_type)
            output_dict['smooth_loss'] = smooth_loss

            # ?? photo loss
            if self.if_use_boundary_warp:
                # im1_warp = tools.nianjin_warp.warp_im(im2, flow_fw, start)  # warped im1 by forward flow and im2
                # im2_warp = tools.nianjin_warp.warp_im(im1, flow_bw, start)
                im1_s, im2_s, start_s = input_dict['im1_raw'], input_dict['im2_raw'], input_dict['start']
                im1_warp = tools.nianjin_warp.warp_im(im2_s, flow_f_out, start_s)  # warped im1 by forward flow and im2
                im2_warp = tools.nianjin_warp.warp_im(im1_s, flow_b_out, start_s)
            else:
                im1_warp = tools.torch_warp(im2_ori, flow_f_out)  # warped im1 by forward flow and im2
                im2_warp = tools.torch_warp(im1_ori, flow_b_out)

            # im_diff_fw = im1 - im1_warp
            # im_diff_bw = im2 - im2_warp
            # photo loss
            if self.stop_occ_gradient:
                occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
            photo_loss = 0
            if self.learn_direction == 'fw':
                photo_loss += network_tools.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.photo_loss_type,
                                                                  photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            elif self.learn_direction == 'bw':
                photo_loss += network_tools.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.photo_loss_type,
                                                                  photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
            else:
                raise ValueError('wrong')
            output_dict['photo_loss'] = photo_loss
            output_dict['im1_warp'] = im1_warp
            output_dict['im2_warp'] = im2_warp

            # ?? census loss
            if self.photo_loss_census_weight > 0:
                if self.learn_direction == 'fw':
                    census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                                                                   charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                elif self.learn_direction == 'bw':
                    census_loss = loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                                                                   charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                else:
                    raise ValueError('wrong')
                # census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.photo_loss_delta,
                #                                                charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True) + \
                #               loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.photo_loss_delta,
                #                                                charbonnier_or_abs_robust=False, if_use_occ=self.photo_loss_use_occ, averge=True)
                census_loss *= self.photo_loss_census_weight
            else:
                census_loss = None
            output_dict['census_loss'] = census_loss

            # ???????msd loss
            if self.multi_scale_distillation_weight > 0:
                flow_fw_label = flow_f_out.clone().detach()
                flow_bw_label = flow_b_out.clone().detach()
                msd_loss_ls = []
                for i, (scale_fw, scale_bw) in enumerate(flows):
                    if self.multi_scale_distillation_style == 'down':
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    elif self.multi_scale_distillation_style == 'upup':
                        flow_fw_label_sacle = flow_fw_label
                        scale_fw = upsample_flow(scale_fw, target_flow=flow_fw_label_sacle)
                        occ_scale_fw = occ_fw
                        flow_bw_label_sacle = flow_bw_label
                        scale_bw = upsample_flow(scale_bw, target_flow=flow_bw_label_sacle)
                        occ_scale_bw = occ_bw
                    elif self.multi_scale_distillation_style == 'updown':
                        scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                        flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)  #
                        occ_scale_fw = F.interpolate(occ_fw, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')  # occ
                        scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                        flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                        occ_scale_bw = F.interpolate(occ_bw, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                    else:
                        raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)
                    msd_loss_scale_fw = network_tools.photo_loss_multi_type(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_fw)
                    msd_loss_scale_bw = network_tools.photo_loss_multi_type(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw, photo_loss_type='abs_robust',
                                                                            photo_loss_use_occ=self.multi_scale_distillation_occ)
                    msd_loss_ls.append(msd_loss_scale_bw)
                msd_loss = sum(msd_loss_ls)
                msd_loss = self.multi_scale_distillation_weight * msd_loss
            else:
                # ???????photo loss? multi_scale_photo_weight
                if self.multi_scale_photo_weight > 0:
                    _, _, h_raw, w_raw = im1_s.size()
                    _, _, h_temp_crop, h_temp_crop = im1_ori.size()
                    msd_loss_ls = []
                    for i, (scale_fw, scale_bw) in enumerate(flows):
                        if self.multi_scale_distillation_style == 'down':  # ??resize???photo loss
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        elif self.multi_scale_distillation_style == 'upup':  # ???flow resize???????photo loss
                            occ_f_resize = occ_fw
                            occ_b_resize = occ_bw
                            scale_fw = upsample_flow(scale_fw, target_flow=im1_ori)
                            scale_bw = upsample_flow(scale_bw, target_flow=im2_ori)
                            im1_crop_resize = im1
                            im2_crop_resize = im2
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_s, scale_fw, start_s)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_s, scale_bw, start_s)
                        elif self.multi_scale_distillation_style == 'updown':
                            scale_fw = upsample_flow(scale_fw, target_size=(scale_fw.size(2) * 4, scale_fw.size(3) * 4))  #
                            scale_bw = upsample_flow(scale_bw, target_size=(scale_bw.size(2) * 4, scale_bw.size(3) * 4))
                            _, _, h_temp, w_temp = scale_fw.size()
                            rate = h_temp_crop / h_temp
                            occ_f_resize, occ_b_resize = self.occ_check_model(flow_f=scale_fw, flow_b=scale_bw)
                            im1_crop_resize = F.interpolate(im1_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im2_crop_resize = F.interpolate(im2_ori, [h_temp, w_temp], mode="bilinear", align_corners=True)
                            im1_raw_resize = F.interpolate(im1_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im2_raw_resize = F.interpolate(im2_s, [int(h_raw / rate), int(w_raw / rate)], mode="bilinear", align_corners=True)
                            im1_resize_warp = tools.nianjin_warp.warp_im(im2_raw_resize, scale_fw, start_s / rate)  # use forward flow to warp im2
                            im2_resize_warp = tools.nianjin_warp.warp_im(im1_raw_resize, scale_bw, start_s / rate)
                        else:
                            raise ValueError('wrong multi_scale_distillation_style: %s' % self.multi_scale_distillation_style)

                        temp_mds_fw = network_tools.photo_loss_multi_type(im1_crop_resize, im1_resize_warp, occ_f_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_fw)
                        temp_mds_bw = network_tools.photo_loss_multi_type(im2_crop_resize, im2_resize_warp, occ_b_resize, photo_loss_type=self.photo_loss_type,
                                                                          photo_loss_delta=self.photo_loss_delta, photo_loss_use_occ=self.photo_loss_use_occ)
                        msd_loss_ls.append(temp_mds_bw)
                    msd_loss = sum(msd_loss_ls)
                    msd_loss = self.multi_scale_photo_weight * msd_loss
                else:
                    msd_loss = None

            output_dict['msd_loss'] = msd_loss

            # appearance flow restore loss
            if app_loss is None:
                pass
            else:
                app_loss = app_loss * self.app_v2_app_loss_weight
                output_dict['app_loss'] = app_loss

        return output_dict

    @classmethod
    def demo(cls):
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=1,
                                       app_distilation_weight=1,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       ).cuda()
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        output_dict = net(input_dict)
        print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])

    @classmethod
    def demo_model_size_app_module(cls):
        from thop import profile
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,

                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_v2_if_app=True,  # if use app flow in each scale
                                       app_v2_if_app_level=(False, True, True, True, True, True),  # if use app flow in each level,(1/64,1/32,1/16,1/8,1/4,output)
                                       app_v2_app_loss_weight=1,  # app loss weight
                                       app_v2_app_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                                       app_v2_if_app_small_level=1,

                                       multi_scale_distillation_weight=0,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_photo_weight=0,

                                       featureExtractor_if_end_relu=True,
                                       featureExtractor_if_end_norm=True
                                       ).cuda()
        # without upsample: flops: 39.893 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 50.525 G, params: 3.996 M

        '''

        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch, 'im1_sp': im_torch, 'im2_sp': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': True}
        # out=net(input_dict)
        flops, params = profile(net, inputs=(input_dict,), verbose=False)
        print('temp(%s): flops: %.3f G, params: %.3f M' % (' ', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def demo_model_size_upsample(cls):
        from thop import profile
        net = PWCNet_unsup_irr_bi_v5_2(occ_type='for_back_check', occ_alpha_1=0.1, occ_alpha_2=0.5, occ_check_sum_abs_or_squar=True,
                                       occ_check_obj_out_all='obj', stop_occ_gradient=False,
                                       smooth_level='final', smooth_type='edge', smooth_order_1_weight=1, smooth_order_2_weight=0,
                                       photo_loss_type='abs_robust', photo_loss_use_occ=False, photo_loss_census_weight=0,
                                       if_norm_before_cost_volume=True, norm_moments_across_channels=False, norm_moments_across_images=False,
                                       multi_scale_distillation_weight=1,
                                       multi_scale_distillation_style='upup',
                                       multi_scale_distillation_occ=True,
                                       # appearance flow params
                                       if_froze_pwc=False,
                                       app_occ_stop_gradient=True,
                                       app_loss_weight=0,
                                       app_distilation_weight=1,
                                       if_upsample_flow=False,
                                       if_upsample_flow_mask=False,
                                       if_upsample_flow_output=False,
                                       if_upsample_small=False,
                                       if_upsample_cost_volume=False,
                                       if_upsample_mask_inpainting=False,
                                       if_dense_decode=True,
                                       if_decoder_small=True,
                                       ).cuda()
        # without upsample: flops: 39.893 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 50.525 G, params: 3.996 M
        |    True   |   False   |   False   |   False   |   True    |   False   |flops: 51.321 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   True    |   False   |flops: 60.498 G, params: 4.043 M
        |    True   |   False   |   True    |   False   |   False   |   False   |flops: 59.103 G, params: 3.996 M
        |    True   |   False   |   True    |   True    |   True    |   False   |flops: 47.208 G, params: 3.597 M
        |    True   |   False   |   True    |   True    |   False   |   False   |flops: 46.506 G, params: 3.573 M
        |    True   |   False   |   False   |   True    |   False   |   False   |flops: 43.339 G, params: 3.573 M
        |    True   |   True    |   True    |   True    |   True    |   False   |flops: 47.273 G, params: 3.599 M
        '''
        # dense decode=True   without upsample:flops: 144.675 G, params: 3.354 M
        '''
        model size when using the upsample module
        | meta flow | meta mask |  meta out |   small   |cost volume| mask inp  |     
        |    True   |   False   |   False   |   False   |   False   |   False   |flops: 183.825 G, params: 3.996 M
        '''
        net.eval()
        im = np.random.random((1, 3, 320, 320))
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().cuda()
        im_torch = torch.from_numpy(im).float().cuda()
        input_dict = {'im1': im_torch, 'im2': im_torch, 'im1_sp': im_torch, 'im2_sp': im_torch,
                      'im1_raw': im_torch, 'im2_raw': im_torch, 'start': start, 'if_loss': False}
        flops, params = profile(net, inputs=(input_dict,), verbose=False)
        print('temp(%s): flops: %.3f G, params: %.3f M' % (' ', flops / 1000 / 1000 / 1000, params / 1000 / 1000))
        # output_dict = net(input_dict)
        # print('smooth_loss', output_dict['smooth_loss'], 'photo_loss', output_dict['photo_loss'], 'census_loss', output_dict['census_loss'], output_dict['app_loss'], output_dict['appd_loss'])


if __name__ == '__main__':
    PWCNet_unsup_irr_bi_v2.demo()
