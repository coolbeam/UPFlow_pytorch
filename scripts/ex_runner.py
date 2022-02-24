# -*- coding: utf-8 -*-
import os
from utils.tools import tools
import cv2
import numpy as np
from copy import deepcopy
import torch
import warnings  # ignore warnings
import torch.nn.functional as F
import torch.optim as optim
from dataset.kitti_dataset import kitti_train, kitti_flow
from model.upflow import UPFlow_net
from torch.utils.data import DataLoader
import time

''' scripts for training：
1. simply using photo loss and smooth loss
2. add occlusion checking
3. add teacher-student loss(ARFlow)
'''

class Trainer_model(tools.abs_test_model):
    class config(tools.abstract_config):  # TODO

        def __init__(self, **kwargs):
            self.lr = 1e-2
            self.weight_decay = 1e-5
            self.optmizer_name = 'adam'
            self.gamma = 0.95
            self.gpu_opt = None  # None is multi GPU
            self.model_path = '/data/luokunming/Optical_Flow_all/training/unsup_PWC_flyc_photo_smooth/unsupPWC_epoch_27_Flyc_epe_error(5.125).pth'  # loading the model
            self.load_relax = False  # load the pretrain model的时候是不是放宽松要求
            self.print_every = 20

            # parameters of spatial transform
            self.if_train_sp = False  # 开关，是否使用spatial transform增强
            self.sptrans_add_noise = True
            self.sptrans_hflip = True
            self.sptrans_rotate = [-0.01, 0.01, -0.01, 0.01]
            self.sptrans_squeeze = [1.0, 1.0, 1.0, 1.0]
            self.sptrans_trans = [0.04, 0.005]
            self.sptrans_vflip = True
            self.sptrans_zoom = [1.0, 1.4, 0.99, 1.01]
            self.spatial_trans_if_mask = True  # 在spatial transform蒸馏的时候使用使用occ mask
            self.spatial_trans_eps = 0.0
            self.spatial_trans_q = 1.0
            self.spatial_trans_loss_weight = 0.01  # 计算在loss里面的权重
            self.sp_input_or_sp_input = 1
            self.train_sp_msd_loss_weight = 0  # sp的时候也算多尺度的损失
            self.train_sp_msd_loss_style = 'down'  # 暂时只有'down'和'up'

            self.final_sp_train_weight = 0  # 用clean的输出来监督一波final, 必须要'sp_input_or_sp_input' <=0才能使用
            self.final_sp_train_style = 'down'  # 暂时没有用，

            self.multi_scale_eval = False  # 验证的时候也对比计算多尺度的结果

            self.train_dir = '/data/luokunming/Optical_Flow_all/training/demo_unsupervised_train'  # 这个参数主函数里面会设置
            self.update(kwargs)

        def __call__(self, net_work: tools.abstract_model):
            # load network
            if self.model_path is not None:
                net_work.load_model(self.model_path, if_relax=self.load_relax)
            return Trainer_model(self, net_work)

    def __init__(self, conf: config, net_work: tools.abstract_model):
        super(Trainer_model, self).__init__(conf=conf, net_work=net_work)
        self.conf = conf
        self.net_work=net_work
        if self.conf.optmizer_name == 'adam':
            self.optimizer = optim.Adam(self.net_work.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.weight_decay)
        else:
            raise ValueError('wrong optmizer name: %s' % self.conf.optmizer_name)
        self.data_clock = tools.Clock_luo()
        self.msd_loss_meter = tools.AverageMeter()
        self.sp_msd_loss_meter = tools.AverageMeter()
        self.final_sp_loss_meter = tools.AverageMeter()
        self.app_loss_meter = tools.AverageMeter()
        self.appd_loss_meter = tools.AverageMeter()
        self.inpaint_img_loss_meter = tools.AverageMeter()
        self.spatial_trans_loss_meter = tools.AverageMeter()
        self.multi_scale_loss_meter = tools.AverageMeter()
        self.census_loss_meter = tools.AverageMeter()
        self.best_name = ''
        self.occ_check_model = tools.occ_check_model(occ_type=self.conf.occ_type, occ_alpha_1=self.conf.alpha_1, occ_alpha_2=self.conf.alpha_2,
                                                     sum_abs_or_squar=self.conf.occ_check_sum_abs_or_squar, obj_out_all=self.conf.occ_check_obj_out_all)
        self.print_str = ''
        self.cnt = 0
        self.temp_save_eval_test = []

        # ===== spatial transform =====
        '''
                    class config():
                def __init__(self):
                    self.add_noise = False
                    self.hflip = False
                    self.rotate = [-0.01, 0.01, -0.01, 0.01]
                    self.squeeze = [1.0, 1.0, 1.0, 1.0]
                    self.trans = [0.04, 0.005]
                    self.vflip = False
                    self.zoom = [1.0, 1.4, 0.99, 1.01]
        '''

        class sp_conf():

            def __init__(self, conf):
                self.add_noise = conf.sptrans_add_noise  # False
                self.hflip = conf.sptrans_hflip  # False
                self.rotate = conf.sptrans_rotate  # [-0.01, 0.01, -0.01, 0.01]
                self.squeeze = conf.sptrans_squeeze  # [1.0, 1.0, 1.0, 1.0]
                self.trans = conf.sptrans_trans  # [0.04, 0.005]
                self.vflip = conf.sptrans_vflip  # False
                self.zoom = conf.sptrans_zoom  # [1.0, 1.4, 0.99, 1.01]

        self.sp_transform = tools.SP_transform.RandomAffineFlow(
            sp_conf(self.conf), addnoise=self.conf.sptrans_add_noise).cuda()  #

    def train_batch(self, batch_step, im1, im2, *args, **kwargs):  # 训练一个batch

        if self.data_clock.start_flag:
            self.data_clock.end()
        if_print = batch_step % self.conf.print_every == 0
        frame_1_ls = []
        frame_2_ls = []
        print_str = '%s %s Epoch%d Iter%d [%.4fs]' % (self.conf.print_name, self.best_name, self.epoch, batch_step, self.data_clock.get_during())
        if_show = batch_step % self.conf.show_every == 0 and self.conf.show_every > 0
        batch_N = im1.shape[0]
        im1_crop_ori, im2_crop_ori, start = args
        sp_img1_ori, sp_img2_ori = kwargs['im1_crop_at'], kwargs['im2_crop_at']  # final的图片
        _, _, h_, w_ = im1_crop_ori.size()
        # self.save_image(im1, 'im1')
        # self.save_image(im2, 'im2')
        # self.save_image(im1_crop_ori, 'im1_crop_ori')
        # self.save_image(im2_crop_ori, 'im2_crop_ori')
        # self.save_image(sp_img1_ori, 'sp_img1_ori')
        # self.save_image(sp_img2_ori, 'sp_img2_ori')
        # while True:
        #     print('return')
        #     time.sleep(1)
        # 决定输入给网络的数据
        im1_crop, im2_crop = im1_crop_ori, im2_crop_ori
        # ============================================================= 网络输出 ===================================================================
        self.optimizer.zero_grad()
        # =========== 计算photo loss和smooth loss以及census loss ===========
        if self.conf.model_name.lower() in ['pwcirrbiv5_v4', ]:
            input_dict = {'im1': im1_crop, 'im2': im2_crop, 'im1_sp': sp_img1_ori, 'im2_sp': sp_img2_ori,
                          'im1_raw': im1, 'im2_raw': im2, 'start': start, 'if_loss': True, 'if_show': if_show}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            occ_fw, occ_bw = output_dict['occ_fw'], output_dict['occ_bw']
            photo_loss, smooth_loss, census_loss = output_dict['photo_loss'].mean(), output_dict['smooth_loss'].mean(), output_dict['census_loss']
            im1_warp = output_dict['im1_warp']
            im2_warp = output_dict['im2_warp']
            loss = photo_loss + smooth_loss
            if census_loss is None:
                pass
            else:
                census_loss = census_loss.mean()
                loss += census_loss
                self.census_loss_meter.update(val=census_loss.item(), num=batch_N)
                print_str += ' cens %.4f(%.4f)' % (self.census_loss_meter.val, self.census_loss_meter.avg)
            if output_dict['msd_loss'] is None:
                pass
            else:
                msd_loss = output_dict['msd_loss'].mean()
                loss += msd_loss
                self.msd_loss_meter.update(val=msd_loss.item(), num=batch_N)
                print_str += ' msd %.4f(%.4f)' % (self.msd_loss_meter.val, self.msd_loss_meter.avg)
            if 'app_loss' not in output_dict.keys():
                pass
            elif output_dict['app_loss'] is None:
                pass
            else:
                app_loss = output_dict['app_loss'].mean()
                loss += app_loss
                self.app_loss_meter.update(val=app_loss.item(), num=batch_N)
                print_str += ' app %.4f(%.4f)' % (self.app_loss_meter.val, self.app_loss_meter.avg)

            self.photo_loss_meter.update(val=photo_loss.item(), num=batch_N)
            self.smooth_loss_meter.update(val=smooth_loss.item(), num=batch_N)
            print_str += ' ph %.4f(%.4f)' % (self.photo_loss_meter.val, self.photo_loss_meter.avg)
            print_str += ' sm %.4f(%.4f)' % (self.smooth_loss_meter.val, self.smooth_loss_meter.avg)
        elif self.conf.model_name.lower() in ['pwcirrbiv5_v5', ]:
            input_dict = {'im1': im1_crop, 'im2': im2_crop, 'im1_sp': sp_img1_ori, 'im2_sp': sp_img2_ori,
                          'im1_raw': im1, 'im2_raw': im2, 'start': start, 'if_loss': True, 'if_show': if_show}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            occ_fw, occ_bw = output_dict['occ_fw'], output_dict['occ_bw']
            photo_loss, smooth_loss, census_loss = output_dict['photo_loss'].mean(), output_dict['smooth_loss'].mean(), output_dict['census_loss']
            im1_warp = output_dict['im1_warp']
            im2_warp = output_dict['im2_warp']
            loss = photo_loss + smooth_loss
            if census_loss is None:
                pass
            else:
                census_loss = census_loss.mean()
                loss += census_loss
                self.census_loss_meter.update(val=census_loss.item(), num=batch_N)
                print_str += ' cens %.4f(%.4f)' % (self.census_loss_meter.val, self.census_loss_meter.avg)
            if output_dict['msd_loss'] is None:
                pass
            else:
                msd_loss = output_dict['msd_loss'].mean()
                loss += msd_loss
                self.msd_loss_meter.update(val=msd_loss.item(), num=batch_N)
                print_str += ' msd %.4f(%.4f)' % (self.msd_loss_meter.val, self.msd_loss_meter.avg)
            if 'occ_loss' not in output_dict.keys():
                pass
            elif output_dict['occ_loss'] is None:
                pass
            else:
                occ_loss = output_dict['occ_loss'].mean()
                loss += occ_loss
                self.occ_loss_meter.update(val=occ_loss.item(), num=batch_N)
                print_str += ' occ %.4f(%.4f)' % (self.occ_loss_meter.val, self.occ_loss_meter.avg)
            self.photo_loss_meter.update(val=photo_loss.item(), num=batch_N)
            self.smooth_loss_meter.update(val=smooth_loss.item(), num=batch_N)
            print_str += ' ph %.4f(%.4f)' % (self.photo_loss_meter.val, self.photo_loss_meter.avg)
            print_str += ' sm %.4f(%.4f)' % (self.smooth_loss_meter.val, self.smooth_loss_meter.avg)
        else:
            raise ValueError('wrong model: %s' % self.conf.model_name)

        # =========== 计算spatial transform的等变损失 ===========
        if self.conf.if_train_sp:
            # s = {'imgs': [sp_img1, sp_img2], 'flows_f': [flow_fw], 'masks_f': [occ_fw]}
            # s=deepcopy(s)
            if self.conf.sp_input_or_sp_input >= 1:  # 取clean的图片算sp
                sp_img1, sp_img2 = im1_crop_ori, im2_crop_ori
            elif self.conf.sp_input_or_sp_input > 0:
                if tools.random_flag(threshold_0_1=self.conf.sp_input_or_sp_input):
                    sp_img1, sp_img2 = sp_img1_ori, sp_img2_ori
                else:
                    sp_img1, sp_img2 = im1_crop_ori, im2_crop_ori
            else:  # 取final的图片算sp
                sp_img1, sp_img2 = sp_img1_ori, sp_img2_ori
            flow_fw_pseudo_label, occ_fw_pseudo_label = flow_fw.clone().detach(), occ_fw.clone().detach()
            flow_bw_pseudo_label, occ_bw_pseudo_label = flow_bw.clone().detach(), occ_bw.clone().detach()
            # 使用final的数据多train一次
            if self.conf.final_sp_train_weight > 0 and self.conf.sp_input_or_sp_input <= 0:
                input_dict_final_sp = {'im1': sp_img1_ori, 'im2': sp_img2_ori, 'if_loss': False,
                                       'if_final_sp_train': True, 'final_sp_train_style': self.conf.final_sp_train_style,
                                       'final_fw_label': flow_fw_pseudo_label, 'final_fw_occ': occ_fw_pseudo_label,
                                       'final_bw_label': flow_bw_pseudo_label, 'final_bw_occ': occ_bw_pseudo_label}
                output_dict_final_sp = self.net_work(input_dict_final_sp)
                if output_dict_final_sp['final_sp_loss'] is None:
                    pass
                else:
                    final_sp_loss = output_dict_final_sp['final_sp_loss'].mean() * self.conf.final_sp_train_weight
                    loss += final_sp_loss
                    self.final_sp_loss_meter.update(val=final_sp_loss.item(), num=batch_N)
                    print_str += ' finalsp %.4f(%.4f)' % (self.final_sp_loss_meter.val, self.final_sp_loss_meter.avg)

            s = {'imgs': [sp_img1, sp_img2], 'flows_f': [flow_fw_pseudo_label], 'masks_f': [occ_fw_pseudo_label]}
            st_res = self.sp_transform(s)
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]
            # run 2nd pass spatial transform
            im1_crop_st, im2_crop_st = st_res['imgs']
            if self.conf.train_sp_msd_loss_weight > 0:
                input_dict_sp = {'im1': im1_crop_st, 'im2': im2_crop_st, 'if_loss': False,
                                 'if_sp_msd_loss': True, 'fw_pseudo_label': flow_t, 'fw_occ_pseudo': noc_t, 'sp_msd_loss_style': self.conf.train_sp_msd_loss_style}
            else:
                input_dict_sp = {'im1': im1_crop_st, 'im2': im2_crop_st, 'if_loss': False}
            output_dict_sp = self.net_work(input_dict_sp)
            flow_fw_st, flow_bw_st = output_dict_sp['flow_f_out'], output_dict_sp['flow_b_out']

            if not self.conf.spatial_trans_if_mask:
                noc_t = torch.ones_like(noc_t)
            if self.conf.spatial_trans_q <= 0:
                l_atst = (flow_fw_st - flow_t).abs()
            else:
                l_atst = ((flow_fw_st - flow_t).abs() + self.conf.spatial_trans_eps) ** self.conf.spatial_trans_q
            l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-6)
            l_atst *= self.conf.spatial_trans_loss_weight
            loss += l_atst
            self.spatial_trans_loss_meter.update(val=l_atst.item(), num=batch_N)
            print_str += ' sp %.4f(%.4f)' % (self.spatial_trans_loss_meter.val, self.spatial_trans_loss_meter.avg)

            if output_dict_sp['sp_msd_loss'] is None:
                pass
            else:
                sp_msd_loss = output_dict_sp['sp_msd_loss'].mean() * self.conf.train_sp_msd_loss_weight
                loss += sp_msd_loss
                self.sp_msd_loss_meter.update(val=sp_msd_loss.item(), num=batch_N)
                print_str += ' spmsd %.4f(%.4f)' % (self.sp_msd_loss_meter.val, self.sp_msd_loss_meter.avg)

        loss.backward()
        self.optimizer.step()

        # show training
        if if_print:
            print(print_str)
        # show img
        if if_show:
            if_show_bw = False  # 是否展示backward flow过程
            # base thing
            im1_crop, im2_crop, im1_warp, flow_fw, occ_fw = tools.tensor_gpu(im1_crop, im2_crop, im1_warp, flow_fw, occ_fw, check_on=False)
            frame_1_ls += [('im1', im1_crop), ('im1     ', im1_crop), ('flow forward', flow_fw), ('occ forward', occ_fw)]
            frame_2_ls += [('im2', im2_crop), ('im1_warp', im1_warp), ('flow forward', flow_fw), ('occ forward', occ_fw)]
            if if_show_bw:
                im2_warp, flow_bw, occ_bw = tools.tensor_gpu(im2_warp, flow_bw, occ_bw, check_on=False)
                frame_1_ls += [('im2_warp', im2_warp), ('flow backward', flow_bw), ('occ backward', occ_bw)]
                frame_2_ls += [('im2', im2_crop), ('flow backward', flow_bw), ('occ backward', occ_bw)]

            # ============================ 有的模型 要加一些操作 =====================
            if self.conf.model_name.lower() == '加油':
                pass
            elif self.conf.model_name.lower() in ['pwcirrbiv5', ]:
                pass
            elif self.conf.model_name.lower() in ['pwcirrbiv5_v5', ]:
                fw_im1 = output_dict['im1_warp_ss']
                fw_im2 = output_dict['im2_warp_ss']
                fw_im1_, fw_im2_ = tools.tensor_gpu(fw_im1.clone().detach(), fw_im2.clone().detach(), check_on=False)
                frame_1_ls += [('fw_im1_', fw_im1_), ('fw_im2_', fw_im2_), ]
                frame_2_ls += [('fw_im1_', fw_im1_), ('fw_im2_', fw_im2_), ]
            else:
                pass  # no operation
                # raise ValueError(' not implemented model name: %s' % self.conf.model_name)

            self.training_shower.get_batch_pair_all_list_nchw_check_flow_frame1_frame2_gif(batch_dict_ls_frame1=frame_1_ls, batch_dict_ls_frame2=frame_2_ls,
                                                                                           name='iter_%s_%s' % (batch_step, print_str))
            self.training_shower.put_frame1_frame2_gif(name='Epoch %d Iteration %d ' % (self.epoch, batch_step))
        # compute data time
        self.data_clock.start()

    @classmethod
    def save_image_v2(cls, tensor_data, name, save_dir_dir, mask_or_flow_or_image='image', if_flow_data_save_png=False):
        def decom(a):
            b = tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        if mask_or_flow_or_image == 'flow':
            flow_f_np = decom(tensor_data)
            if flow_f_np.shape[2] == 2:
                cv2.imwrite(os.path.join(save_dir_dir, name + '_s.png'), tools.flow_to_image(flow_f_np)[:, :, ::-1])
                if if_flow_data_save_png:
                    save_path = os.path.join(save_dir_dir, name + '.png')
                    tools.write_kitti_png_file(save_path, flow_f_np)
            elif flow_f_np.shape[2] == 3:
                flow = flow_f_np[:, :, :2]
                mask = flow_f_np[:, :, 2]
                cv2.imwrite(os.path.join(save_dir_dir, name + '_s.png'), tools.flow_to_image(flow)[:, :, ::-1])
                if if_flow_data_save_png:
                    save_path = os.path.join(save_dir_dir, name + '.png')
                    tools.write_kitti_png_file(save_path, flow, mask_data=mask)
            else:
                raise ValueError('flow_f_np shape not right: %s' % flow_f_np.shape)
        elif mask_or_flow_or_image == 'mask':
            mask = decom(tensor_data)
            cv2.imwrite(os.path.join(save_dir_dir, name + '.png'), tools.Show_GIF.im_norm(mask * 255))
        elif mask_or_flow_or_image == 'image':
            img1_np = tools.Show_GIF.im_norm(decom(tensor_data))
            # img1_np = decom(tensor_data)
            cv2.imwrite(os.path.join(save_dir_dir, name + '.png'), img1_np[:, :, ::-1])
        else:
            raise ValueError('wrong data type: %s' % mask_or_flow_or_image)

    def eval_forward(self, im1, im2, flow, *args):
        # ==================================================================== 网络输出 ======================================================================
        with torch.no_grad():
            if self.conf.model_name.lower() == '加油':
                flow_fw, flow_bw, app_flow_1, app_flow_2, _, _ = self.net_work(im1, im2)  # flow from im1->im2
                pred_flow = flow_fw
            elif self.conf.model_name.lower() in ['pwcirrbiv5_v4', 'pwcirrbiv5_v5']:
                input_dict = {'im1': im1, 'im2': im2, 'if_loss': False, 'if_test': True}
                output_dict = self.net_work(input_dict)
                flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
                flows = output_dict['flows']
                pred_flow = flow_fw
            elif self.conf.model_name.lower() in ['pwcirrbiv5_v4_show', 'pwcirrbiv5_v5_show']:
                if self.conf.save_running_process:
                    running_process_dir = os.path.join(self.training_shower.save_dir, 'running_process')
                    sample_dir = os.path.join(running_process_dir, '%s' % self.cnt)
                    tools.check_dir(sample_dir)
                    input_dict = {'im1': im1, 'im2': im2, 'if_loss': False, 'if_test': True, 'save_running_process': True, 'process_dir': sample_dir}
                else:
                    input_dict = {'im1': im1, 'im2': im2, 'if_loss': False, 'if_test': True}
                output_dict = self.net_work(input_dict)
                flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
                flows = output_dict['flows']
                pred_flow = flow_fw
                # print('======')
                # tools.check_tensor(flow, 'gt flow')
                # tools.check_tensor(flow_fw, 'output_flow_fw')
                # for i, (fw, fb) in enumerate(flows):
                #     tools.check_tensor(fw, '%s scale fw' % i)
                # print('======')
                if len(args) > 0 and self.conf.save_running_process:
                    def decom(a):
                        b = tools.tensor_gpu(a, check_on=False)[0]
                        c = b[0, :, :, :]
                        c = np.transpose(c, (1, 2, 0))
                        return c

                    occ_mask = args[0]
                    # save gt occ_mask
                    gt_occ_mask_np = decom(occ_mask)
                    gt_flow_np = decom(flow)
                    pred_flow_np = decom(pred_flow)
                    # save gt_flow
                    cv2.imwrite(os.path.join(sample_dir, 'gt_flow_np' + '.png'), tools.flow_to_image(gt_flow_np)[:, :, ::-1])
                    # save pred flow_f
                    cv2.imwrite(os.path.join(sample_dir, 'pred_flow_f' + '.png'), tools.flow_to_image(pred_flow_np)[:, :, ::-1])

                    # show flow gt error image
                    flow_error_image = tools.lib_to_show_flow.flow_error_image_np(pred_flow_np, gt_flow_np, gt_occ_mask_np)
                    # print('flow_error_image', np.max(flow_error_image), np.min(flow_error_image))
                    cv2.imwrite(os.path.join(sample_dir, 'flow_error_image' + '.png'), tools.Show_GIF.im_norm(flow_error_image))
                    flow_error_image_gray = tools.lib_to_show_flow.flow_error_image_np(pred_flow_np, gt_flow_np, gt_occ_mask_np, log_colors=False)
                    cv2.imwrite(os.path.join(sample_dir, 'flow_error_image_gray' + '.png'), tools.Show_GIF.im_norm(flow_error_image_gray))
                if self.conf.model_name.lower() == 'pwcirrbiv5_show_v3':  # save some results
                    occmask, noc_gt_flow, nocmask = args
                    save_dir = os.path.join(self.training_shower.save_dir, 'saving_res')
                    tools.check_dir(save_dir)
                    dir_name = '%s' % self.cnt  # + '_occ_%.3f_'%occ_value.item()
                    save_dir_dir = os.path.join(save_dir, dir_name)
                    tools.check_dir(save_dir_dir)
                    occ_fw = output_dict['occ_fw']
                    self.save_image_v2(flow_fw, 'flow_f', save_dir_dir, 'flow', True)
                    self.save_image_v2(flow, 'gt', save_dir_dir, 'flow', True)
                    self.save_image_v2(noc_gt_flow, 'noc_gt_flow', save_dir_dir, 'flow', True)
                    self.save_image_v2(occmask, 'gt_occ_mask', save_dir_dir, 'mask', False)
                    self.save_image_v2(nocmask, 'gt_noc_mask', save_dir_dir, 'mask', False)
                    self.save_image_v2(im1, 'im1', save_dir_dir, 'image', False)
                    self.save_image_v2(im2, 'im2', save_dir_dir, 'image', False)
                    self.save_image_v2(occ_fw, 'occ_mask', save_dir_dir, 'mask', False)
            else:
                raise ValueError(' not implemented model name: %s' % self.conf.model_name)
        if self.conf.if_do_eval:  # 管理在测试或者验证过程中， 是否展示gif结果，
            self.cnt += 1
            if self.conf.if_do_test:
                im1_warp = tools.torch_warp(im2, pred_flow)
                warp_error = torch.sqrt((im1_warp - im1) ** 2)
                warp_error = warp_error.mean()
                print_str = 'iter_%s warp_error%.5f' % (self.cnt, warp_error.item())
                self.print_str = print_str
                if self.conf.if_test_save_show_results:
                    im1_np, im2_np, pred_flow_np, im1_warp_np = tools.tensor_gpu(im1, im2, pred_flow, im1_warp, check_on=False)
                    frame_1_ls = [('im1', im1_np), ('im1          ', im1_np), ('im1_warp', im1_warp_np), ('flow pred', pred_flow_np)]
                    frame_2_ls = [('im2', im2_np), ('im1_warp', im1_warp_np), ('im1_warp', im1_warp_np), ('flow pred', pred_flow_np)]
                    self.training_shower.get_batch_pair_all_list_nchw_check_flow_frame1_frame2_gif(batch_dict_ls_frame1=frame_1_ls, batch_dict_ls_frame2=frame_2_ls,
                                                                                                   name=print_str)
            else:
                im1_warp = tools.torch_warp(im2, pred_flow)
                im1_gt_warp = tools.torch_warp(im2, flow)

                warp_error = torch.sqrt((im1_warp - im1) ** 2)
                warp_error = warp_error.mean()

                gt_warp_error = torch.sqrt((im1_warp - im1_gt_warp) ** 2)
                gt_warp_error = gt_warp_error.mean()
                print_str = 'iter_%s warp_error%.5f gtwarperror_%.5f' % (self.cnt, warp_error.item(), gt_warp_error.item())
                self.print_str = print_str
                if self.conf.if_do_eval_save_show_result:
                    im1_np, im2_np, gt_flow_np, pred_flow_np, im1_warp_np, im1_gt_warp_np = tools.tensor_gpu(im1, im2, flow, pred_flow, im1_warp, im1_gt_warp, check_on=False)
                    frame_1_ls = [('im1', im1_np), ('im1', im1_np), ('gt_warp_im1', im1_gt_warp_np), ('flow pred', pred_flow_np)]
                    frame_2_ls = [('im2', im2_np), ('im1_warp', im1_warp_np), ('im1_warp', im1_warp_np), ('flow gt', gt_flow_np)]
                    self.training_shower.get_batch_pair_all_list_nchw_check_flow_frame1_frame2_gif(batch_dict_ls_frame1=frame_1_ls, batch_dict_ls_frame2=frame_2_ls,
                                                                                                   name=print_str)
        if self.conf.if_save_flow_in_eval_or_test:  # 管理是否保存flow结果，存为.png或者.flo
            self.temp_save_eval_test = [pred_flow, flow]  # 缓存起来
        if self.conf.multi_scale_eval:
            return pred_flow, flows
        return pred_flow

    def eval_save_result(self, save_name, *args, **kwargs):
        def flow_tensor_np_h_w_2(a):
            a_np = tools.tensor_gpu(a, check_on=False)[0]
            a_np = a_np[0, :, :, :]  # n,c,h,w
            a_np = np.transpose(a_np, (1, 2, 0))  # h,w,2
            return a_np

        if self.conf.if_do_eval:
            if self.conf.if_do_eval_print:
                print(self.print_str + '  ' + save_name)
            if self.conf.if_do_test:
                if self.conf.if_test_save_show_results:
                    if len(args) > 0:
                        sample_dir_name = args[0]
                        self.training_shower.put_frame1_frame2_gif(name=sample_dir_name + '_' + save_name + '_' + self.print_str)
                    else:
                        self.training_shower.put_frame1_frame2_gif(name=save_name + '_' + self.print_str)
            else:
                if self.conf.if_do_eval_save_show_result:
                    self.training_shower.put_frame1_frame2_gif(name=save_name + '_' + self.print_str)

            if self.conf.if_save_flow_in_eval_or_test:
                if self.conf.if_do_test:  # test
                    save_dir = os.path.join(self.training_shower.save_dir, 'save_test_flow')
                    tools.check_dir(save_dir)
                    if len(args) > 0:
                        sample_dir_name = args[0]
                        if type(sample_dir_name) == str:
                            save_dir = os.path.join(save_dir, sample_dir_name)
                            tools.check_dir(save_dir)
                    if self.conf.if_save_flow_in_eval_or_test_type == 'png':
                        pred_flow, _ = self.temp_save_eval_test
                        pred_flow_np = flow_tensor_np_h_w_2(pred_flow)
                        save_path = os.path.join(save_dir, save_name + '.png')
                        # 2015上这样save是可以用的,但同样的save方法2012就不能用了
                        tools.write_kitti_png_file(save_path, pred_flow_np)
                        # 尝试一个新的方式
                        # tools.lib_to_show_flow.flow_write_png( u=pred_flow_np[:,:,0], v=pred_flow_np[:,:,1], fpath=save_path)
                    elif self.conf.if_save_flow_in_eval_or_test_type == 'flo':
                        pred_flow, _ = self.temp_save_eval_test
                        pred_flow_np = flow_tensor_np_h_w_2(pred_flow)
                        save_path = os.path.join(save_dir, save_name + '.flo')
                        tools.write_flo(flow=pred_flow_np, filename=save_path)  # write_flow, or, write_flo
                    else:
                        raise ValueError('wrong if_save_flow_eval_test_type, should be png or flo, but got: %s' % self.conf.if_save_flow_in_eval_or_test_type)
                else:  # eval
                    save_dir = os.path.join(self.training_shower.save_dir, 'eval_test')
                    tools.check_dir(save_dir)
                    if len(args) > 0:
                        sample_dir_name = args[0]
                        if type(sample_dir_name) == str:
                            save_dir = os.path.join(save_dir, sample_dir_name)
                            tools.check_dir(save_dir)
                    if self.conf.if_save_flow_in_eval_or_test_type == 'png':
                        pred_flow, gt_flow = self.temp_save_eval_test
                        pred_flow_np = flow_tensor_np_h_w_2(pred_flow)
                        save_path = os.path.join(save_dir, save_name + '.png')
                        tools.write_kitti_png_file(save_path, pred_flow_np)
                        # tools.WriteKittiPngFile(save_path, pred_flow_np)
                        # save gt
                        gt_save_path = os.path.join(save_dir, save_name + '_gt.png')
                        tools.write_kitti_png_file(gt_save_path, flow_tensor_np_h_w_2(gt_flow))
                    elif self.conf.if_save_flow_in_eval_or_test_type == 'flo':
                        pred_flow, gt_flow = self.temp_save_eval_test
                        pred_flow_np = flow_tensor_np_h_w_2(pred_flow)
                        save_path = os.path.join(save_dir, save_name + '.flo')
                        tools.write_flo(flow=pred_flow_np, filename=save_path)  # write_flow, or, write_flo
                        gt_save_path = os.path.join(save_dir, save_name + '_gt.flo')
                        gt_flow_np = flow_tensor_np_h_w_2(gt_flow)
                        tools.write_flo(flow=gt_flow_np, filename=gt_save_path)  # write_flow, or, write_flo
                        if_check = True
                        if if_check:
                            temp = tools.read_flo(save_path)  # read_flow, or, read_flo
                            temp_gt = tools.read_flo(gt_save_path)  # read_flow, or, read_flo
                            temp_error = temp - pred_flow_np
                            temp_gt_error = temp_gt - gt_flow_np
                            print('pred flow save write .flo file, error: ', np.min(temp_error), np.max(temp_error), 'gt r.w.error: ', np.min(temp_gt_error), np.max(temp_gt_error))
                    else:
                        raise ValueError('wrong if_save_flow_eval_test_type, should be png or flo, but got: %s' % self.conf.if_save_flow_in_eval_or_test_type)

    def train(self, epoch=0):  # 进入训练状态
        # torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        self.net_work.train()
        self.app_loss_meter.reset()
        self.appd_loss_meter.reset()
        self.msd_loss_meter.reset()
        self.sp_msd_loss_meter.reset()
        self.final_sp_loss_meter.reset()
        self.occ_loss_meter.reset()
        self.spatial_trans_loss_meter.reset()
        self.photo_loss_meter.reset()
        self.smooth_loss_meter.reset()
        self.inpaint_img_loss_meter.reset()
        self.multi_scale_loss_meter.reset()
        if epoch % 1 == 0:
            self.scheduler.step()
        print('epoch', epoch, 'lr={:.6f}'.format(self.scheduler.get_lr()[0]))
        self.epoch = epoch


class Train_Config(tools.abstract_config):

    def __init__(self, **kwargs):
        self.batchsize = 4
        self.gpu_opt = None  # gpu option
        self.n_epoch = 1000  # number of epoch
        self.if_eval = True  # do evaluation during the training process
        self.train_data_name = 'kitti_2015_mv'  # or kitti_2012_mv
        self.eval_data_name = '2015_train'  # or 2015_train
        self.eval_per = -1  # do evaluation every N iters
        self.eval_batchsize = 1  # batch size for evaluation
        self.use_prefether = True  # faster loader
        self.if_histmatch = False  # do not use this
        self.update(kwargs)


class Training():
    def __init__(self, **kwargs):
        self.conf = Train_Config(**kwargs)
        self.data_conf = self.get_train_data(**kwargs)

    def get_train_data(self, **kwargs):
        '''
        get dataset
            data config = {
                'crop_size': (256, 832),
                'rho': 8,
                'swap_images': True,
                'normalize': True,
                'horizontal_flip_aug': True,
            }
        '''
        if self.conf.train_data_name == 'kitti_2015_mv':
            data_conf = kitti_train.kitti_data_with_start_point.config(mv_type='2015', **kwargs)
        elif self.conf.train_data_name == 'kitti_2012_mv':
            data_conf = kitti_train.kitti_data_with_start_point.config(mv_type='2012', **kwargs)
        else:
            raise ValueError('not implemented train data: %s' % self.conf.train_data_name)
        return data_conf

    def get_network(self):
        pass

    def get_eval_benchmark(self):
        pass

    def do_training(self):
        pass


param_dict = {
    # training
    'batchsize': 4,
    'gpu_opt': None,
    'n_epoch': 1000,
    'if_eval': True,
    'train_data_name': 'kitti_2015_mv',
    'eval_data_name': '2015_train',  # or 2015_train
    'eval_per': -1,  # 隔多少个iter做验证
    'eval_batchsize': 1,  # 验证batch size
    'use_prefether': True,  # 这个速度快一点，会好一点

    # data
    'crop_size': (256, 832),
    'rho': 8,
    'swap_images': True,
    'normalize': True,
    'horizontal_flip_aug': True,

    # network


}
