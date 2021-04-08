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

if_cuda = True


class Test_model(tools.abs_test_model):
    def __init__(self, pretrain_path='./scripts/upflow_kitti2015.pth'):
        super(Test_model, self).__init__()
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': False,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': True,
        }
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()  # .cuda()
        net.load_model(pretrain_path, if_relax=True, if_print=True)
        if if_cuda:
            net = net.cuda()
        net.eval()
        self.net_work = net

    def eval_forward(self, im1, im2, gt, *args):
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        print(save_name)


def kitti_2015_test():
    pretrain_path = './scripts/upflow_kitti2015.pth'
    # note that eval batch size should be 1 for KITTI 2012 and KITTI 2015 (image size may be different for different sequence)
    bench = kitti_flow.Evaluation_bench(name='2015_train', if_gpu=if_cuda, batch_size=1)
    testmodel = Test_model(pretrain_path=pretrain_path)
    epe_all, f1, epe_noc, epe_occ = bench(testmodel)
    print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))


if __name__ == '__main__':
    kitti_2015_test()
