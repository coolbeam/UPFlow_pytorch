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

# save and log loss value during training
class Loss_manager():
    def __init__(self):
        self.error_meter = tools.Avg_meter_ls()

    def fetch_loss(self, loss, loss_dict, name, batch_N, short_name=None):
        if name not in loss_dict.keys():
            pass
        elif loss_dict[name] is None:
            pass
        else:
            this_loss = loss_dict[name].mean()
            self.error_meter.update(name=name, val=this_loss.item(), num=batch_N, short_name=short_name)
            loss = loss + this_loss
        return loss

    def prepare_epoch(self):
        self.error_meter.reset()

    def log_info(self):
        p_str = self.error_meter.print_all_losses()
        return p_str

    def compute_loss(self, loss_dict, batch_N):
        loss = 0
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='photo_loss', short_name='ph', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='smooth_loss', short_name='sm', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='census_loss', short_name='cen', batch_N=batch_N)
        # photo_loss, smooth_loss, census_loss = output_dict['photo_loss'].mean(), output_dict['smooth_loss'], output_dict['census_loss']
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='msd_loss', short_name='msd', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='eq_loss', short_name='eq', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='oi_loss', short_name='oi', batch_N=batch_N)
        return loss

class Eval_model(tools.abs_test_model):
    def __init__(self):
        super(Eval_model, self).__init__()
        self.net_work = None

    def eval_forward(self, im1, im2, gt, *args):
        if self.net_work is None:
            raise ValueError('not network for evaluation')
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        # print(save_name)
        pass

    def change_model(self, net):
        net.eval()
        self.net_work = net


class Trainer():
    class Config(tools.abstract_config):
        def __init__(self, **kwargs):
            self.exp_dir = './demo_exp'
            self.if_cuda = True

            self.batchsize = 2
            self.NUM_WORKERS = 4
            self.n_epoch = 1000
            self.batch_per_epoch = 500
            self.batch_per_print = 20
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.scheduler_gamma = 1

            # init
            self.update(kwargs)

        def __call__(self, ):
            t = Trainer(self)
            return t

    def __init__(self, conf: Config):
        self.conf = conf

        tools.check_dir(self.conf.exp_dir)

        # load network
        self.net = self.load_model()

        # for evaluation
        self.bench = self.load_eval_bench()
        self.eval_model = Eval_model()

        # load training dataset
        self.train_set = self.load_training_dataset()

    def training(self):
        train_loader = tools.data_prefetcher(self.train_set, batch_size=self.conf.batchsize, shuffle=True, num_workers=self.conf.NUM_WORKERS, pin_memory=True, drop_last=True)
        optimizer = optim.Adam(self.net.parameters(), lr=self.conf.lr, amsgrad=True, weight_decay=self.conf.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.conf.scheduler_gamma)
        loss_manager = Loss_manager()
        timer = tools.time_clock()
        print("start training" + '=' * 10)
        i_batch = 0
        epoch = 0
        loss_manager.prepare_epoch()
        current_val, best_val, best_epoch = 0, 0, 0
        timer.start()
        while True:
            # prepare batch data
            batch_value = train_loader.next()
            if batch_value is None:
                batch_value = train_loader.next()
                assert batch_value is not None
            batchsize = batch_value['im1'].shape[0] # check if the im1 exists
            i_batch += 1
            # train batch
            self.net.train()
            optimizer.zero_grad()
            out_data = self.net(batch_value)  #

            loss_dict = out_data['loss_dict']
            loss = loss_manager.compute_loss(loss_dict=loss_dict, batch_N=batchsize)

            loss.backward()
            optimizer.step()
            if i_batch % self.conf.batch_per_print == 0:
                pass
            if i_batch % self.conf.batch_per_epoch == 0:
                # do eval  and check if save model todo===
                epoch+=1
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                scheduler.step(epoch=epoch)
                timer.start()

    def evaluation(self):
        self.eval_model.change_model(self.net)
        epe_all, f1, epe_noc, epe_occ = self.bench(self.eval_model)
        print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))
        print_str = 'EPE_%.2f__F1_%.2f__Noc_%.2f__Occ_%.2f' % (epe_all, f1, epe_noc, epe_occ)
        return epe_all, print_str

    # ======
    def load_model(self):
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': False,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': False,  # 先把这个关掉跑通吧
        }
        pretrain_path = None  # pretrain path
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()  # .cuda()
        if pretrain_path is not None:
            net.load_model(pretrain_path, if_relax=True, if_print=False)
        if self.conf.if_cuda:
            net = net.cuda()
        return net

    def load_eval_bench(self):
        bench = kitti_flow.Evaluation_bench(name='2015_train', if_gpu=self.conf.if_cuda, batch_size=1)
        return bench

    def load_training_dataset(self):
        data_config = {
            'crop_size': (256, 832),
            'rho': 8,
            'swap_images': True,
            'normalize': True,
            'horizontal_flip_aug': True,
        }
        data_conf = kitti_train.kitti_data_with_start_point.config(mv_type='2015', **data_config)
        dataset = data_conf()
        return dataset


if __name__ == '__main__':
    training_param = {}  # change param here
    conf = Trainer.Config(**training_param)
    trainer = conf()
    trainer.training()
