import torch
from torch.utils.data.dataloader import _DataLoaderIter, DataLoader  # need torch.__version__ == '1.1.0'
# from torch.utils.data import DataLoader
# if torch.__version__ == '1.5.1' or torch.__version__ == '1.4.0' use this
# from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as _DataLoaderIter
# from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import imageio
import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn.init import xavier_normal, kaiming_normal
from torch.utils.data import Dataset
import pickle
import argparse
import collections
import random
from shutil import rmtree
import time
import zipfile
import png
import array
import warnings
import shutil


class tools():
    class abstract_config():
        name_filter_out_list = []  # some attributes should not appear in the file name. Write its name here

        def get_name(self, print_now=True):
            temp = dir(self)
            name_filter_out_list = self.name_filter_out_list + ['name_filter_out_list', 'get_name', 'update', 'update_ex_name', 'get_dict',
                                                                'check_length_of_file_path', 'check_length_of_file_name']
            temp = list(filter(lambda x: True if x.find('__') < 0 and x not in name_filter_out_list else False, temp))
            temp = sorted(temp)
            if print_now:
                norm_length = 50
                print('=' * 10)
                print('{')
                for i in temp:
                    temp_str = "'%s'" % i
                    if len(temp_str) < norm_length:
                        temp_str += ' ' * (norm_length - len(temp_str))
                    try:
                        temp_str += ": '%s,%s', " % (getattr(self, i), '')
                    except:
                        continue
                    # print((i, getattr(self, i)))
                    print('\t' + temp_str)
                print('}')
                print('=' * 10)
            name = ''
            for i in temp:
                name += '%s|%s_' % (i, getattr(self, i))
            return name

        @classmethod
        def check_length_of_file_name(cls, file_name):
            if len(file_name) >= 255:
                return False
            else:
                return True

        @classmethod
        def check_length_of_file_path(cls, filepath):
            if len(filepath) >= 4096:
                return False
            else:
                return True

        def update(self, data: dict):
            def dict_class(obj):
                temp = {}
                for name in dir(obj):
                    value = getattr(obj, name)
                    if not name.startswith('_'):
                        temp[name] = value
                return temp

            s_dict = dict_class(self)
            t_key = list(data.keys())
            for i in s_dict.keys():
                if i in t_key:
                    setattr(self, i, data[i])
                    print('set param ====  %s:   %s' % (i, data[i]))

        def get_dict(self):
            def dict_class(obj):
                temp = {}
                for name in dir(obj):
                    value = getattr(obj, name)
                    if not name.startswith('_'):
                        temp[name] = value
                return temp

            s_dict = dict_class(self)
            return s_dict

        def update_ex_name(self, ex_name: str):
            return ex_name

    class abstract_model(nn.Module):

        def save_model(self, save_path):
            torch.save(self.state_dict(), save_path)

        def load_model(self, load_path, if_relax=False, if_print=True):
            if if_print:
                print('loading protrained model from %s' % load_path)
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = torch.load(load_path)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(load_path))

        @classmethod
        def choose_gpu(cls, model, gpu_opt=None):
            # choose gpu
            if gpu_opt is None:
                # gpu=0
                model = model.cuda()
                # torch.cuda.set_device(gpu)
                # model.cuda(gpu)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                # print('torch.cuda.device_count()  ',torch.cuda.device_count())
                # model=torch.nn.parallel.DistributedDataParallel(model,device_ids=range(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # multi gpu
            elif gpu_opt == 0:
                model = model.cuda()
            else:
                if type(gpu_opt) != int:
                    raise ValueError('wrong gpu config, it show be int:  %s' % (str(gpu_opt)))
                torch.cuda.set_device(gpu_opt)
                model = model.cuda(gpu_opt)
            return model

        @classmethod
        def save_model_gpu(cls, model, path):
            name_dataparallel = torch.nn.DataParallel.__name__
            if type(model).__name__ == name_dataparallel:
                model = model.module
            model.save_model(path)

    class abs_test_model():
        save_dir = ''

        def eval_forward(self, im1, im2, gt, *args):  # do model forward and cache forward results
            return 0

        def eval_save_result(self, save_name, predflow, *args, **kwargs):
            pass

    class data_prefetcher():

        def __init__(self, dataset, gpu_opt=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
            self.dataset = dataset
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
            # self.loader = iter(loader)
            self.loader = _DataLoaderIter(loader)
            self.stream = torch.cuda.Stream()
            self.gpu_opt = gpu_opt

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last

        def build(self):
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)
            self.loader = _DataLoaderIter(loader)
            # self.loader = iter(loader)

        def next(self):
            try:
                # batch = next(self.loader)
                batch = self.loader.next()
            except StopIteration:
                self.build()
                return None
            # print('self.batch',type(self.batch))
            # for i in range(len(self.batch)):
            #     print('i',i,type(self.batch[i]))
            with torch.cuda.stream(self.stream):
                batch = tools.tensor_gpu(*batch, check_on=True, non_blocking=True, gpu_opt=self.gpu_opt)
                # self.next_img = self.next_img.cuda(non_blocking=True).float()
                # self.next_seg = self.next_seg.cuda(non_blocking=True).float()
                # self.next_weight = self.next_weight.cuda(non_blocking=True)
                # self.mask2 = self.mask2.cuda(non_blocking=True).float()
                # self.mask3 = self.mask3.cuda(non_blocking=True).float()

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
                # self.next_input = self.next_input.float()
                # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

            return batch

    class DataProvider:

        def __init__(self, dataset, batch_size, shuffle=True, num_worker=4, drop_last=True, pin_memory=True):
            self.batch_size = batch_size
            self.dataset = dataset
            self.dataiter = None
            self.iteration = 0  #
            self.epoch = 0  #
            self.shuffle = shuffle
            self.pin_memory = pin_memory
            self.num_worker = num_worker
            self.drop_last = drop_last

        def build(self):
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_worker,
                                    pin_memory=self.pin_memory,
                                    drop_last=self.drop_last)
            self.dataiter = _DataLoaderIter(dataloader)

        def next(self):
            if self.dataiter is None:
                self.build()
            try:
                batch = self.dataiter.next()
                self.iteration += 1

                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

            except StopIteration:  # ??epoch???reload
                self.epoch += 1
                self.build()
                self.iteration = 1  # reset and return the 1st batch

                batch = self.dataiter.next()
                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

    # read/write something as npz
    class npz_saver():

        @classmethod
        def save_npz(cls, files, npz_save_path):
            np.savez(npz_save_path, files=[files, 0])

        @classmethod
        def load_npz(cls, npz_save_path):
            with np.load(npz_save_path) as fin:
                files = fin['files']
                files = list(files)
                return files[0]

    # read/write something as pkl
    class pickle_saver():

        @classmethod
        def save_pickle(cls, files, file_path):
            with open(file_path, 'wb') as data:
                pickle.dump(files, data)

        @classmethod
        def load_picke(cls, file_path):
            with open(file_path, 'rb') as data:
                data = pickle.load(data)
            return data

    class AverageMeter():

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, num):
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    class Avg_meter_ls():
        def __init__(self):
            self.data_ls = {}
            self.short_name_ls = {}

        def update(self, name, val, num, short_name=None):
            if name not in self.data_ls.keys():
                self.data_ls[name] = tools.AverageMeter()
                if short_name is None:
                    short_name = name
                self.short_name_ls[name] = short_name
            self.data_ls[name].update(val=val, num=num)

        def print_loss(self, name):
            a = ' %s %.4f(%.4f)' % (self.short_name_ls[name], self.data_ls[name].val, self.data_ls[name].avg)
            return a

        def print_all_losses(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s %.4f(%.4f)' % (self.short_name_ls[i], self.data_ls[i].val, self.data_ls[i].avg)
            return a

        def reset(self):
            for name in self.data_ls.keys():
                self.data_ls[name].reset()

    # tik tok
    class time_clock():

        def __init__(self):
            self.st = 0
            self.en = 0
            self.start_flag = False

        def start(self):
            self.reset()
            self.start_flag = True
            self.st = time.time()

        def reset(self):
            self.start_flag = False
            self.st = 0
            self.en = 0

        def end(self):
            self.en = time.time()

        def get_during(self):
            return self.en - self.st

    # boundary dilated warping
    class boundary_dilated_warp():

        @classmethod
        def get_grid(cls, batch_size, H, W, start):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            ones = torch.ones_like(xx)
            grid = torch.cat((xx, yy, ones), 1).float()
            if torch.cuda.is_available():
                grid = grid.cuda()
            # print("grid",grid.shape)
            # print("start", start)
            grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量

            return grid

        @classmethod
        def transformer(cls, I, vgrid, train=True):
            # I: Img, shape: batch_size, 1, full_h, full_w
            # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
            # outsize: (patch_h, patch_w)

            def _repeat(x, n_repeats):

                rep = torch.ones([n_repeats, ]).unsqueeze(0)
                rep = rep.int()
                x = x.int()

                x = torch.matmul(x.reshape([-1, 1]), rep)
                return x.reshape([-1])

            def _interpolate(im, x, y, out_size, scale_h):
                # x: x_grid_flat
                # y: y_grid_flat
                # out_size: same as im.size
                # scale_h: True if normalized
                # constants
                num_batch, num_channels, height, width = im.size()

                out_height, out_width = out_size[0], out_size[1]
                # zero = torch.zeros_like([],dtype='int32')
                zero = 0
                max_y = height - 1
                max_x = width - 1
                if scale_h:
                    # scale indices from [-1, 1] to [0, width or height]
                    # print('--Inter- scale_h:', scale_h)
                    x = (x + 1.0) * (height) / 2.0
                    y = (y + 1.0) * (width) / 2.0

                # do sampling
                x0 = torch.floor(x).int()
                x1 = x0 + 1
                y0 = torch.floor(y).int()
                y1 = y0 + 1

                x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
                x1 = torch.clamp(x1, zero, max_x)
                y0 = torch.clamp(y0, zero, max_y)
                y1 = torch.clamp(y1, zero, max_y)

                dim1 = torch.from_numpy(np.array(width * height))
                dim2 = torch.from_numpy(np.array(width))

                base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
                # base = torch.arange(0,num_batch) * dim1
                # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
                # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
                if torch.cuda.is_available():
                    dim2 = dim2.cuda()
                    dim1 = dim1.cuda()
                    y0 = y0.cuda()
                    y1 = y1.cuda()
                    x0 = x0.cuda()
                    x1 = x1.cuda()
                    base = base.cuda()
                base_y0 = base + y0 * dim2
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im = im.permute(0, 2, 3, 1)
                im_flat = im.reshape([-1, num_channels]).float()

                idx_a = idx_a.unsqueeze(-1).long()
                idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
                Ia = torch.gather(im_flat, 0, idx_a)

                idx_b = idx_b.unsqueeze(-1).long()
                idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
                Ib = torch.gather(im_flat, 0, idx_b)

                idx_c = idx_c.unsqueeze(-1).long()
                idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
                Ic = torch.gather(im_flat, 0, idx_c)

                idx_d = idx_d.unsqueeze(-1).long()
                idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
                Id = torch.gather(im_flat, 0, idx_d)

                # and finally calculate interpolated values
                x0_f = x0.float()
                x1_f = x1.float()
                y0_f = y0.float()
                y1_f = y1.float()

                wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
                wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
                wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
                wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
                output = wa * Ia + wb * Ib + wc * Ic + wd * Id

                return output

            def _transform(I, vgrid, scale_h):

                C_img = I.shape[1]
                B, C, H, W = vgrid.size()

                x_s_flat = vgrid[:, 0, ...].reshape([-1])
                y_s_flat = vgrid[:, 1, ...].reshape([-1])
                out_size = vgrid.shape[2:]
                input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

                output = input_transformed.reshape([B, H, W, C_img])
                return output

            # scale_h = True
            output = _transform(I, vgrid, scale_h=False)
            if train:
                output = output.permute(0, 3, 1, 2)
            return output

        @classmethod
        def warp_im(cls, I_nchw, flow_nchw, start_n211):
            batch_size, _, img_h, img_w = I_nchw.size()
            _, _, patch_size_h, patch_size_w = flow_nchw.size()
            patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
            vgrid = patch_indices[:, :2, ...]
            # grid_warp = vgrid - flow_nchw
            grid_warp = vgrid + flow_nchw
            pred_I2 = cls.transformer(I_nchw, grid_warp)
            return pred_I2

    class occ_check_model():

        def __init__(self, occ_type='for_back_check', occ_alpha_1=1.0, occ_alpha_2=0.05, sum_abs_or_squar=True, obj_out_all='all'):
            '''
            :param occ_type: method to check occ mask: bidirection check, or froward warping check(not implemented)
            :param occ_alpha_1: threshold
            :param occ_alpha_2: threshold
            :param obj_out_all: occ mask for: (1) all occ area; (2) only moving object occ area; (3) only out-plane occ area.
            '''
            self.occ_type_ls = ['for_back_check', 'forward_warp']
            assert occ_type in self.occ_type_ls
            assert obj_out_all in ['obj', 'out', 'all']
            self.occ_type = occ_type
            self.occ_alpha_1 = occ_alpha_1
            self.occ_alpha_2 = occ_alpha_2
            self.sum_abs_or_squar = True  # found that false is not OK
            self.obj_out_all = obj_out_all

        def __call__(self, flow_f, flow_b, scale=1):
            # 输入进来是可使用的光流
            if self.obj_out_all == 'all':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                return occ_1, occ_2
            elif self.obj_out_all == 'obj':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                elif self.occ_type == 'for_back_check&forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
                obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
                return obj_occ_fw, obj_occ_bw
            elif self.obj_out_all == 'out':
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                return out_occ_fw, out_occ_bw
            else:
                raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

        def _forward_backward_occ_check(self, flow_fw, flow_bw, scale=1):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2 / scale
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            return occ_fw.float(), occ_bw.float()

        def forward_backward_occ_check(self, flow_fw, flow_bw, alpha1, alpha2, obj_out_all='obj'):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = alpha1 * mag_sq + alpha2
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            occ_fw = occ_fw.float()
            occ_bw = occ_bw.float()
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            if obj_out_all == 'obj':
                out_occ_fw = self.torch_outgoing_occ_check(flow_fw)
                out_occ_bw = self.torch_outgoing_occ_check(flow_bw)
                occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_fw, out_occ=out_occ_fw)
                occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_bw, out_occ=out_occ_bw)
            return occ_fw, occ_bw

        def _forward_warp_occ_check(self, flow_bw):  # TODO
            return 0

        @classmethod
        def torch_outgoing_occ_check(cls, flow):

            B, C, H, W = flow.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            flow_x, flow_y = torch.split(flow, 1, 1)
            if flow.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            # tools.check_tensor(flow_x, 'flow_x')
            # tools.check_tensor(flow_y, 'flow_y')
            # tools.check_tensor(xx, 'xx')
            # tools.check_tensor(yy, 'yy')
            pos_x = xx + flow_x
            pos_y = yy + flow_y
            # tools.check_tensor(pos_x, 'pos_x')
            # tools.check_tensor(pos_y, 'pos_y')
            # print(' ')
            # check mask
            outgoing_mask = torch.ones_like(pos_x)
            outgoing_mask[pos_x > W - 1] = 0
            outgoing_mask[pos_x < 0] = 0
            outgoing_mask[pos_y > H - 1] = 0
            outgoing_mask[pos_y < 0] = 0
            return outgoing_mask.float()

        @classmethod
        def torch_get_obj_occ_check(cls, occ_mask, out_occ):
            outgoing_mask = torch.zeros_like(occ_mask)
            if occ_mask.is_cuda:
                outgoing_mask = outgoing_mask.cuda()
            outgoing_mask[occ_mask == 1] = 1
            outgoing_mask[out_occ == 0] = 1
            return outgoing_mask

    class lib_to_show_flow():

        @classmethod
        def point_vec(cls, img, flow, valid=None):
            meshgrid = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
            dispimg = cv2.resize(img, None, fx=4, fy=4)
            colorflow = tools.flow_to_image(flow).astype(int)
            if valid is None:
                valid = np.ones((img.shape[0], img.shape[1]), dtype=flow.dtype)
            for i in range(img.shape[1]):  # x
                for j in range(img.shape[0]):  # y
                    # if flow[j, i, 2] != 1: continue
                    if valid[j, i] != 1: continue
                    if j % 10 != 0 or i % 10 != 0: continue
                    xend = int((meshgrid[0][j, i] + flow[j, i, 0]) * 4)
                    yend = int((meshgrid[1][j, i] + flow[j, i, 1]) * 4)
                    leng = np.linalg.norm(flow[j, i, :2])
                    if leng < 1: continue
                    dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j, i] * 4, meshgrid[1][j, i] * 4), (xend, yend),
                                              (int(colorflow[j, i, 0]), int(colorflow[j, i, 1]), int(colorflow[j, i, 2])), 3,
                                              tipLength=8 / leng, line_type=cv2.LINE_AA)
            return dispimg

        @classmethod
        def flow_error_image_np(cls, flow_pred, flow_gt, mask_occ, mask_noc=None, log_colors=True):
            """Visualize the error between two flows as 3-channel color image.
            Adapted from the KITTI C++ devkit.
            Args:
                flow_pred: prediction flow of shape [ height, width, 2].
                flow_gt: ground truth
                mask_occ: flow validity mask of shape [num_batch, height, width, 1].
                    Equals 1 at (occluded and non-occluded) valid pixels.
                mask_noc: Is 1 only at valid pixels which are not occluded.
            """
            # mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
            mask_noc = np.ones(mask_occ.shape) if mask_noc is None else mask_noc
            diff_sq = (flow_pred - flow_gt) ** 2
            # diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keep_dims=True))
            diff = np.sqrt(np.sum(diff_sq, axis=2, keepdims=True))
            if log_colors:
                height, width, _ = flow_pred.shape
                # num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
                colormap = [
                    [0, 0.0625, 49, 54, 149],
                    [0.0625, 0.125, 69, 117, 180],
                    [0.125, 0.25, 116, 173, 209],
                    [0.25, 0.5, 171, 217, 233],
                    [0.5, 1, 224, 243, 248],
                    [1, 2, 254, 224, 144],
                    [2, 4, 253, 174, 97],
                    [4, 8, 244, 109, 67],
                    [8, 16, 215, 48, 39],
                    [16, 1000000000.0, 165, 0, 38]]
                colormap = np.asarray(colormap, dtype=np.float32)
                colormap[:, 2:5] = colormap[:, 2:5] / 255
                # mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keep_dims=True))
                tempp = np.square(flow_gt)
                # temp = np.sum(tempp, axis=2, keep_dims=True)
                # mag = np.sqrt(temp)
                mag = np.sqrt(np.sum(tempp, axis=2, keepdims=True))
                # error = tf.minimum(diff / 3, 20 * diff / mag)
                error = np.minimum(diff / 3, 20 * diff / (mag + 1e-7))
                im = np.zeros([height, width, 3])
                for i in range(colormap.shape[0]):
                    colors = colormap[i, :]
                    cond = np.logical_and(np.greater_equal(error, colors[0]), np.less(error, colors[1]))
                    # temp=np.tile(cond, [1, 1, 3])
                    im = np.where(np.tile(cond, [1, 1, 3]), np.ones([height, width, 1]) * colors[2:5], im)
                # temp=np.cast(mask_noc, np.bool)
                # im = np.where(np.tile(np.cast(mask_noc, np.bool), [1, 1, 3]), im, im * 0.5)
                im = np.where(np.tile(mask_noc == 1, [1, 1, 3]), im, im * 0.5)
                im = im * mask_occ
            else:
                error = (np.minimum(diff, 5) / 5) * mask_occ
                im_r = error  # errors in occluded areas will be red
                im_g = error * mask_noc
                im_b = error * mask_noc
                im = np.concatenate([im_r, im_g, im_b], axis=2)
                # im = np.concatenate(axis=2, values=[im_r, im_g, im_b])
            return im[:, :, ::-1]

    class SP_transform():
        @classmethod
        def denormalize_coords(cls, xx, yy, width, height):
            """ scale indices from [-1, 1] to [0, width/height] """
            xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
            yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
            return xx, yy

        @classmethod
        def normalize_coords(cls, xx, yy, width, height):
            """ scale indices from [0, width/height] to [-1, 1] """
            xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
            yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
            return xx, yy

        @classmethod
        def apply_transform_to_params(cls, theta0, theta_transform):
            a1 = theta0[:, 0]
            a2 = theta0[:, 1]
            a3 = theta0[:, 2]
            a4 = theta0[:, 3]
            a5 = theta0[:, 4]
            a6 = theta0[:, 5]
            #
            b1 = theta_transform[:, 0]
            b2 = theta_transform[:, 1]
            b3 = theta_transform[:, 2]
            b4 = theta_transform[:, 3]
            b5 = theta_transform[:, 4]
            b6 = theta_transform[:, 5]
            #
            c1 = a1 * b1 + a4 * b2
            c2 = a2 * b1 + a5 * b2
            c3 = b3 + a3 * b1 + a6 * b2
            c4 = a1 * b4 + a4 * b5
            c5 = a2 * b4 + a5 * b5
            c6 = b6 + a3 * b4 + a6 * b5
            #
            new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
            return new_theta

        class _IdentityParams(nn.Module):
            def __init__(self):
                super(tools.SP_transform._IdentityParams, self).__init__()
                self._batch_size = 0
                self.register_buffer("_o", torch.FloatTensor())
                self.register_buffer("_i", torch.FloatTensor())

            def _update(self, batch_size):
                torch.zeros([batch_size, 1], out=self._o)
                torch.ones([batch_size, 1], out=self._i)
                return torch.cat([self._i, self._o, self._o, self._o, self._i, self._o], dim=1)

            def forward(self, batch_size):
                if self._batch_size != batch_size:
                    self._identity_params = self._update(batch_size)
                    self._batch_size = batch_size
                return self._identity_params

        class RandomMirror(nn.Module):
            def __init__(self, vertical=True, p=0.5):
                super(tools.SP_transform.RandomMirror, self).__init__()
                self._batch_size = 0
                self._p = p
                self._vertical = vertical
                self.register_buffer("_mirror_probs", torch.FloatTensor())

            def update_probs(self, batch_size):
                torch.ones([batch_size, 1], out=self._mirror_probs)
                self._mirror_probs *= self._p

            def forward(self, theta_list):
                batch_size = theta_list[0].size(0)
                if batch_size != self._batch_size:
                    self.update_probs(batch_size)
                    self._batch_size = batch_size

                # apply random sign to a1 a2 a3 (these are the guys responsible for x)
                sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                i = torch.ones_like(sign)
                horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
                theta_list = [theta * horizontal_mirror for theta in theta_list]

                # apply random sign to a4 a5 a6 (these are the guys responsible for y)
                if self._vertical:
                    sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                    vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
                    theta_list = [theta * vertical_mirror for theta in theta_list]

                return theta_list

        class RandomAffineFlow(nn.Module):
            def __init__(self, cfg, addnoise=True):
                super(tools.SP_transform.RandomAffineFlow, self).__init__()
                self.cfg = cfg
                self._interp2 = tools.Interpolation.Interp2(clamp=False)
                self._flow_interp2 = tools.Interpolation.Interp2(clamp=False)
                self._meshgrid = tools.Interpolation.Meshgrid()
                self._identity = tools.SP_transform._IdentityParams()
                self._random_mirror = tools.SP_transform.RandomMirror(cfg.vflip) if cfg.hflip else tools.SP_transform.RandomMirror(p=1)
                self._addnoise = addnoise

                self.register_buffer("_noise1", torch.FloatTensor())
                self.register_buffer("_noise2", torch.FloatTensor())
                self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
                self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
                self.register_buffer("_x", torch.IntTensor(1))
                self.register_buffer("_y", torch.IntTensor(1))

            def inverse_transform_coords(self, width, height, thetas, offset_x=None,
                                         offset_y=None):
                xx, yy = self._meshgrid(width=width, height=height)

                xx = torch.unsqueeze(xx, dim=0).float()
                yy = torch.unsqueeze(yy, dim=0).float()

                if offset_x is not None:
                    xx = xx + offset_x
                if offset_y is not None:
                    yy = yy + offset_y

                a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
                a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
                a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
                a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
                a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
                a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

                xx, yy = tools.SP_transform.normalize_coords(xx, yy, width=width, height=height)
                xq = a1 * xx + a2 * yy + a3
                yq = a4 * xx + a5 * yy + a6
                xq, yq = tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def transform_coords(self, width, height, thetas):
                xx1, yy1 = self._meshgrid(width=width, height=height)
                xx, yy = tools.SP_transform.normalize_coords(xx1, yy1, width=width, height=height)

                def _unsqueeze12(u):
                    return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

                a1 = _unsqueeze12(thetas[:, 0])
                a2 = _unsqueeze12(thetas[:, 1])
                a3 = _unsqueeze12(thetas[:, 2])
                a4 = _unsqueeze12(thetas[:, 3])
                a5 = _unsqueeze12(thetas[:, 4])
                a6 = _unsqueeze12(thetas[:, 5])
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = xx - a3
                yhat = yy - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat

                xq, yq = tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def find_invalid(self, width, height, thetas):
                x = self._xbounds
                y = self._ybounds
                #
                a1 = torch.unsqueeze(thetas[:, 0], dim=1)
                a2 = torch.unsqueeze(thetas[:, 1], dim=1)
                a3 = torch.unsqueeze(thetas[:, 2], dim=1)
                a4 = torch.unsqueeze(thetas[:, 3], dim=1)
                a5 = torch.unsqueeze(thetas[:, 4], dim=1)
                a6 = torch.unsqueeze(thetas[:, 5], dim=1)
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = x - a3
                yhat = y - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat
                xq, yq = tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                #
                invalid = (
                                  (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                          ).sum(dim=1, keepdim=True) > 0

                return invalid

            def apply_random_transforms_to_params(self,
                                                  theta0,
                                                  max_translate,
                                                  min_zoom, max_zoom,
                                                  min_squeeze, max_squeeze,
                                                  min_rotate, max_rotate,
                                                  validate_size=None):
                max_translate *= 0.5
                batch_size = theta0.size(0)
                height, width = validate_size

                # collect valid params here
                thetas = torch.zeros_like(theta0)

                zoom = theta0.new(batch_size, 1).zero_()
                squeeze = torch.zeros_like(zoom)
                tx = torch.zeros_like(zoom)
                ty = torch.zeros_like(zoom)
                phi = torch.zeros_like(zoom)
                invalid = torch.ones_like(zoom).byte()

                while invalid.sum() > 0:
                    # random sampling
                    zoom.uniform_(min_zoom, max_zoom)
                    squeeze.uniform_(min_squeeze, max_squeeze)
                    tx.uniform_(-max_translate, max_translate)
                    ty.uniform_(-max_translate, max_translate)
                    phi.uniform_(-min_rotate, max_rotate)

                    # construct affine parameters
                    sx = zoom * squeeze
                    sy = zoom / squeeze
                    sin_phi = torch.sin(phi)
                    cos_phi = torch.cos(phi)
                    b1 = cos_phi * sx
                    b2 = sin_phi * sy
                    b3 = tx
                    b4 = - sin_phi * sx
                    b5 = cos_phi * sy
                    b6 = ty

                    theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
                    theta_try = tools.SP_transform.apply_transform_to_params(theta0, theta_transform)
                    thetas = invalid.float() * theta_try + (1 - invalid).float() * thetas

                    # compute new invalid ones
                    invalid = self.find_invalid(width=width, height=height, thetas=thetas)

                # here we should have good thetas within borders
                return thetas

            def transform_image(self, images, thetas):
                batch_size, channels, height, width = images.size()
                xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
                transformed = self._interp2(images, xq, yq)
                return transformed

            def transform_flow(self, flow, theta1, theta2):
                batch_size, channels, height, width = flow.size()
                u = flow[:, 0, :, :]
                v = flow[:, 1, :, :]

                # inverse transform coords
                x0, y0 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta1)

                x1, y1 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

                # subtract and create new flow
                u = x1 - x0
                v = y1 - y0
                new_flow = torch.stack([u, v], dim=1)

                # transform coords
                xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

                # interp2
                transformed = self._flow_interp2(new_flow, xq, yq)
                return transformed

            def forward(self, data):
                # 01234 flow 12 21 23 32
                imgs = data['imgs']
                flows_f = data['flows_f']
                masks_f = data['masks_f']

                batch_size, _, height, width = imgs[0].size()

                # identity = no transform
                theta0 = self._identity(batch_size)

                # global transform
                theta_list = [self.apply_random_transforms_to_params(
                    theta0,
                    max_translate=self.cfg.trans[0],
                    min_zoom=self.cfg.zoom[0], max_zoom=self.cfg.zoom[1],
                    min_squeeze=self.cfg.squeeze[0], max_squeeze=self.cfg.squeeze[1],
                    min_rotate=self.cfg.rotate[0], max_rotate=self.cfg.rotate[1],
                    validate_size=[height, width])
                ]

                # relative transform
                for i in range(len(imgs) - 1):
                    theta_list.append(
                        self.apply_random_transforms_to_params(
                            theta_list[-1],
                            max_translate=self.cfg.trans[1],
                            min_zoom=self.cfg.zoom[2], max_zoom=self.cfg.zoom[3],
                            min_squeeze=self.cfg.squeeze[2], max_squeeze=self.cfg.squeeze[3],
                            min_rotate=-self.cfg.rotate[2], max_rotate=self.cfg.rotate[2],
                            validate_size=[height, width])
                    )

                # random flip images
                theta_list = self._random_mirror(theta_list)

                # 01234
                imgs = [self.transform_image(im, theta) for im, theta in zip(imgs, theta_list)]

                if len(imgs) > 2:
                    theta_list = theta_list[1:-1]
                # 12 23
                flows_f = [self.transform_flow(flo, theta1, theta2) for flo, theta1, theta2 in
                           zip(flows_f, theta_list[:-1], theta_list[1:])]

                masks_f = [self.transform_image(mask, theta) for mask, theta in
                           zip(masks_f, theta_list)]

                if self._addnoise:
                    '''
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.040912468
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.03847942
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4187718 mean -0.011021424
                    '''
                    stddev = np.random.uniform(0.0, 0.04)
                    for im in imgs:
                        noise = torch.zeros_like(im)
                        noise.normal_(std=stddev)
                        im.add_(noise)
                        im.clamp_(-1.0, 1.0)

                data['imgs'] = imgs
                data['flows_f'] = flows_f
                data['masks_f'] = masks_f
                return data

        @classmethod
        def demo(cls):
            import pickle
            import cv2
            im0 = cv2.imread("/data/luokunming/Optical_Flow_all/projects/Forward-Warp-master/test/im0.png")[np.newaxis, :, :, :]
            im1 = cv2.imread("/data/luokunming/Optical_Flow_all/projects/Forward-Warp-master/test/im1.png")[np.newaxis, :, :, :]
            mask = np.ones((1, 1, im1.shape[1], im1.shape[2]))
            with open("/data/luokunming/Optical_Flow_all/projects/Forward-Warp-master/test/flow.pkl", "rb+") as f:
                flow = pickle.load(f)
            im0 = torch.FloatTensor(im0).permute(0, 3, 1, 2)
            im1 = torch.FloatTensor(im1).permute(0, 3, 1, 2)
            mask = torch.FloatTensor(mask)  # .permute(0, 3, 1, 2)
            flow = torch.FloatTensor(flow)
            flow = flow.permute(0, 3, 1, 2)
            tools.check_tensor(im0, 'im0')
            tools.check_tensor(im1, 'im1')
            tools.check_tensor(flow, 'flow')
            tools.check_tensor(mask, 'mask')

            class config():
                def __init__(self):
                    self.add_noise = False
                    self.hflip = True
                    self.rotate = [-0.01, 0.01, -0.01, 0.01]
                    self.squeeze = [1.0, 1.0, 1.0, 1.0]
                    self.trans = [0.04, 0.005]
                    self.vflip = True
                    self.zoom = [1.0, 1.4, 0.99, 1.01]

            model = tools.SP_transform.RandomAffineFlow(config(), addnoise=False)
            input = {'imgs': [im0 / 255, im1 / 255], 'flows_f': [flow], 'masks_f': [mask]}
            data = model(input)
            imgs0, imgs1 = data['imgs']
            flows_f = data['flows_f'][0]
            # show
            tools.check_tensor(imgs0, 'imgs0 out')

            def process_image(tens, ind=0):
                tens_, = tools.tensor_gpu(tens, check_on=False)
                im = tens_[ind, :, :, :]
                im = np.transpose(im, (1, 2, 0))
                return im

            im0_ori = tools.Show_GIF.im_norm(process_image(im0))
            im0_res = tools.Show_GIF.im_norm(process_image(imgs0))
            flow_ori = tools.flow_to_image(process_image(flow))
            flow_res = tools.flow_to_image(process_image(flows_f))
            tools.cv2_show_dict(im0_ori=im0_ori, im0_res=im0_res, flow_ori=flow_ori, flow_res=flow_res)

    class txt_read_write():
        @classmethod
        def read(cls, path):
            with open(path, "r") as f:
                data = f.readlines()
            return data

        @classmethod
        def write(cls, path, data_ls):
            file_write_obj = open(path, 'a')
            for i in data_ls:
                file_write_obj.writelines(i)
            file_write_obj.close()

        @classmethod
        def demo(cls):
            txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\frame_0001_match.txt'
            data = tools.txt_read_write.read(txt_path)
            print(' ')
            write_txt_path = txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\temp.txt'
            tools.txt_read_write.write(write_txt_path, data[:10])
            print(' ')

    class KITTI_png_flow_read_write():
        def demo(self):
            def write_kitti_png_file(flow_fn, flow_data, mask_data=None):
                '''
                :param flow_fn: save file path, .png file
                :param flow_data:  [H,W,2]
                :param mask_data: can be occlusion mask (0-1 mask) or None
                :return:
                '''
                flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                                    dtype=np.uint16)
                if mask_data is None:
                    mask_data = np.ones([flow_data.shape[0], flow_data.shape[1]], dtype=np.uint16)
                flow_img[:, :, 2] = (flow_data[:, :, 0] * 64.0 + 2 ** 15).astype(np.uint16)
                flow_img[:, :, 1] = (flow_data[:, :, 1] * 64.0 + 2 ** 15).astype(np.uint16)
                flow_img[:, :, 0] = mask_data[:, :]
                cv2.imwrite(flow_fn, flow_img)

            def read_png_flow(fpath):
                """
                Read KITTI optical flow, returns u,v,valid mask

                """

                R = png.Reader(fpath)
                width, height, data, _ = R.asDirect()
                # This only worked with python2.
                # I = np.array(map(lambda x:x,data)).reshape((height,width,3))
                gt = np.array([x for x in data]).reshape((height, width, 3))
                flow = gt[:, :, 0:2]
                flow = (flow.astype('float64') - 2 ** 15) / 64.0
                flow = flow.astype(np.float)
                mask = gt[:, :, 2:3]
                mask = np.uint8(mask)
                flow = np.transpose(flow, [2, 0, 1])
                mask = np.transpose(mask, [2, 0, 1])

                return flow, mask

    @classmethod
    def MSE(cls, img1, img2):
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("L1_.jpg",img1gray-img2gray)
        rows, cols = img1gray.shape[:2]
        loss = 0.0
        pixel_nums = 0
        for row in range(30, rows - 30):
            for col in range(60, cols - 60):
                if img1gray[row][col] == 0 or img2gray[row][col] == 0:
                    continue
                else:
                    pixel_nums += 1
                    loss += np.square(np.abs(img1gray[row][col] - img2gray[row][col]))

        loss /= pixel_nums

        return loss

    @classmethod
    def torch_warp_mask(cls, x, flo):
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
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output, mask

    @classmethod
    def torch_warp(cls, x, flo):
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
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('conv') != -1:
            # torch.nn.init.xavier_normal(m.weight)
            torch.nn.init.kaiming_normal(m.weight)

            torch.nn.init.constant(m.bias, 0)

    @classmethod
    def warp_cv2(cls, img_prev, flow):
        # calculate mat
        w = int(img_prev.shape[1])
        h = int(img_prev.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
        return new_frame

    @classmethod
    def flow_to_image(cls, flow, display=False):
        """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

        def compute_color(u, v):
            def make_color_wheel():
                """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
                RY = 15
                YG = 6
                GC = 4
                CB = 11
                BM = 13
                MR = 6

                ncols = RY + YG + GC + CB + BM + MR

                colorwheel = np.zeros([ncols, 3])

                col = 0

                # RY
                colorwheel[0:RY, 0] = 255
                colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
                col += RY

                # YG
                colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
                colorwheel[col:col + YG, 1] = 255
                col += YG

                # GC
                colorwheel[col:col + GC, 1] = 255
                colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
                col += GC

                # CB
                colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
                colorwheel[col:col + CB, 2] = 255
                col += CB

                # BM
                colorwheel[col:col + BM, 2] = 255
                colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
                col += + BM

                # MR
                colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
                colorwheel[col:col + MR, 0] = 255

                return colorwheel

            """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
            [h, w] = u.shape
            img = np.zeros([h, w, 3])
            nanIdx = np.isnan(u) | np.isnan(v)
            u[nanIdx] = 0
            v[nanIdx] = 0

            colorwheel = make_color_wheel()
            ncols = np.size(colorwheel, 0)

            rad = np.sqrt(u ** 2 + v ** 2)

            a = np.arctan2(-v, -u) / np.pi

            fk = (a + 1) / 2 * (ncols - 1) + 1

            k0 = np.floor(fk).astype(int)

            k1 = k0 + 1
            k1[k1 == ncols + 1] = 1
            f = fk - k0

            for i in range(0, np.size(colorwheel, 1)):
                tmp = colorwheel[:, i]
                col0 = tmp[k0 - 1] / 255
                col1 = tmp[k1 - 1] / 255
                col = (1 - f) * col0 + f * col1

                idx = rad <= 1
                col[idx] = 1 - rad[idx] * (1 - col[idx])
                notidx = np.logical_not(idx)

                col[notidx] *= 0.75
                img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

            return img

        UNKNOWN_FLOW_THRESH = 1e7
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        if display:
            print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    @classmethod
    def write_flow_png(cls, filename, uv, v=None, mask=None):

        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert (u.shape == v.shape)

        height_img, width_img = u.shape
        if mask is None:
            valid_mask = np.ones([height_img, width_img], dtype=np.uint16)
        else:
            valid_mask = mask

        flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
        flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)

        output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

        with open(filename, 'wb') as f:
            # writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
            # temp = np.reshape(output, (-1, width_img * 3))
            # writer.write(f, temp)

            png_writer = png.Writer(width=width_img, height=height_img, bitdepth=16, compression=3, greyscale=False)
            # png_writer.write_array(f, output)
            temp = np.reshape(output, (-1, width_img * 3))
            png_writer.write(f, temp)

    @classmethod
    def write_kitti_png_file(cls, flow_fn, flow_data, mask_data=None):
        flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                            dtype=np.uint16)
        if mask_data is None:
            mask_data = np.ones([flow_data.shape[0], flow_data.shape[1]], dtype=np.uint16)
        flow_img[:, :, 2] = (flow_data[:, :, 0] * 64.0 + 2 ** 15).astype(np.uint16)
        flow_img[:, :, 1] = (flow_data[:, :, 1] * 64.0 + 2 ** 15).astype(np.uint16)
        flow_img[:, :, 0] = mask_data[:, :]
        cv2.imwrite(flow_fn, flow_img)

    @classmethod
    def WriteKittiPngFile(cls, path, uv, mask=None):
        """ Write 16-bit .PNG file as specified by KITTI-2015 (flow).
        u, v are lists of float values
        mask is a list of floats, denoting the *valid* pixels.
        """
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
        height, width = u.shape
        if mask is None:
            valid_mask = np.ones([height, width])
        else:
            valid_mask = mask
        data = array.array('H', [0]) * width * height * 3

        # for i, (u_, v_, mask_) in enumerate(zip(u, v, mask)):
        data[0] = int(u * 64.0 + 2 ** 15)
        data[1] = int(v * 64.0 + 2 ** 15)
        data[2] = int(valid_mask)

        # if mask_ > 0:
        #     print(data[3*i], data[3*i+1],data[3*i+2])

        with open(path, 'wb') as png_file:
            png_writer = png.Writer(width=width, height=height, bitdepth=16, compression=3, greyscale=False)
            png_writer.write_array(png_file, data)

    @classmethod
    def write_flow(cls, flow, filename):
        """
        write optical flow in Middlebury .flo format
        :param flow: optical flow map
        :param filename: optical flow file path to be saved
        :return: None
        """
        f = open(filename, 'wb')
        magic = np.array([202021.25], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
        f.close()

    @classmethod
    def read_flow(cls, filename):
        """
        read optical flow from Middlebury .flo file
        :param filename: name of the flow file
        :return: optical flow data in matrix
        """
        f = open(filename, 'rb')
        try:
            magic = np.fromfile(f, np.float32, count=1)[0]  # For Python3.x
        except:
            magic = np.fromfile(f, np.float32, count=1)  # For Python2.x
        data2d = None

        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print("Reading %d x %d flo file" % (h, w))
            data2d = np.fromfile(f, np.float32, count=2 * w * h)
            # reshape data into 3D array (columns, rows, channels)
            data2d = np.resize(data2d, (h[0], w[0], 2))
        f.close()
        return data2d

    @classmethod
    def read_flo(cls, filename):
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                data = np.fromfile(f, np.float32, count=int(2 * w * h))
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (h[0], w[0], 2))
                return data2D

    @classmethod
    def write_flo(cls, flow, filename):
        """
        write optical flow in Middlebury .flo format
        :param flow: optical flow map
        :param filename: optical flow file path to be saved
        :return: None
        """
        f = open(filename, 'wb')
        magic = np.array([202021.25], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
        f.close()

    @classmethod
    def tensor_gpu(cls, *args, check_on=True, gpu_opt=None, non_blocking=True):
        def check_on_gpu(tensor_):
            if type(gpu_opt) == int:
                tensor_g = tensor_.cuda(gpu_opt, non_blocking=non_blocking)
            else:
                tensor_g = tensor_.cuda()
            return tensor_g

        def check_off_gpu(tensor_):
            if tensor_.is_cuda:
                tensor_c = tensor_.cpu()
            else:
                tensor_c = tensor_
            tensor_c = tensor_c.detach().numpy()
            # tensor_c = cv2.normalize(tensor_c.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return tensor_c

        if torch.cuda.is_available():
            if check_on:
                data_ls = [check_on_gpu(a) for a in args]
            else:
                data_ls = [check_off_gpu(a) for a in args]
        else:
            if check_on:
                data_ls = args
            else:
                # data_ls = args
                data_ls = [a.detach().numpy() for a in args]
                # data_ls = [cv2.normalize(a.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for a in args]
                # data_ls = args
        return data_ls

    @classmethod
    def tryremove(cls, name, file=False):
        try:
            if file:
                os.remove(name)
            else:
                rmtree(name)
        except OSError:
            pass

    @classmethod
    def check_tensor(cls, data, name, print_data=False, print_in_txt=None):
        if data.is_cuda:
            temp = data.detach().cpu().numpy()
        else:
            temp = data.detach().numpy()
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s,%s' % (name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp), ' min:%.2f' % np.min(temp),
                                                    ' mean:%.2f' % np.mean(temp), ' sum:%.2f' % np.sum(temp), data.device)
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str

    @classmethod
    def extract_zip(cls, zip_path, extract_dir):
        print('unzip file: %s' % zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    @classmethod
    def clear(cls):
        os.system("clear")  # 清屏

    @classmethod
    def random_flag(cls, threshold_0_1=0.5):
        a = random.random() < threshold_0_1
        return a

    @classmethod
    def compute_model_size(cls, model, *args):
        from thop import profile
        flops, params = profile(model, inputs=args, verbose=False)
        print('flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def im_norm(cls, img):
        eps = 1e-5
        a = np.max(img)
        b = np.min(img)
        img = (img - b) / (a - b)
        img = img * 255
        img = img.astype('uint8')
        return img
