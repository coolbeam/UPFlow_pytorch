import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.tools import tools


class Corr_pyTorch(tools.abstract_model):
    '''
    my implementation of correlation layer using pytorch
    note that the Ispeed is much slower than cuda version
    '''

    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        bz, cn, hei, wid = in1.shape
        # print(self.kernel_size, self.pad_size, self.stride1)
        f1 = F.unfold(in1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride1)
        f2 = F.unfold(in2, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride2)  # 在这一步抽取完了kernel以后做warping插值岂不美哉？
        # tools.check_tensor(in2, 'in2')
        # tools.check_tensor(f2, 'f2')
        searching_kernel_size = f2.shape[1]
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, hei, wid))
        f2_ = torch.reshape(f2_, (bz * searching_kernel_size, hei, wid)).unsqueeze(1)
        # tools.check_tensor(f2_, 'f2_reshape')
        f2 = F.unfold(f2_, kernel_size=(hei, wid), padding=self.pad_size, stride=self.stride2)
        # tools.check_tensor(f2, 'f2_reunfold')
        _, kernel_number, window_number = f2.shape
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, kernel_number, window_number))
        f2_2 = torch.transpose(f2_, dim0=1, dim1=3).transpose(2, 3)
        f1_2 = f1.unsqueeze(1)
        # tools.check_tensor(f1_2, 'f1_2_reshape')
        # tools.check_tensor(f2_2, 'f2_2_reshape')
        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = torch.reshape(res, (bz, window_number, hei, wid))
        # tools.check_tensor(res, 'res')
        return res
