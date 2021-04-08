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

''' scripts for training # TODO '''


