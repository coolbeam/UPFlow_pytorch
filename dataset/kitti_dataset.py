#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from utils.tools import tools
import random
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import tensorflow as tf
import warnings  # ignore warnings
import zipfile
from glob import glob
from torchvision import transforms as vision_transforms
import imageio
import png

'''
Here tensorflow is not necessary, it is needed in my early implementation from UnFlow and DDFlow.
'''

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}, ignore tensorflow log information
'''
    you can download kitti mv dataset from:
        kitti 2012 multi view: http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow
            (17GB) data_stereo_flow_multiview.zip, here I save it in .../KITTI_data_mv_stereo_flow_2012/data_stereo_flow_multiview.zip
        kitti 2015 multi view: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
            (14GB) data_scene_flow_multiview.zip, here I save it in .../KITTI_data_mv_stereo_flow_2015/data_scene_flow_multiview.zip
'''
mv_data_dir = '/data/Optical_Flow_all/datasets/KITTI_data/KITTI_data_mv'  # TODO important path
'''  
    KITTI flow data:
        kitti 2012 data: data_stereo_flow.zip (2.0GB) 
        kitti 2015 data: data_scene_flow.zip  (1.7GB)
    here you should unzip them
'''
kitti_flow_dir = '/data/Optical_Flow_all/datasets/KITTI_data'  # TODO important path


class img_func():

    @classmethod
    def get_process_img(cls, img_name, normalize=True, if_horizontal_flip=False):
        def _normalize_image(image):
            mean = [104.920005, 110.1753, 114.785955]
            stddev = 1 / 0.0039216
            unflow_im = (image - mean) / stddev
            # std__ = [69.85, 68.81, 72.45]
            # mean_ = [118.93, 113.97, 102.60]
            # unflow_pytorch = unflow_im * std__ + mean_
            # # check(img_rgb, 'img_rgb')
            # unflow_pytorch = unflow_pytorch * (1.0 / 255.0)
            return unflow_im

        # img = cv2.imread(img_name)
        data = tf.io.read_file(img_name)
        img = tf.image.decode_image(data).numpy()
        if if_horizontal_flip:
            img = np.flip(img, 1)
        if normalize:
            img = _normalize_image(img)
        # img = img / 255.0
        img = np.transpose(img, [2, 0, 1])
        return img

    @classmethod
    def get_process_img_only_img(cls, img, normalize=True, if_horizontal_flip=False):
        def _normalize_image(image):
            mean = [104.920005, 110.1753, 114.785955]
            stddev = 1 / 0.0039216
            unflow_im = (image - mean) / stddev
            # std__ = [69.85, 68.81, 72.45]
            # mean_ = [118.93, 113.97, 102.60]
            # unflow_pytorch = unflow_im * std__ + mean_
            # # check(img_rgb, 'img_rgb')
            # unflow_pytorch = unflow_pytorch * (1.0 / 255.0)
            return unflow_im

        # img = cv2.imread(img_name)
        # img = imageio.imread(img_name)
        if if_horizontal_flip:
            img = np.flip(img, 1)
        if normalize:
            img = _normalize_image(img)
        # img = img / 255.0
        img = np.transpose(img, [2, 0, 1])
        return img

    @classmethod
    def frame_name_to_num(cls, name):
        stripped = name.split('.')[0].lstrip('0')
        if stripped == '':
            return 0
        return int(stripped)

    @classmethod
    def np_2_tensor(cls, *args):
        def func(a):
            b = torch.from_numpy(a)
            b = b.float()
            return b

        return [func(a) for a in args]

    @classmethod
    def read_flow(cls, filename):
        gt = cv2.imread(filename)
        flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask = gt[:, :, 2:3]
        flow = np.transpose(flow, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        return flow, mask

    @classmethod
    def read_flow_tf(cls, filename):  # need tensorflow 2.0.0, WRONG!!! use the function read_png_flow
        data = tf.io.read_file(filename)
        gt = tf.image.decode_png(data, channels=3, dtype=tf.uint16).numpy()
        # gt=cv2.imread(filename)
        flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        flow = flow.astype(np.float)
        mask = gt[:, :, 2:3]
        mask = np.uint8(mask)
        flow = np.transpose(flow, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        # print('mask: max', np.max(mask), 'min', np.min(mask))
        return flow, mask

    @classmethod
    def read_png_flow(cls, fpath):
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
    def censusTransform(cls, src_bytes, if_debug=False):
        def censusTransformSingleChannel(src_bytes):
            h, w = src_bytes.shape
            # Initialize output array
            census = np.zeros((h - 2, w - 2), dtype='uint8')
            # census1 = np.zeros((h, w), dtype='uint8')

            # centre pixels, which are offset by (1, 1)
            cp = src_bytes[1:h - 1, 1:w - 1]

            # offsets of non-central pixels
            offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

            # Do the pixel comparisons
            for u, v in offsets:
                census = (census << 1) | (src_bytes[v:v + h - 2, u:u + w - 2] >= cp)

            return census

        # chk num of channels. if 1 call as is, if 3 call it with each channel
        if len(src_bytes.shape) == 2:  # single channel
            census = censusTransformSingleChannel(np.lib.pad(src_bytes, 1, 'constant', constant_values=0))

        elif len(src_bytes.shape) == 3 and src_bytes.shape[2] == 3:
            temp = np.lib.pad(src_bytes[:, :, 0], 1, 'constant', constant_values=0)
            census_a = censusTransformSingleChannel(temp)
            if if_debug:
                cv2.imshow(mat=temp, winname='pad')
                print('temp', temp.shape, np.max(temp), np.min(temp))
                print('census_a', census_a.shape, np.max(census_a), np.min(census_a))
                cv2.imshow(mat=census_a, winname='census_a')
                cv2.waitKey()
            census_b = censusTransformSingleChannel(np.lib.pad(src_bytes[:, :, 1], 1, 'constant', constant_values=0))
            census_c = censusTransformSingleChannel(np.lib.pad(src_bytes[:, :, 2], 1, 'constant', constant_values=0))
            census = np.dstack((census_a, census_b, census_c))
        else:
            raise ValueError('wrong channel RGB ')

        return census


class kitti_train:
    @classmethod
    def mv_data_get_file_names(cls, if_test=False):
        file_names_save_path = os.path.join(mv_data_dir, 'kitti_mv_file_names.pkl')
        if os.path.isfile(file_names_save_path) and not if_test:
            data = tools.pickle_saver.load_picke(file_names_save_path)
            return data
        else:
            mv_2012_name = 'stereo_flow_2012'
            mv_2012_file_name = 'data_stereo_flow_multiview.zip'
            mv_2012_zip_file = os.path.join(mv_data_dir, mv_2012_name, mv_2012_file_name)
            mv_2012_dir = os.path.join(mv_data_dir, mv_2012_name, mv_2012_file_name[:-4])
            if os.path.isdir(mv_2012_dir):
                pass
            else:
                tools.extract_zip(mv_2012_zip_file, mv_2012_dir)

            mv_2015_name = 'stereo_flow_2015'
            mv_2015_file_name = 'data_scene_flow_multiview.zip'
            mv_2015_zip_file = os.path.join(mv_data_dir, mv_2015_name, mv_2015_file_name)
            mv_2015_dir = os.path.join(mv_data_dir, mv_2015_name, mv_2015_file_name[:-4])
            if os.path.isdir(mv_2015_dir):
                pass
            else:
                tools.extract_zip(mv_2015_zip_file, mv_2015_dir)

            def read_mv_data(d_path):
                def tf_read_img(im_path):
                    data_img = tf.io.read_file(im_path)
                    img_read = tf.image.decode_image(data_img).numpy()  # get image 1
                    return img_read

                sample_ls = []
                for sub_dir in ['testing', 'training']:
                    img_dir = os.path.join(d_path, sub_dir, 'image_2')
                    file_ls = os.listdir(img_dir)
                    file_ls.sort()
                    print(' ')
                    for ind in range(len(file_ls) - 1):
                        name = file_ls[ind]
                        nex_name = file_ls[ind + 1]
                        id_ = int(name[-6:-4])
                        id_nex = int(nex_name[-6:-4])
                        if id_ != id_nex - 1 or 12 >= id_ >= 9 or 12 >= id_nex >= 9:
                            pass
                        else:
                            file_path = os.path.join(img_dir, name)
                            file_path_nex = os.path.join(img_dir, nex_name)
                            # # test
                            if if_test:
                                # im1 = cv2.imread(file_path)
                                # im2 = cv2.imread(file_path_nex)
                                im1 = tf_read_img(file_path)[:, :, ::-1]
                                im2 = tf_read_img(file_path_nex)[:, :, ::-1]
                                cv2.imshow(name, im1)
                                k = cv2.waitKey()
                                c = 0
                                while k != ord('q'):
                                    c += 1
                                    if c % 2 == 0:
                                        cv2.imshow(name, im1)
                                    else:
                                        cv2.imshow(name, im2)
                                    k = cv2.waitKey()
                                cv2.destroyAllWindows()
                            sample_ls.append((file_path, file_path_nex))

                return sample_ls

            filenames = {}
            filenames['2012'] = read_mv_data(mv_2012_dir)
            filenames['2015'] = read_mv_data(mv_2015_dir)
            tools.pickle_saver.save_pickle(files=filenames, file_path=file_names_save_path)
            return filenames

    class kitti_data_with_start_point(Dataset):
        class config(tools.abstract_config):

            def __init__(self, **kwargs):
                self.crop_size = (256, 832)  # original size is (512,1152), we directly set as (256, 832) during training
                self.rho = 8
                self.swap_images = True
                self.normalize = True
                self.repeat = None  # if repeat the dataset in one epoch
                self.horizontal_flip_aug = True
                self.mv_type = None  # '2015' or '2012'
                self.update(kwargs)

            def __call__(self):
                return kitti_train.kitti_data_with_start_point(self)

        def __init__(self, conf: config):
            self.conf = conf
            if self.conf.mv_type in ['2015', '2012']:
                file_dict_ = kitti_train.mv_data_get_file_names()
                if self.conf.mv_type == '2015':
                    print('=' * 5)
                    print('use multi_view dataset 2015')
                    print('=' * 5)
                    filenames_extended = file_dict_['2015']
                elif self.conf.mv_type == '2012':
                    print('=' * 5)
                    print('use multi_view dataset 2012')
                    print('=' * 5)
                    filenames_extended = file_dict_['2012']
                else:
                    raise ValueError('wrong type mv dataset: %s' % self.conf.mv_type)
            else:
                raise ValueError('mv_type should be 2012 or 2015')
            # =====================================
            self.filenames_extended = filenames_extended
            self.N = len(self.filenames_extended)

        def __len__(self):
            if self.conf.repeat is None or self.conf.repeat <= 0:
                return len(self.filenames_extended)
            else:
                assert type(self.conf.repeat) == int
                return len(self.filenames_extended) * self.conf.repeat

        def __getitem__(self, index):
            im1, im2 = self.read_img(index=index)
            im1_crop, im2_crop, start = self.random_crop(im1, im2)
            im1, im2, im1_crop, im2_crop, start = img_func.np_2_tensor(im1, im2, im1_crop, im2_crop, start)
            return im1, im2, im1_crop, im2_crop, start

        def random_crop(self, im1, im2):
            height, width = im1.shape[1:]
            patch_size_h, patch_size_w = self.conf.crop_size
            x = np.random.randint(self.conf.rho, width - self.conf.rho - patch_size_w)
            # print(self.rho, height - self.rho - patch_size_h)
            y = np.random.randint(self.conf.rho, height - self.conf.rho - patch_size_h)
            start = np.array([x, y])
            start = np.expand_dims(np.expand_dims(start, 1), 2)
            img_1_patch = im1[:, y: y + patch_size_h, x: x + patch_size_w]
            img_2_patch = im2[:, y: y + patch_size_h, x: x + patch_size_w]
            return img_1_patch, img_2_patch, start

        def read_img(self, index):
            if self.conf.horizontal_flip_aug and random.random() < 0.5:
                if_horizontal_flip = True
            else:
                if_horizontal_flip = False
            im1_path, im2_path = self.filenames_extended[index % self.N]
            im1 = img_func.get_process_img(im1_path, normalize=self.conf.normalize, if_horizontal_flip=if_horizontal_flip)
            im2 = img_func.get_process_img(im2_path, normalize=self.conf.normalize, if_horizontal_flip=if_horizontal_flip)
            if self.conf.swap_images and tools.random_flag(0.5):
                return im2, im1
            else:
                return im1, im2

        @classmethod
        def demo(cls):
            def process(a):
                b = a.numpy()
                b = np.squeeze(b)
                b = np.transpose(b, (1, 2, 0))
                b = tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            data_conf = kitti_train.kitti_data_with_start_point.config()
            print('begin!!!' + '=' * 10)
            data_conf.crop_size = (256, 832)
            data_conf.rho = 8
            data_conf.repeat = None  # if repeat the dataset in one epoch
            data_conf.normalize = True
            data_conf.horizontal_flip_aug = False
            data_conf.mv_type = '2012'
            data_conf.get_name(print_now=True)
            data = data_conf()
            N = len(data)
            print('len: %5s' % N)
            for i in range(len(data)):
                if i > 5:
                    break
                im1, im2, im1_crop, im2_crop, start = data.__getitem__(i)
                tools.check_tensor(im1, 'im1')
                tools.check_tensor(im2, 'im2')
                tools.check_tensor(im1_crop, 'im1_crop')
                tools.check_tensor(im2_crop, 'im2_crop')
                tools.check_tensor(start, 'start')
                # im1, im2 = tools.func_decorator(process, im1, im2)
                # tools.cv2_show_dict(im1=im1, im2=im2)


class kitti_flow:
    class Evaluation_bench():

        def __init__(self, name, if_gpu=True, batch_size=1):
            assert if_gpu == True
            self.batch_size = batch_size
            assert name in ['2012_train', '2015_train', '2012_test', '2015_test']
            self.name = name
            self.dataset = kitti_flow.kitti_train(name=name)
            self.loader = tools.data_prefetcher(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
            # self.loader = DataLoader(dataset=self.dataset, batch_size=batc_size, num_workers=4, shuffle=False, drop_last=False)
            # self.loader = tools.data_prefetcher(self.loader)
            self.if_gpu = if_gpu
            self.timer = tools.time_clock()

        def __call__(self, test_model: tools.abs_test_model):
            def calculate(predflow, gt_flow, gt_mask):
                error_ = self.flow_error_avg(gt_flow, predflow, gt_mask)
                # error_avg_ = summarized_placeholder('AEE/' + name, key='eval_avg')
                outliers_ = self.outlier_pct(gt_flow, predflow, gt_mask)
                # outliers_avg = summarized_placeholder('outliers/' + name, key='eval_avg')
                # values_.extend([error_, outliers_])
                # averages_.extend([error_avg_, outliers_avg])
                return error_, outliers_

            if self.name in ['2012_test', '2015_test']:
                self.timer.start()
                index = -1
                # with torch.no_grad():
                for i in range(len(self.dataset)):
                    im1, im2, img_name = self.dataset.__getitem__(i)
                    im1 = torch.unsqueeze(im1, 0)
                    im2 = torch.unsqueeze(im2, 0)
                    if self.if_gpu:
                        im1, im2 = tools.tensor_gpu(im1, im2, check_on=True)
                    index += 1
                    # im1, im2 = batch
                    predflow = test_model.eval_forward(im1, im2, 0)
                    test_model.eval_save_result(img_name, predflow)
                self.timer.end()
                print('=' * 3 + ' test time %ss ' % self.timer.get_during() + '=' * 3)
                return
            all_pep_error_meter = tools.AverageMeter()
            f1_rate_meter = tools.AverageMeter()
            occ_pep_error_meter = tools.AverageMeter()
            noc_pep_error_meter = tools.AverageMeter()
            self.timer.start()
            index = -1
            batch = self.loader.next()
            # with torch.no_grad():
            while batch is not None:
                index += 1
                im1, im2, occ, occmask, noc, nocmask = batch
                num = im1.shape[0]
                predflow = test_model.eval_forward(im1, im2, occ, occmask, noc, nocmask)

                pep_error_all, f1_rate = calculate(predflow=predflow, gt_flow=occ, gt_mask=occmask)
                all_pep_error_meter.update(val=pep_error_all.item(), num=num)
                f1_rate_meter.update(val=f1_rate.item(), num=num)

                noc_pep_error_, _ = calculate(predflow=predflow, gt_flow=noc, gt_mask=nocmask)
                noc_pep_error_meter.update(val=noc_pep_error_.item(), num=num)

                occ_erea_mask = occmask - nocmask
                pep_error_occ, _ = calculate(predflow=predflow, gt_flow=occ, gt_mask=occ_erea_mask)
                occ_pep_error_meter.update(val=pep_error_occ.item(), num=num)
                save_name = 'all_%.2f f1_%.1f noc_%.2f occ_%.2f__%d' % (all_pep_error_meter.val, f1_rate_meter.val, noc_pep_error_meter.val, occ_pep_error_meter.val, index)
                test_model.eval_save_result(save_name, predflow, occmask=occmask)
                batch = self.loader.next()
            self.timer.end()
            print('=' * 3 + ' eval time %ss ' % self.timer.get_during() + '=' * 3)
            return all_pep_error_meter.avg, f1_rate_meter.avg, noc_pep_error_meter.avg, occ_pep_error_meter.avg

        @classmethod
        def flow_error_avg_tf(cls, flow_1, flow_2, mask):
            """Evaluates the average endpoint error between flow batches.tf batch is n h w c"""

            def euclidean(t):
                return tf.sqrt(tf.reduce_sum(t ** 2, [3], keepdims=True))

            diff = euclidean(flow_1 - flow_2) * mask
            error = tf.reduce_sum(diff) / tf.reduce_sum(mask)
            return error

        @classmethod
        def flow_error_avg(cls, flow_1, flow_2, mask):
            """Evaluates the average endpoint error between flow batches. torch batch is n c h w"""

            def euclidean(t):
                return torch.sqrt(torch.sum(t ** 2, dim=(1,), keepdim=True))

            diff = euclidean(flow_1 - flow_2) * mask
            mask_s = torch.sum(mask)
            diff_s = torch.sum(diff)
            # print('diff_s', diff_s, 'mask_s', mask_s)
            error = diff_s / (mask_s + 1e-6)
            return error

        @classmethod
        def outlier_pct(cls, gt_flow, predflow, mask, threshold=3.0, relative=0.05):
            def euclidean(t):
                return torch.sqrt(torch.sum(t ** 2, dim=(1,), keepdim=True))

            def outlier_ratio(gtflow, predflow, mask, threshold=3.0, relative=0.05):
                diff = euclidean(gtflow - predflow) * mask
                threshold = torch.tensor(threshold).type_as(gtflow)
                if relative is not None:
                    threshold_map = torch.max(threshold, euclidean(gt_flow) * relative)
                    # outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
                    outliers = diff > threshold_map
                else:
                    # outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
                    outliers = diff > threshold
                mask_s = torch.sum(mask)
                outliers_s = torch.sum(outliers)
                # print('outliers_s', outliers_s, 'mask_s', mask_s)
                ratio = outliers_s / mask_s
                return ratio

            frac = outlier_ratio(gt_flow, predflow, mask, threshold, relative) * 100
            return frac

        @classmethod
        def demo(cls):
            class test_model(tools.abs_test_model):

                def eval_forward(self, im1, im2, gt, *args):
                    return gt

                def eval_save_result(self, save_name, predflow, *args, **kwargs):
                    print(save_name)

            eval_ben = kitti_flow.Evaluation_bench(name='2012_train')
            model = test_model()
            occ_pep, occ_rate, noc_pep, noc_rate = eval_ben(model)
            print('occ_pep', occ_pep, 'occ_rate', occ_rate, 'noc_pep', noc_pep, 'noc_rate', noc_rate)

    @classmethod
    def get_file_names(cls, if_test=False):
        def get_img_flow_path_pair(im_dir, flow_occ_dir, flow_noc_dir):
            a = []
            image_files = os.listdir(im_dir)
            image_files.sort()
            flow_occ_files = os.listdir(flow_occ_dir)
            flow_occ_files.sort()
            flow_noc_files = os.listdir(flow_noc_dir)
            flow_noc_files.sort()
            assert len(image_files) % 2 == 0, 'expected pairs of images'
            assert len(flow_occ_files) == len(flow_noc_files), 'here goes wrong'
            assert len(flow_occ_files) == len(image_files) / 2, 'here goes wrong'
            for i in range(len(image_files) // 2):
                filenames_1 = os.path.join(image_dir, image_files[i * 2])
                filenames_2 = os.path.join(image_dir, image_files[i * 2 + 1])
                filenames_gt_occ = os.path.join(flow_dir_occ, flow_occ_files[i])
                filenames_gt_noc = os.path.join(flow_dir_noc, flow_noc_files[i])
                print('occ', flow_occ_files[i], 'noc', flow_noc_files[i], 'im1', image_files[i * 2], 'im2', image_files[i * 2 + 1])
                a.append({'flow_occ': filenames_gt_occ, 'flow_noc': filenames_gt_noc, 'im1': filenames_1, 'im2': filenames_2})
            return a

        def get_img_path_dir(im_dir):
            a = []
            image_files = os.listdir(im_dir)
            image_files.sort()
            assert len(image_files) % 2 == 0, 'expected pairs of images'
            for i in range(len(image_files) // 2):
                filenames_1 = os.path.join(image_dir, image_files[i * 2])
                filenames_2 = os.path.join(image_dir, image_files[i * 2 + 1])
                a.append({'im1': filenames_1, 'im2': filenames_2})
            return a

        file_names_save_path = os.path.join(kitti_flow_dir, 'kitti_flow_2012_2015_file_names.pkl')
        if os.path.isfile(file_names_save_path) and not if_test:
            data = tools.pickle_saver.load_picke(file_names_save_path)
            return data
        else:
            data = {}
            # get 2012 train dataset paths
            image_dir = os.path.join(kitti_flow_dir, 'data_stereo_flow', 'training', 'colored_0')
            flow_dir_occ = os.path.join(kitti_flow_dir, 'data_stereo_flow', 'training', 'flow_occ')
            flow_dir_noc = os.path.join(kitti_flow_dir, 'data_stereo_flow', 'training', 'flow_noc')
            data['2012_train'] = get_img_flow_path_pair(im_dir=image_dir, flow_occ_dir=flow_dir_occ, flow_noc_dir=flow_dir_noc)
            # get 2015 train dataset paths
            image_dir = os.path.join(kitti_flow_dir, 'data_scene_flow', 'training', 'image_2')
            flow_dir_occ = os.path.join(kitti_flow_dir, 'data_scene_flow', 'training', 'flow_occ')
            flow_dir_noc = os.path.join(kitti_flow_dir, 'data_scene_flow', 'training', 'flow_noc')
            data['2015_train'] = get_img_flow_path_pair(im_dir=image_dir, flow_occ_dir=flow_dir_occ, flow_noc_dir=flow_dir_noc)

            # get 2012 test dataset paths
            image_dir = os.path.join(kitti_flow_dir, 'data_stereo_flow', 'testing', 'colored_0')
            data['2012_test'] = get_img_path_dir(im_dir=image_dir)
            # get 2015 test dataset paths
            image_dir = os.path.join(kitti_flow_dir, 'data_scene_flow', 'testing', 'image_2')
            data['2015_test'] = get_img_path_dir(im_dir=image_dir)
            tools.pickle_saver.save_pickle(files=data, file_path=file_names_save_path)
            return data

    class kitti_train():

        def __init__(self, name):
            assert name in ['2012_train', '2015_train', '2012_test', '2015_test']  # ['2012_train', '2015_train']
            data = kitti_flow.get_file_names()
            self.file_names = data[name]
            self.normalize = True
            self.name = name

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, index):
            def pro(*args):
                def func(a):
                    # crop
                    b = a[:, y: y + th, x: x + tw]
                    # expand to 1,c,h,w
                    # b = np.expand_dims(b, 0)
                    return b

                return [func(a) for a in args]

            def read_img(img_path):
                data_ = tf.io.read_file(img_path)
                img = tf.image.decode_image(data_).numpy()
                return img

            data = self.file_names[index]

            im1 = read_img(data['im1'])
            im2 = read_img(data['im2'])
            im1 = img_func.get_process_img_only_img(im1, normalize=self.normalize)
            im2 = img_func.get_process_img_only_img(im2, normalize=self.normalize)
            if self.name in ['2012_test', '2015_test']:
                # crop
                img_name = os.path.basename(data['im1']).replace('.png', '')
                h, w = im1.shape[1:]
                th = int(int(h / 32) * 32)
                tw = int(int(w / 32) * 32)
                x = int((h - th) / 2)
                y = int((w - tw) / 2)
                # im1, im2 = pro(im1, im2)
                im1, im2 = img_func.np_2_tensor(im1, im2)
                return im1, im2, img_name
            else:
                # use tf
                occ, occmask = img_func.read_png_flow(data['flow_occ'])
                noc, nocmask = img_func.read_png_flow(data['flow_noc'])
                # crop
                h, w = im1.shape[1:]
                th = int(int(h / 32) * 32)
                tw = int(int(w / 32) * 32)
                x = int((h - th) / 2)
                y = int((w - tw) / 2)
                im1, im2, occ, occmask, noc, nocmask = img_func.np_2_tensor(im1, im2, occ, occmask, noc, nocmask)
                return im1, im2, occ, occmask, noc, nocmask

        @classmethod
        def demo(cls):
            def process(a):
                b = a.numpy()
                b = np.squeeze(b)
                if len(b.shape) > 2:
                    b = np.transpose(b, (1, 2, 0))
                b = tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            data = kitti_flow.kitti_train('2015_test')
            # loader = tools.data_prefetcher(data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
            for i in range(len(data)):
                im1, im2, occ, occmask, noc, nocmask = data.__getitem__(i)
                tools.check_tensor(im1, 'im1')
                tools.check_tensor(im2, 'im2')


if __name__ == '__main__':
    kitti_flow.Evaluation_bench.demo()
