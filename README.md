# <center> [CVPR2021] UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning

<h3 align="center"> Kunming Luo$^1$, Chuan Wang$^1$, Shuaicheng Liu$^2$, Haoqiang Fan$^1$, Jue Wang$^1$, Jian Sun$^1$</h3>

<h3 align="center"> 1. Megvii Technology, 2. University of Electronic Science and Technology of China</h3>

This is the official implementation of the paper [***UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning***](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf) CVPR 2021.


```
    @inproceedings{luo2021upflow,
      title={Upflow: Upsampling pyramid for unsupervised optical flow learning},
      author={Luo, Kunming and Wang, Chuan and Liu, Shuaicheng and Fan, Haoqiang and Wang, Jue and Sun, Jian},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={1045--1054},
      year={2021}
    }
```


## Introduction
We present an unsupervised learning approach for optical flow estimation by improving the upsampling and learning of pyramid network. We design a self-guided upsample module to tackle the interpolation blur problem caused by bilinear upsampling between pyramid levels. Moreover, we propose a pyramid distillation loss to add supervision for intermediate levels via distilling the finest flow as pseudo labels. By integrating these two components together, our method achieves the best performance for unsupervised optical flow learning on multiple leading benchmarks, including MPI-SIntel, KITTI 2012 and KITTI 2015. In particular, we achieve EPE=1.4 on KITTI 2012 and F1=9.38% on KITTI 2015, which outperform the previous state-of-the-art methods by 22.2% and 15.7%, respectively.

This repository includes(is coming):

- inferring scripts; and 
- pretrain model; 

## Usage

Please first install the environments following `how_to_install.md`.

Run `python3 test.py` to test our trained model on KITTI 2015 dataset. Note that Cuda is needed.
    
## Acknowledgement
Part of our codes are adapted from [IRR-PWC](https://github.com/visinf/irr), [UnFlow](https://github.com/simonmeister/UnFlow) [ARFlow](https://github.com/lliuz/ARFlow) and [UFlow](https://github.com/google-research/google-research/tree/master/uflow), we thank the authors for their contributions.
