# UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning

By Kunming Luo, [Chuan Wang](http://wangchuan.github.io/index.html), [Shuaicheng Liu](http://www.liushuaicheng.org/), [Haoqiang Fan](https://scholar.google.com.hk/citations?hl=zh-CN&user=bzzBut4AAAAJ), [Jue Wang](http://www.juew.org/), [Jian Sun](https://scholar.google.com.hk/citations?hl=zh-CN&user=ALVSZAYAAAAJ)

Megvii Technology, University of Electronic Science and Technology of China

[[Preprint](https://arxiv.org/abs/2012.00212)]

    @article{luo2020upflow,
      title={UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning},
      author={Luo, Kunming and Wang, Chuan and Liu, Shuaicheng and Fan, Haoqiang and Wang, Jue and Sun, Jian},
      journal={arXiv preprint arXiv:2012.00212},
      year={2020}
    }

## Introduction
We present an unsupervised learning approach for optical flow estimation by improving the upsampling and learning of pyramid network. We design a self-guided upsample module to tackle the interpolation blur problem caused by bilinear upsampling between pyramid levels. Moreover, we propose a pyramid distillation loss to add supervision for intermediate levels via distilling the finest flow as pseudo labels. By integrating these two components together, our method achieves the best performance for unsupervised optical flow learning on multiple leading benchmarks, including MPI-SIntel, KITTI 2012 and KITTI 2015. In particular, we achieve EPE=1.4 on KITTI 2012 and F1=9.38% on KITTI 2015, which outperform the previous state-of-the-art methods by 22.2% and 15.7%, respectively.

This repository includes(is coming):

- Training scripts based on Python and PyTorch; and
- inferring scripts for several benchmarks; and 
- pretrain models. 


    
## Acknowledgement
Part of our codes are adapted from [IRR-PWC](https://github.com/visinf/irr) and [UnFlow](https://github.com/simonmeister/UnFlow), we thank the authors for their contributions.
