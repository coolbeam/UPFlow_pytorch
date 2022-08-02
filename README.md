# <center> [CVPR2021] UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning

<h4 align="center"> Kunming Luo<sup>1</sup>, Chuan Wang<sup>1</sup>, Shuaicheng Liu<sup>2,1</sup>, Haoqiang Fan<sup>1</sup>, Jue Wang<sup>1</sup>, Jian Sun<sup>1</sup></h4>

<h4 align="center"> 1. Megvii Technology, 2. University of Electronic Science and Technology of China</h4>

This is the official implementation of the paper [***UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning***](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf) CVPR 2021.

## Abstract
We present an unsupervised learning approach for optical flow estimation by improving the upsampling and learning of pyramid network. We design a self-guided upsample module to tackle the interpolation blur problem caused by bilinear upsampling between pyramid levels. Moreover, we propose a pyramid distillation loss to add supervision for intermediate levels via distilling the finest flow as pseudo labels. By integrating these two components together, our method achieves the best performance for unsupervised optical flow learning on multiple leading benchmarks, including MPI-SIntel, KITTI 2012 and KITTI 2015. In particular, we achieve EPE=1.4 on KITTI 2012 and F1=9.38% on KITTI 2015, which outperform the previous state-of-the-art methods by 22.2% and 15.7%, respectively.

## This repository includes:
- inferring scripts; and 
- pretrain model; 

## Presentation Video
[[Youtube](https://www.youtube.com/watch?v=voD3tA8q-lk&t=4s)], [[Bilibili](https://www.bilibili.com/video/BV1vg41137eH/)]

## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/181239443-f376a2e9-06db-44ba-9602-4915de655aa4.JPG)
Illustration of the pipeline of our network, which contains two stage: pyramid encoding to extract feature pairs in different scales and pyramid decoding to estimate optical flow in each scale. Note that the parameters of the decoder module and the upsample module are shared across all the pyramid levels.

## Self-Guided Upsample Module
![fig4](https://user-images.githubusercontent.com/1344482/181240032-64764d99-908c-4f44-92ae-1d2285a3c791.JPG)

## Usage

Please first install the environments following `how_to_install.md`.

Run `python3 test.py` to test our trained model on KITTI 2015 dataset. Note that Cuda is needed.

## Results
![results](https://user-images.githubusercontent.com/1344482/181240201-e64d788e-a4ae-4e00-8b28-becea0753075.JPG)
Visual example of our self-guided upsample module (SGU) on MPI-Sintel Final dataset. Results of bilinear method and our SGU are shown.

## Citation
If you think this work is helpful, please cite
```
    @inproceedings{luo2021upflow,
      title={Upflow: Upsampling pyramid for unsupervised optical flow learning},
      author={Luo, Kunming and Wang, Chuan and Liu, Shuaicheng and Fan, Haoqiang and Wang, Jue and Sun, Jian},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={1045--1054},
      year={2021}
    }
```



## Acknowledgement
Part of our codes are adapted from [IRR-PWC](https://github.com/visinf/irr), [UnFlow](https://github.com/simonmeister/UnFlow) [ARFlow](https://github.com/lliuz/ARFlow) and [UFlow](https://github.com/google-research/google-research/tree/master/uflow), we thank the authors for their contributions.
