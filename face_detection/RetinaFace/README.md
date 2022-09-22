# RetinaFace in MindSpore


## Introduction
MindSpore is a new generation of full-scenario AI computing framework launched by Huawei in August 2019 and released On March 28, 2020.

RetinaFace is a practical single-stage SOTA face detector which is accepted by [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html). 


This repository is the mindspore implementation of RetinaFace and has achieved great performance. We implemented two versions based on ResNet50 and MobileNet0.25 to meet different needs.
![retinaface_picture](imgs/0000_pred.jpg)

## Updates
Comming soon!


<!--   
## WiderFace Val Performance

WiderFace Val Performance When using Resnet50 or MobileNet0.25 as backbone, comparing with MXNet implement.
| Model | Easy-set | Mesium-set | Hard-set |
| :-- | :-: | :-: | :-: |
| RetinaFace_mobile025(MindSpore) | 88.62% | 86.96% | 79.93% |
| RetinaFace_mobile025(MXNet) | 88.72% | 86.97% | 79.19% |
| RetinaFace_resnet50(MindSpore) | 94.42% | 93.37% | 89.25% |
| RetinaFace_resnet50(MXNet) | 94.86% | 93.87% | 88.33% | -->

## WiderFace Val Performance in single scale When using Resnet50 as backbone.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| MindSpore (same parameter with MXNet) | 94.42% | 93.37% | 89.25% |
| MindSpore (original image scale) | 95.34% | 93.91% | 84.01% |
| PyTorch (same parameter with MXNet) | 94.82 % | 93.84% | 89.60% |
| PyTorch (original image scale) | 95.48% | 94.04% | 84.43% |
| MXNet | 94.86% | 93.87% | 88.33% |
| MXNet(original image scale) | 94.97% | 93.89% | 82.27% |

## WiderFace Val Performance in single scale When using Mobilenet0.25 as backbone.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| MindSpore (same parameter with MXNet) | 88.62% | 86.96% | 79.93% |
| MindSpore (original image scale) | 90.73% | 88.24% | 73.87% |
| PyTorch (same parameter with MXNet) | 88.67% | 87.09% | 80.99% |
| PyTorch (original image scale) | 90.70% | 88.16% | 73.82% |
| MXNet | 88.72% | 86.97% | 79.19% |
| MXNet(original image scale) | 89.58% | 87.11% | 69.12% |


## Quick Start
1. Installation

    1.1 Git clone this [repo](https://github.com/harryjun-ustc/MindFace)

    ```
    git clone https://github.com/harryjun-ustc/MindFace.git
    ```

    1.2 Install dependencies

    ```
    cd MindFace/face detection/RetinaFace
    pip install -r requirements.txt
    ```

2. Prepare Data

    2.1. Download WIDERFACE dataset and annotations that we used from [baidu cloud](https://pan.baidu.com/s/1eET2kirYbyM04Bg1s12K5g?pwd=jgcf) or [google drive](https://drive.google.com/file/d/1pBHUJRWepXZj-X3Brm0O-nVhTchH11YY/view?usp=sharing).
    


    2.2. Organise the dataset directory under MindFace/RetinaFace/ as follows:
    ```
    data/WiderFace/
        train/
            images/
            label.txt
        val/
            images/
            label.txt
        ground_truth/
            wider_easy_val.mat
            wider_medium_val.mat
            wider_hard_val.mat
            wider_face_val.mat
    ```
3. Set Config File

    You can Modify the parameters of the config file in ```./configs```.

    We provide two versions of configs (MobileNet0.25 and ResNet50).

4. Train


```
    python tools/train.py --backbone ResNet50 or MobileNet025
```

5. Eval
```
    python tools/eval.py --backbone ResNet50 or MobileNet025 --checkpoint pretrained/weight.ckpt
```

6. Predict
```
    python tools/infer.py --backbone ResNet50 or MobileNet025 --checkpoint pretrained/weight.ckpt --image_path ./imgs/0000.jpg --conf 0.5
```



## RetinaFace Pretrained Models
You can download the pretrained model from RetinaFace-ResNet50 ([baidu cloud](link) or [googledrive](link)) and  RetinaFace-MobileNet025 ([baidu cloud](link) or [googledrive](link)). 

You can verify the results in the [table](#widerface-val-performance-in-single-scale-when-using-resnet50-as-backbone-net) with the downloaded pretrained model.


## References
- [Retinaface (MXNet)](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
- [Retinaface (PyTorch)](https://github.com/biubug6/Pytorch_Retinaface)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```


