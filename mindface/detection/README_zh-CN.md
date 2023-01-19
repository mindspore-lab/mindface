# MindFace人脸检测库


<div align="center">

[English](README.md) | 简体中文

</div>


## 简介

MindFace是一款基于MindSpore的开源工具包，目前包含最先进的人脸识别和检测模型，如ArcFace、RetinaFace等。MindFace有统一应用程序编程接口和强大的可扩展性，可以用于面部识别和检测等常见应用场景。

本仓库为MindFace人脸套件的人脸检测库。目前可供用户使用的深度学习模型有基于不同骨干网络（resnet50和mobiletnet0.25）实现的RetinaFace网络模型。
RetinaFace是一种实用的单级SOTA人脸检测器，被[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)会议收录。其中，我们实现的基于ResNet50的版本能提供更好的精度，而基于MobileNet0.25的轻量版本，检测速度更快。

## 性能

### 使用不同backbone进行多尺度测试在WiderFace Val数据集上的表现.

| backbone | Easy | Medium | Hard |
|:-|:-:|:-:|:-:|
| mobileNet0.25 | 91.60% | 89.50% | 82.39% |
| ResNet50 | 95.81% | 94.89% | 90.10% |

### 使用Resnet50作为backbone进行单尺度测试在WiderFace Val数据集上的表现.
| Style | Easy | Medium | Hard |
|:-|:-:|:-:|:-:|
| MindSpore (same parameter with MXNet) | 94.46% | 93.64% | 89.42% |
| MindSpore (original image scale) | 95.07% | 93.61% | 84.84% |
| PyTorch (same parameter with MXNet) | 94.82 % | 93.84% | 89.60% |
| PyTorch (original image scale) | 95.48% | 94.04% | 84.43% |
| MXNet | 94.86% | 93.87% | 88.33% |
| MXNet(original image scale) | 94.97% | 93.89% | 82.27% |

### 使用Mobilenet0.25作为backbone进行单尺度测试在WiderFace Val数据集上的表现.
| Style | Easy | Medium | Hard |
|:-|:-:|:-:|:-:|
| MindSpore (same parameter with MXNet) | 88.51% | 86.86% | 80.88% |
| MindSpore (original image scale) | 90.77% | 88.20% | 74.76% |
| PyTorch (same parameter with MXNet) | 88.67% | 87.09% | 80.99% |
| PyTorch (original image scale) | 90.70% | 88.16% | 73.82% |
| MXNet | 88.72% | 86.97% | 79.19% |
| MXNet(original image scale) | 89.58% | 87.11% | 69.12% |

## 快速入门
1. 安装

    1.1 从[此处](https://github.com/mindspore-lab/mindface.git)下载mindface仓库并安装mindface

    ```shell 
    git clone https://github.com/mindspore-lab/mindface.git
    cd mindface
    python setup.py install
    ```

    1.2 安装依赖包

    ```
    cd mindface/detection/
    pip install -r requirements.txt
    ```

2. 数据集准备

    2.1. 从[百度云](https://pan.baidu.com/s/1eET2kirYbyM04Bg1s12K5g?pwd=jgcf)或[谷歌云盘](https://drive.google.com/file/d/1pBHUJRWepXZj-X3Brm0O-nVhTchH11YY/view?usp=sharing)下载WiderFace数据集和标签。
    


    2.2. 在 mindface/detection/ 目录下存放数据集，结构树如下所示:
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
3. 修改配置文件

    请在配置文件 ```./configs```中修改参数.

    我们提供了两种配置文件([RetinaFace_mobilenet025](./configs/RetinaFace_mobilenet025.yaml) and [RetinaFace_resnet50](./configs/RetinaFace_resnet50.yaml)).

4. 训练

    通过运行`train.py`可以训练你自己的人脸检测模型，使用的模型方法和训练策略可以通过命令行参数或者yaml配置文件来配置。

    - 单卡训练
    ```shell
        python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```

    - 多卡训练

        为了使用多卡训练，需要先安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/).
    ```shell
        export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs
        mpirun -n 4 python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```
    > 注意：如果你的设备是Ascend，请在配置文件中设置 "device_target "为 "Ascend"。
5. 验证

    通过运行`eval.py`可以对模型在WiderFace上的性能做评估。
```
    python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
```

6. 推理

    通过运行`infer.py`可以对单张图片进行推理。
```
    python infer.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt --image_path test/detection/imgs/0000.jpg --conf 0.5
```

 
## 推理结果示意图
![retinaface_picture](/test/detection/imgs/0000_pred.jpg)


## RetinaFace预训练模型
从此处下载预训练模型
[RetinaFace-ResNet50](https://download.mindspore.cn/toolkits/mindface/retinaface/RetinaFace_ResNet50.ckpt)
[RetinaFace-MobileNet025](https://download.mindspore.cn/toolkits/mindface/retinaface/RetinaFace_MobileNet025.ckpt)

你可以在此[表格](#widerface-val-performance-in-single-scale-when-using-resnet50-as-backbone-net)中核验预训练模型和结果.


## 参考
- [Retinaface (MXNet)](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
- [Retinaface (PyTorch)](https://github.com/biubug6/Pytorch_Retinaface)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```


