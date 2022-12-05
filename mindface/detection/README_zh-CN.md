# 基于MindSpore框架的RetinaFace网络实现


<div align="center">

[English](README.md) | 简体中文

</div>


## 简介
MindSpore是华为在2019年8月推出的新一代全场景AI计算框架，于2020年3月28日正式发布。

RetinaFace是一种实用的单级SOTA人脸检测器，被[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)会议收录。


本存储库包含了RetinaFace的mindspore实现，并取得了很好的性能。我们基于ResNet50和MobileNet0.25做了两个不同版本实现，以满足不同的需求。
![retinaface_picture](imgs/0000_pred.jpg)

## 更新日志
敬请期待!


## 使用Resnet50作为backbone进行单尺度训练在WiderFace Val数据集上的表现.
| 版本 | 简单样本 | 中等样本 | 困难样本 |
|:-|:-:|:-:|:-:|
| MindSpore (与MXNet相同参数) | 94.42% | 93.37% | 89.25% |
| MindSpore (原始图像尺寸) | 95.34% | 93.91% | 84.01% |
| PyTorch (与MXNet相同参数) | 94.82 % | 93.84% | 89.60% |
| PyTorch (原始图像尺寸) | 95.48% | 94.04% | 84.43% |
| MXNet | 94.86% | 93.87% | 88.33% |
| MXNet(原始图像尺寸) | 94.97% | 93.89% | 82.27% |

## 使用Mobilenet0.25作为backbone进行单尺度训练在WiderFace Val数据集上的表现.
| 版本 | 简单样本 | 中等样本 | 困难样本 |
|:-|:-:|:-:|:-:|
| MindSpore (与MXNet相同参数) | 88.62% | 86.96% | 79.93% |
| MindSpore (原始图像尺寸) | 90.73% | 88.24% | 73.87% |
| PyTorch (与MXNet相同参数) | 88.67% | 87.09% | 80.99% |
| PyTorch (原始图像尺寸) | 90.70% | 88.16% | 73.82% |
| MXNet | 88.72% | 86.97% | 79.19% |
| MXNet(原始图像尺寸) | 89.58% | 87.11% | 69.12% |


## 快速入门
1. 安装

    1.1 从[此处](https://github.com/mindspore-ecosystem/mindface.git)下载mindface仓库并安装mindface

    ```shell 
    git clone https://github.com/mindspore-ecosystem/mindface.git
    cd mindface
    python setup.py install
    ```

    1.2 安装依赖包

    ```
    cd mindface/detection/
    pip install -r requirements.txt
    ```

2. 数据集准备

    2.1. 从[百度云](https://pan.baidu.com/s/1eET2kirYbyM04Bg1s12K5g?pwd=jgcf)或[谷歌云盘](https://drive.google.com/file/d/1pBHUJRWepXZj-X3Brm0O-nVhTchH11YY/view?usp=sharing)下载WIDERFACE数据集和标签。
    


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

4. Train

```
    python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
```

5. Eval
```
    python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
```

6. Predict
```
    python infer.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt --image_path ./imgs/0000.jpg --conf 0.5
```



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


