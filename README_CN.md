# MindFace: 
<div align="center">

[English](README.md) | 简体中文

</div>

| [简介](#简介) | [安装](#安装) | [快速入门](#快速入门) | [教程](#教程) | [模型列表](#模型列表) | [更新日志](#更新日志) |

## 简介

人脸识别与检测在人脸领域占有重要地位。MindFace是一款基于MindSpore的开源工具包，包含最先进的人脸识别和检测模型，如ArcFace、RetinaFace和其他模型，主要用于面部识别和检测等常见应用场景。

您可以点击[贡献者](#贡献者)，以了解本项目主要贡献者。

<details>
<summary>主要优点</summary>
MindFace主要具有以下优点:

- 统一应用程序编程接口

    MindFace通过解耦模型为人脸识别和检测提供了统一的应用程序编程接口，从而可以使用MindFace API直接调用模型，这大大降低了用户构建算法的难度。

- 强大的可扩展性

    MindFace目前支持基于统一API的人脸识别和检测，具有强大可扩展性，它可以支持许多主干网络、数据集和损失函数。此外，MindFace还支持多平台调试，包括CPU、GPU和Ascend平台。

</details>

### 基准结果
#### 识别任务
基于MindSpore实现的ArcFace系列模型取得了良好性能。我们基于ResNet、MobileNet和vit实现了三个版本，以满足不同的需求。详细结果如下表所示。

| 数据集       | 主干网络            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
|:---------------|:--------------------|:------------|:------------|:------------|:------------|:------------|
| CASIA         | mobilefacenet-0.45g | 0.98483+-0.00425 | 0.86843+-0.01838 | 0.90133+-0.02118 | 0.90917+-0.01294 | 0.81217+-0.02232 |
| CASIA         | r50 | 0.98667+-0.00435 | 0.90357+-0.01300 | 0.91750+-0.02277 | 0.92033+-0.01122 | 0.83667+-0.01719 |
| CASIA         | r100 | 0.98950+-0.00366 | 0.90943+-0.01300 | 0.91833+-0.01655 | 0.92433+-0.01017 | 0.84967+-0.01904 |
| CASIA         | vit-t | 0.98400+-0.00704 | 0.83229+-0.01877 | 0.87283+-0.02468 | 0.90667+-0.00934 | 0.80700+-0.01767 |
| CASIA         | vit-s | 0.98550+-0.00806 | 0.85557+-0.01617 | 0.87850+-0.02194 | 0.91083+-0.00876 | 0.82500+-0.01685 |
| CASIA         | vit-b | 0.98333+-0.00553 | 0.85829+-0.01836 | 0.87417+-0.01838 | 0.90800+-0.00968 | 0.81400+-0.02236 |
| CASIA         | vit-l | 0.97600+-0.00898 | 0.84543+-0.01718 | 0.85317+-0.01411 | 0.89733+-0.00910 | 0.79550+-0.01648 |
| MS1MV2         | mobilefacenet-0.45g| 0.98700+-0.00364 | 0.88214+-0.01493 | 0.90950+-0.02076 | 0.91750+-0.01088 | 0.82633+-0.02014 |
| MS1MV2         | r50 | 0.99767+-0.00260 | 0.97186+-0.00652 | 0.97783+-0.00869 | 0.96067+-0.01121 | 0.92033+-0.01732 |
| MS1MV2         | r100 | 0.99383+-0.00334 | 0.96800+-0.01042 | 0.93767+-0.01724 | 0.93267+-0.01327 | 0.89150+-0.01763 |
| MS1MV2         | vit-t | 0.99717+-0.00279 | 0.92714+-0.01389 | 0.96717+-0.00727 | 0.95600+-0.01198 | 0.89950+-0.01291 |
| MS1MV2         | vit-s | 0.99767+-0.00260 | 0.95771+-0.01058 | 0.97617+-0.00972 | 0.95800+-0.01142 | 0.91267+-0.01104 |
| MS1MV2         | vit-b | 0.99817+-0.00252 | 0.94200+-0.01296 | 0.97517+-0.00858 | 0.96000+-0.01179 | 0.90967+-0.01152 |
| MS1MV2         | vit-l | 0.99750+-0.00291 | 0.93714+-0.01498 | 0.96483+-0.01031 | 0.95817+-0.01158 | 0.90450+-0.01062 |

#### 检测任务
对于检测任务，我们选取了Resnet50和Mobilenet0.25作为主干网络，Retinaface作为模型架构，以实现良好的人脸检测性能。详细结果如下表所示。

| 数据集 | 主干网络 | 简单 | 中等 | 困难 |
|:-|:-|:-:|:-:|:-:|
| WiderFace | mobileNet0.25 | 91.60% | 89.50% | 82.39% |
| WiderFace | ResNet50 | 95.81% | 94.89% | 90.10% |


## 安装

### 依赖包

- mindspore_gpu==1.8.1
- numpy==1.21.6
- opencv_python==4.6.0.66
- scipy==1.7.3
- pyyaml>=5.3
- scikit-learn==1.1.2
- Pillow==9.2.0
- matplotlib==3.6.0

请运行下示指令以安装所需依赖包
```shell
pip install -r requirements.txt
```

参见MindSpore官网[安装教程](https://www.mindspore.cn/install)，我们可以便捷地完成框架安装。为了能够分布式运行程序，我们还需安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/) 。

### 利用源进行安装
利用源安装MindFace，请运行：
```shell
# Clone the MindFace repository.
git clone https://github.com/mindspore-lab/mindface.git
cd mindface

# Install
python setup.py install
```


## 快速入门

为了快速入门MindFace，请点击[检测教程](tutorials/detection/get_started_CN.md)和[识别教程](tutorials/recognition/get_started.md)，仔细阅读文档，其中将会对Mindface中的每个关键组件和训练、验证、预测进行快速介绍。


## 教程

我们提供以下[教程](tutorials)

### 检测任务

- [检测任务配置](tutorials/detection/config.md)  
- [使用预训练的检测模型进行推理](tutorials/detection/infer.md) 
- [在WiderFace数据集上微调预训练模型](tutorials/detection/finetune.md)

### 识别任务

- [识别任务配置](tutorials/recognition/config.md)
- [使用预训练的检测模型进行推理](tutorials/recognition/inference.md)
- [在WiderFace数据集上微调预训练模型](tutorials/recognition/finetune.md)


## 模型列表
目前，MindFace支持以下模型。更多带有预训练权重的模型正在开发中，将于近期发布。

<details>
<summary>支持模型</summary>

- 检测任务
  - Resnet50
  - Mobilenet0.25
- 识别任务
  - arcface-mobilefacenet-0.45g
  - arcface-r50
  - arcface-r100
  - arcface-vit-t
  - arcface-vit-s
  - arcface-vit-b
  - arcface-vit-l

</details>

请点击[这里](mindface/detection/configs)了解更多关于检测任务模型，点击[这里](mindface/recognition/configs)了解更多关于识别任务模型。


## 更新日志

**`2022-06-18`**: 我们已经发布了基于MindSpore的面部研究官方报告。


## 许可证

本项目基于[Apache 许可证 2.0](LICENSE.md).


## 反馈和联系

新版本正在开发中，如果您有任何问题或者建议，请通过[issues](https://github.com/mindspore-lab/mindface/issues)与我们联系。


## 引用需知

MindFace是一款开源项目，我们欢迎任何贡献和反馈。我们希望工具箱和基准性能可以通过提供灵活和标准化的工具箱来重新实现现有方法，并基于此开发出新的计算机视觉算法，为研究社区贡献一份力量。

如果您觉得*MindFace*对您的研究有所帮助，希望您能考虑引用以下文章:

```
@misc{MindFace 2022,
    title={{mindface}:mindface for face recognition and detection},
    author={mindface},
    howpublished = {\url{https://github.com/mindspore-lab/mindface/}},
    year={2022}
}

```


## 贡献者

*MindFace*项目主要由中国科学技术大学(USTC)跨媒体智能计算联合实验室(以下简称“CMIC实验室”)和华为技术有限公司共同维护。

CMIC实验室的研究主题包括多媒体计算、多模态信息感知、认知和合成。 
目前已在TPAMI、TIP、TMM、TASLP、TCSVT、TCYB、TITS、TOMM、TCDS、NeurIPS、ACL、CVPR、ICCV、MM、ICLR、SIGGRAPH、VR、AAAI、IJCAI上发表了200多篇期刊文章和会议论文，收获了包括CVPR、MAVOC、ICCV MFR、ICME、FG在内的6项顶级会议的最佳论文奖。CMIC实验室也从包括CVPR、ICCV、MM、ECCV、AAAI、ICME在内的顶级会议上获得了24项大奖。


**主要贡献者:**

- [Jun Yu](https://github.com/harryjun-ustc), ``harryjun[at]ustc.edu.cn``
- Guochen xie, ``xiegc[at]mail.ustc.edu.cn``
- Shenshen Du, ``dushens[at]mail.ustc.edu.cn``
- Zhongpeng Cai, ``czp_2402242823[at]mail.ustc.edu.cn``
- Peng He, ``hp0618[at]mail.ustc.edu.cn``
- Liwen Zhang, ``zlw1113[at]mail.ustc.edu.cn``
- Hao Chang, ``changhaoustc[at]mail.ustc.edu.cn``
- Mohan Jing, ``jing_mohan@mail.ustc.edu.cn``
- Haoxiang Shi, ``shihaoxiang@mail.ustc.edu.cn``
- Keda Lu, ``lukeda@mail.ustc.edu.cn``
- Pengwei Li, ``lipw@mail.ustc.edu.cn``
