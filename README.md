# MindFace
<div align="center">

English | [简体中文](README_CN.md)

</div>

| [Introduction](#introduction) | [Installation](#installation) | [Get Started](#get-started) | [Tutorials](#tutorials) | [Model List](#model-list) | [Notes](#notes) | [License](#license) | [Feedbacks and Contact](#feedbacks-and-contact) | [Acknowledgement](#acknowledgement) | [Contributing](#contributing) |

## Introduction

Face recognition and detection occupy an important position in the face field. MindFace is an open source toolkit based on MindSpore, containing the most advanced face recognition and detection models, such as ArcFace, RetinaFace and other models, mainly for face recognition and detection and other common application scenarios.

For all main contributors, please check [contributing](#contributing).

<details>
<summary>Major Features</summary>
MindFace mainly has the following features.

- Unified Application Programming Interface

    MindFace provides a unified application programming interface for face recognition and detection by decoupling the models, so that the model can be called directly using the MindFace APIs, which greatly improves the ease of building algorithms for users

- Strong Extensibility

    MindFace currently supports face recognition and detection, based on the unified APIs. MindFace is highly scalable, it can support lots of backbones, datasets, and loss functions. What's more, MindFace also supports many platforms, including CPU/GPU/Ascend.

</details>

### Benchmark Results
#### Recognition
The MindSpore implementation of ArcFace and has achieved great performance. We implemented three versions based on ResNet and MobileNet to meet different needs. Detailed results are shown in the table below.

| Datasets       | Backbone            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
|:---------------|:--------------------|:------------|:------------|:------------|:------------|:------------|
| CASIA         | mobilefacenet-0.45g | 0.98483+-0.00425 | 0.86843+-0.01838 | 0.90133+-0.02118 | 0.90917+-0.01294 | 0.81217+-0.02232 |
| CASIA         | r50 | 0.98667+-0.00435 | 0.90357+-0.01300 | 0.91750+-0.02277 | 0.92033+-0.01122 | 0.83667+-0.01719 |
| CASIA         | r100 | 0.98950+-0.00366 | 0.90943+-0.01300 | 0.91833+-0.01655 | 0.92433+-0.01017 | 0.84967+-0.01904 |
| MS1MV2         | mobilefacenet-0.45g| 0.98700+-0.00364 | 0.88214+-0.01493 | 0.90950+-0.02076 | 0.91750+-0.01088 | 0.82633+-0.02014 |
| MS1MV2         | r50 | 0.99767+-0.00260 | 0.97186+-0.00652 | 0.97783+-0.00869 | 0.96067+-0.01121 | 0.92033+-0.01732 |
| MS1MV2         | r100 | 0.99383+-0.00334 | 0.96800+-0.01042 | 0.93767+-0.01724 | 0.93267+-0.01327 | 0.89150+-0.01763 |

#### Detection
For face detection, we choose resnet50 and mobilenet0.25 as the backbone, retinaface as the model architecture to achieve efficient performance of face detection. Detailed results are shown in the table below.

| Backbone | Easy | Middle | Hard |
|:-|:-:|:-:|:-:|
| mobileNet0.25 | 91.60% | 89.50% | 82.39% |
| ResNet50 | 95.81% | 94.89% | 90.10% |


## Installation

### Dependency

- mindspore_gpu==1.8.1
- numpy==1.21.6
- opencv_python==4.6.0.66
- scipy==1.7.3
- pyyaml>=5.3
- scikit-learn==1.1.2
- Pillow==9.2.0
- matplotlib==3.6.0

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instruction](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.   

### Install from source
To install MindFace from source, please run
```shell
# Clone the MindFace repository.
git clone https://github.com/mindspore-lab/mindface.git
cd mindface

# Install
python setup.py install
```


## Get Started

To get started with MindFace, please click the following links see the [tutorials for detection](tutorials/detection/get_started.md) and [tutorials for recognition](tutorials/recognition/get_started.md), which will give a quick tour on each key component and the train/validate/predict pipelines in Mindface.


## Tutorials

We provide [tutorials](tutorials) for  

### Detection

- [Learn about detection configs](tutorials/detection/config.md)  
- [Inference with a pretrained detection model](tutorials/detection/infer.md) 
- [Finetune a pretrained detection model on WiderFace](tutorials/detection/finetune.md)

### Recognition

- [Learn about recognition configs](tutorials/recognition/config.md)
- [Inference with a pretrained recognition model](tutorials/recognition/inference.md)
- [Finetune a pretrained recognition model on WiderFace](tutorials/recognition/finetune.md)


## Model List

Currently, Mindface supports the model families listed below. More models with pretrained weights are under development and will be released soon.

<details>
<summary>Supported Models</summary>

- Detection
  - Resnet50
  - Mobilenet0.25
- Recognition
  - arcface-mobilefacenet-0.45g
  - arcface-r50
  - arcface-r100
  - arcface-vit-t
  - arcface-vit-s
  - arcface-vit-b
  - arcface-vit-l

</details>

Please see [here](mindface/detection/configs) for the details about detection models and [here](mindface/recognition/configs) for recognition models.


## Notes

**`2022-06-18`**: We have created our official repo about face research based on MindSpore.


## License

This project is released under the [Apache License 2.0](LICENSE.md).


## Feedbacks and Contact

The dynamic version is still under development, if you find any issues or have an idea on new features, please don't hesitate to contact us via [issues](https://github.com/mindspore-lab/mindface/issues).


## Acknowledgement

MindFace is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

If you find *MindFace* useful in your research, please consider citing the following related papers:

```
@misc{MindFace 2022,
    title={{mindface}:mindface for face recognition and detection},
    author={mindface},
    howpublished = {\url{https://github.com/mindspore-lab/mindface/}},
    year={2022}
}

```


## Contributing

*MindFace* is mainly maintained by the Cross-Media Intelligent Computing (**CMIC**) Laboratory, University of Science and Technology of China (**USTC**), and cooperated with Huawei Technologies Co., Ltd. 

The research topics of CMIC include multimedia computing, multi-modal information perception, cognition and synthesis. 

CMIC has published more than 200 journal articles and conference papers, including TPAMI, TIP, TMM, TASLP, TCSVT, TCYB, TITS, TOMM, TCDS, NeurIPS, ACL, CVPR, ICCV, MM, ICLR, SIGGRAPH, VR, AAAI, IJCAI. 

CMIC has received 6 best paper awards from premier conferences, including CVPR MAVOC, ICCV MFR, ICME, FG. 

CMIC has won 24 Grand Challenge Champion Awards from premier conferences, including CVPR, ICCV, MM, ECCV, AAAI, ICME.

**Main contributors:**

- [Jun Yu](https://github.com/harryjun-ustc), ``harryjun[at]ustc.edu.cn``
- Guochen xie, ``xiegc[at]mail.ustc.edu.cn``
- Shenshen Du, ``dushens[at]mail.ustc.edu.cn``
- Zhongpeng Cai, ``czp_2402242823[at]mail.ustc.edu.cn``
- Peng He, ``hp0618[at]mail.ustc.edu.cn``
- Liwen Zhang, ``zlw1113[at]mail.ustc.edu.cn``
- Hao Chang, ``changhaoustc[at]mail.ustc.edu.cn``
- Mohan Jing, ``jing_mohan@mail.ustc.edu.cn``
- Haoxiang Shi, ``shihaoxiang@mail.ustc.edu.cn``
- Keda Lu, ``wujiekd666[at]gmail.com``
- Pengwei Li, ``3163398705[at]qq.com``
