
# MindFace: 
Face recognition and detection occupy an important position in the face field. MindFace is an open source toolkit based on MindSpore, containing the most advanced face recognition and detection models, such as ArcFace, RetinaFace and other models, mainly for face recognition and detection and other common application scenarios.

For all main contributors, please check [contributing](#contributing).


## Introduction
---
MindFace mainly has the following features.
- Unified Application Programming Interface

    MindFace provides a unified application programming interface for face recognition and detection by decoupling the models, so that the model can be called directly using the mindface APIs, which greatly improves the ease of building algorithms for users

- Strong extensibility

    MindFace currently supports face recognition and detection, based on the unified APIs. MindFace is highly scalable, it can support lots of backbones, datasets, and loss functions. What's More, MindFace also supports many platforms, including CPU/GPU/Ascend.


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
- easydict==1.9

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instruction](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.   



### Install with pip
MindFace can be installed with pip. 
```shell
pip install mindface
```

### Install from source
To install MindFace from source, please run,
```shell
# Clone the mindface repository.
git clone https://github.com/mindspore-lab/mindface.git
cd mindface

# Install
python setup.py install
```
## Get Started
- [Recognition get started](tutorials/recognition/get_started.md)
- [Detection get started](tutorials/detection/get_started.md)
## Tutorials
---
We provide [tutorials](tutorials) for the recognition and detection task.
### Recognition
- [Get started](tutorials/recognition/get_started.md)
- [Learn about recognition configs](tutorials/recognition/config.md) 
- [Learn to reproduce the eval result and inference with a pretrained model](tutorials/recognition/inference.md) 
- [Learn about how to create dataset](tutorials/recognition/dataset.md)
- [Learn about how to train/finetune a pretrained model](tutorials/recognition/finetune.md)
- [Learn about how to use the loss function](tutorials/recognition/loss.md)
- [Learn about how to create model and custom model](tutorials/recognition/model.md)

### Detection
- [Get started](tutorials/detection/get_started.md)
- [Learn about detection configs](tutorials/detection/config.md)  
- [Inference with a pretrained detection model](tutorials/detection/infer.md) 
- [Finetune a pretrained detection model on WiderFace](tutorials/detection/finetune.ipynb)

---


### Supported Projects

#### Recognition
---
The mindspore implementation of ArcFace and has achieved great performance. We implemented three versions based on ResNet, MobileNet and vit to meet different needs. Detailed results are shown in the table below.

| Datasets       | Backbone            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
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

#### Detection
---
For Face detection, We choose resnet50 and mobilenet0.25 as the backbone, retinaface as the model architecture to achieve efficient performance of face detection. Detailed results are shown in the table below. 

##### WiderFace Val Performance in multiscale When using ResNet50 or mobileNet025 as backbone.
| backbone | Easy | Medium | Hard |
|:-|:-:|:-:|:-:|
| mobileNet0.25 | 91.60% | 89.50% | 82.39% |
| ResNet50 | 95.81% | 94.89% | 90.10% |

## License

This project is released under the [Apache License 2.0](LICENSE.md).

## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindface/issues).

## Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

If you find *MindFace* useful in your research, please consider to cite the following related papers:

```
@misc{MindFace 2022,
    title={{mindface}:mindface for face recognition and detection},
    author={mindface},
    howpublished = {\url{https://github.com/mindlab-ai/mindface/}},
    year={2022}
}

```
---

## Contributing
---
*MindFace* is mainly maintained by the Cross-Media Intelligent Computing (**CMIC**) Laboratory, University of Science and Technology of China (**USTC**), and cooperated with Huawei Technologies Co., Ltd. 

The research topics of CMIC include multimedia computing, multi-modal information perception, cognition and synthesis. 

CMIC has published more than 200 journal articles and conference papers, including TPAMI, TIP, TMM, TASLP, TCSVT, TCYB, TITS, TOMM, TCDS, NeurIPS, ACL, CVPR, ICCV, MM, ICLR, SIGGRAPH, VR, AAAI, IJCAI. 

CMIC has received 6 best paper awards from premier conferences, including CVPR MAVOC, ICCV MFR, ICME, FG. 

CMIC has won 24 Grand Challenge Champion Awards from premier conferences, including CVPR, ICCV, MM, ECCV, AAAI, ICME.

## Notes
* We have created our official repo about face research based on MindSpore. 
* MindFace supports recognition and detection task.

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
- Pengwei Li ``lipw@mail.ustc.edu.cn``
