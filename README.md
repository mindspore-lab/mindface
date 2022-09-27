
# MindFace: 
Face recognition and detection occupy an important position in the face field. MindFace is an open source toolkit based on MindSpore, containing the most advanced face recognition and detection models, such as ArcFace, RetinaFace and other models, mainly for face recognition and detection and other common application scenarios.

For all main contributors, please check [contributing](#contributing).

## Top News
---
**`2022-06-18`**: We have created our official repo about face research based on MindSpore. 

## License
---
The code of MindFace is released under the MIT License. There is no limitation for academic usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.


## Introduction
---
MindFace mainly has the following features.
- Unified algorithm interface


    MindFace provides a unified algorithm application interface for face recognition and detection by decoupling the models, so that the model can be called directly using the algorithm interface, which greatly improves the reusability and ease of use of the model and reduces the cost of using the algorithm.
- Strong extensibility

    For face algorithm scalability, MindFace currently supports face classification and detection models, based on a unified algorithm interface, the model is highly scalable, the model can support multiple backbone, support multiple datasets, support multiple platforms, including CPU/GPU/Ascend, support multiple Loss, etc., and can also support other models in the field of face.
- Easy to use

    For the fragmentation between different task algorithms, MindFace puts the overall model into the code for packaging, which can be installed directly by command, and can be applied directly to different tasks by calling the interface after installation, making it simple to use the algorithm between different tasks.

## Installation
---


### Projects
---
#### Recognition
---
This repository is the mindspore implementation of ArcFace and has achieved great performance. We implemented two versions based on ResNet and MobileNet to meet different needs.

| Datasets       | Backbone            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
|:---------------|:--------------------|:------------|:------------|:------------|:------------|:------------|
| CASIA         | mobilefacenet-0.45g | 0.98483+-0.00425 | 0.86843+-0.01838 | 0.90133+-0.02118 | 0.90917+-0.01294 | 0.81217+-0.02232 |
| CASIA         | r50 | 0.98667+-0.00435 | 0.90357+-0.01300 | 0.91750+-0.02277 | 0.92033+-0.01122 | 0.83667+-0.01719 |
| CASIA         | r100 | 0.98950+-0.00366 | 0.90943+-0.01300 | 0.91833+-0.01655 | 0.92433+-0.01017 | 0.84967+-0.01904 |
| MS1MV2         | mobilefacenet-0.45g| 0.98700+-0.00364 | 0.88214+-0.01493 | 0.90950+-0.02076 | 0.91750+-0.01088 | 0.82633+-0.02014 |
| MS1MV2         | r50 | 0.99767+-0.00260 | 0.97186+-0.00652 | 0.97783+-0.00869 | 0.96067+-0.01121 | 0.92033+-0.01732 |
| MS1MV2         | r100 | 0.99383+-0.00334 | 0.96800+-0.01042 | 0.93767+-0.01724 | 0.93267+-0.01327 | 0.89150+-0.01763 |

#### Detection
---





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
