# ArcFace in MindSpore

## Introduction
MindSpore is a new generation of full-scenario AI computing framework launched by Huawei in August 2019 and released On March 28, 2020.

This repository is the mindspore implementation of ArcFace and has achieved great performance. We implemented two versions based on ResNet and MobileNet to meet different needs.
<div align="center"><img src="image/arcface.png" width="600" ></div>


## Updates!!
+ 【2022/09/24】 We upload ArcFace based on MindSpore and update the result of eval dataset.
+ 【2022/06/18】 We create this repository.

## Coming soon
- [ ] MS1MV3, Glint360K, WebFace42M dataset pretrained models.
- [ ] Support vit backbone.



## Performance on lfw, cfp_fp, agedb_30, calfw and cplfw

## 1. Training on Multi-Host GPU

| Datasets       | Backbone            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
|:---------------|:--------------------|:------------|:------------|:------------|:------------|:------------|
| CASIA         | mobilefacenet-0.45g | 0.98483+-0.00425 | 0.86843+-0.01838 | 0.90133+-0.02118 | 0.90917+-0.01294 | 0.81217+-0.02232 |
| CASIA         | r50 | 0.98667+-0.00435 | 0.90357+-0.01300 | 0.91750+-0.02277 | 0.92033+-0.01122 | 0.83667+-0.01719 |
| CASIA         | r100 | 0.98950+-0.00366 | 0.90943+-0.01300 | 0.91833+-0.01655 | 0.92433+-0.01017 | 0.84967+-0.01904 |
| MS1MV2         | mobilefacenet-0.45g| 0.98700+-0.00364 | 0.88214+-0.01493 | 0.90950+-0.02076 | 0.91750+-0.01088 | 0.82633+-0.02014 |
| MS1MV2         | r50 | 0.99767+-0.00260 | 0.97186+-0.00652 | 0.97783+-0.00869 | 0.96067+-0.01121 | 0.92033+-0.01732 |
| MS1MV2         | r100 | 0.99383+-0.00334 | 0.96800+-0.01042 | 0.93767+-0.01724 | 0.93267+-0.01327 | 0.89150+-0.01763 |


## Pretained models
Soon

## Quick Start

<summary>Installation</summary>
Step1. Git clone this repo
```shell
git clone https://github.com/mindlab-ai/mindface.git
```
Step2. Install dependencies
```shell
cd mindface
pip install -r requirements.txt
```


<summary>Prepare Data</summary>

Step1. Prepare and Download the dataset 
- [CASIA](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#casia-webface-10k-ids05m-images-1) (10K ids/0.5M images)
- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)

Step2. Convert the dataset from rec format to jpg format
```python
python utils/rec2jpg_dataset.py --include the/path/to/rec --output output/path
```






<summary>Train and Eval</summary>
The example commands below show how to run distributed training.

Step1. Train
```shell
# Distributed training example
bash scripts/run_distribute_train.sh rank_size /path/dataset
```

Step2. Eval
```shell
# Evaluation example
bash scripts/run_eval.sh /path/evalset /path/ckpt
```




<summary>Tutorials</summary>
Soon



## Citations

```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{An_2022_CVPR,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={4042-4051}
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10492--10502},
  year={2021}
}
```