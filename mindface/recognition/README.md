# Face Recognition in MindSpore
<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction
MindSpore is a new generation of full-scenario AI computing framework launched by Huawei in August 2019 and released On March 28, 2020.

This repository will be continuously updated with a series of algorithms for face recognition, and it contains the mindspore implementation of ArcFace. We implemented two versions based on ResNet and MobileNet to meet different needs.

![arcface](https://user-images.githubusercontent.com/39859528/213472422-844c80c5-fea9-4d92-920a-d8bcac39c37a.png)

<div align="center">Training a DCNN for face recognition supervised by the ArcFace loss. </div>

### Updates!!
+ 【2022/12/20】 vit tiny/small/base/large backbone are supported.
+ 【2022/09/24】 We upload ArcFace based on MindSpore and update the result of eval dataset.
+ 【2022/06/18】 We create this repository.

## Performance on test datasets

### Training on Multi-Host GPU

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


### Pretained models
You can download the pretrained models from [baidu cloud](https://pan.baidu.com/s/1iLw5kOt4Bzr5slA2L9yG_A?pwd=3ggw) or [googledrive](https://drive.google.com/drive/folders/1VoaRX2hpbnC0D1pQ7ex1Trp2EpbYlcAj?usp=sharing) .

You can reproduce the results in the table above with the downloaded pretrained model.

## Quick Start

<summary>Installation</summary>
Step1. Git clone this repo

```shell
git clone https://github.com/mindspore-lab/mindface.git
```

Step2. Install dependencies
```shell
cd mindface
pip install -r requirements.txt
```


<summary>Prepare Data</summary>

Step1. Prepare and download the dataset 
- [CASIA](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#casia-webface-10k-ids05m-images-1) (10K ids/0.5M images)
- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)

Step2. Convert the dataset from rec format to jpg format
```python
cd mindface/recognition
python utils/rec2jpg_dataset.py --include the/path/to/rec --output output/path
```

<summary>Train</summary>

```python
# Distributed training example
mpirun -n 4 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py --config 'configs/train_config_casia_vit_t.yaml' --device_target 'GPU'
```

<summary>Eval</summary>

```python
# Evaluation example
python eval.py --ckpt_url 'pretrained/arcface_vit_t.ckpt' --device_target "GPU" --model "vit_t" --target lfw,cfp_fp,agedb_30,calfw,cplfw
```

<summary>Infer</summary>

```python
# Use infer() to predict the image
>>> img = input_img
>>> out1 = infer(input_img, backbone="vit_t", pretrained="pretrained/arcface_vit_t.ckpt")
```


<summary>Tutorials</summary>

- [Getting Started](../../tutorials/recognition/get_started.md) 
  
- [Learn about recognition configs](../../tutorials/recognition/config.md) 
- [Learn to reproduce the eval result and inference with a pretrained model](../../tutorials/recognition/inference.md) 
- [Learn about how to create dataset](../../tutorials/recognition/dataset.md)
- [Learn about how to train/finetune a model](../../tutorials/recognition/finetune.md)
- [Learn about how to use the loss function](../../tutorials/recognition/loss.md)
- [Learn about how to create model and custom model](../../tutorials/recognition/model.md)



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
