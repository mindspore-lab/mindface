# 快速上手指南

本教程旨在用一个简单的案例，带领大家了解MindFace人脸套件的功能，使大家对MindFace人脸套件有一个直观的认识

## 读取图像
使用者可以使用自己收集的图像或本文提供的示例图像
```
# 下载图像
wget https://img0.baidu.com/it/u=870969850,1023437826&fm=253&fmt=auto&app=138&f=JPEG?w=369&h=550
```
加载图像并转换为Tensor格式, 尺寸resize为112x112
```
from PIL import Image
import numpy as np
import mindspore as ms

img_path = "/path/to/images" # 此处可以更换为自己的图像路径
img = Image.open(img_path)
img = img.resize((112, 112), Image.BICUBIC)
img = np.array(img).transpose(2,0,1)
img = ms.Tensor(img, ms.float32)
print(img.shape)
```

## 执行推理

为方便用户使用，在MindFace人脸套件中，我们提供了`infer`函数，使用训练好的模型处理输入图像，获取特征向量。`infer`函数的参数介绍如下：

`def infer(img, backbone="iresnet50", num_features=512, pretrained=False):`
* `img`: ms.Tensor，可以是单张图像也可以是图像batch
* `backbone`: str，使用的模型，当前MindFace支持的模型包括iresnet50, iresnet100, mobilefacenet等
* `num_features`: int，特征维度
* `pretrained`: 模型权重存放的位置，如果没有权重，此处设置为False

代码示例如下：

```
from mindface.recognition.infer import infer

backbone = 'iresnet50'
pretrained = '/path/to/ckpt'
num_features=512
feature = infer(img, backbone=backbone, num_features=num_features, pretrained=pretrained)
print(feature.shape)
```

## 模型验证
MindFace为用户提供了模型验证的接口`face_eval`函数，能够在指定的人脸验证数据集上验证模型的性能，当前

`def face_eval(model_name, ckpt_url, eval_url, num_features=512, target='lfw,cfp_fp,agedb_30,calfw,cplfw', device_id=0, device_target="GPU", batch_size=64, nfolds=10):`
* `model_name`: str, 使用的模型，当前MindFace支持的模型包括iresnet50, iresnet100, mobilefacenet等
* `ckpt_url`: str, 待验证的checkpoint的路径
* `eval_url`: str, 验证数据集的路径
* `num_features`: int，特征维度
* `target`: str, 待验证的数据集，目前支持lfw,cfp_fp,agedb_30,calfw,cplfw五个数据集
* `device_id`: int, 使用的设备ID序号
* `device_target`: str, 运行程序所使用的平台，目前支持GPU和Ascend平台两种
* `batch_size`: int, 验证过程中每批数据数据的大小
* `nfolds`: int, 验证折数

代码示例如下：

```
from mindface.recognition.eval import face_eval

model_name = "iresnet100"
ckpt_path = "/path/to/ckpt"
eval_path = "/path/to/eval_dataset"
num_features=512

face_eval(model_name, ckpt_path, eval_path, num_features)
```