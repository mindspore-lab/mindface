# 使用模型

MindFace 人脸套件中包含两类网络结构组件

- 骨干网络（backbone）：目前支持的有`iresnet50`, `iresnet100`, `mobilefacenet`, `vit-tiny`, `vit-small`, `vit-base`, `vit-large`等
- 分类层：`PartialFC`，用于支持大规模的分类训练

## 引入现有模型

在人脸识别任务中，常用的需要配置的参数是 backbone 输出的向量维度，下边的例子的输入图像 shape 均为 (3, 112, 112)，输出维度均为 512：

```python
import mindspore as ms
import numpy as np
from mindface.recognition.models import *
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="GPU", save_graphs=False)

imgs = np.random.randn(4,3,112,112)
imgs = ms.Tensor(imgs, dtype=ms.float32)

model_name = "iresnet50"

if model_name == "iresnet50":
    model = iresnet50(num_features=512)
elif model_name == "iresnet100":
    model = iresnet100(num_features=512)
elif model_name == "mobilefacenet":
    model = get_mbf(num_features=512)
elif train_info['backbone'] == 'vit_t':
    net = vit_t(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_s':
    net = vit_s(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_b':
    net = vit_b(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_l':
    net = vit_l(num_features=train_info['num_features'])
else:
    raise NotImplementedError

output = model(imgs)
print(output.shape)

fc = PartialFC(num_classes=100, world_size=1)
out = fc(output)
print(out.shape)
```

## 添加新模型

模型的基本结构如下所示

```python
import mindspore.nn as nn

class NewModel(nn.Cell):
    def __init__(self, arg1, arg2):
        super(NewModel, self).__init__()
        pass
    
    def construct(self, data):
        pass
```

其中`__init__()`函数用初始化模型，包括模型结构搭建和权重初始化，construct 执行前向计算。模型的输入为 B, C, H, W，当前所支持的模型，输入均为三通道（C=3），模型的输出为特征向量，作为各类任务 head 的输入执行下游的计算。

添加到`mindface/recognition/models/__init__.py`中

```python
from .NewModel import *
```
