# 使用模型
mindface人脸套件中包含两类网络结构组件
- 骨干网络(backbone)：目前支持的有`iresnet50`, `iresnet100`, `mobilefacenet`等
- 分类层：`PartialFC`，用于支持大规模的分类训练

## 引入现有模型
在人脸识别任务中，常用的需要配置的参数是backbone输出的向量维度，下边的例子的输入图像shape均为(3, 112, 112)，输出维度均为512

```
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
    model = get_mbf(False, 512)
else:
    raise NotImplementedError

output = model(imgs)
print(output.shape)

fc = PartialFC(num_classes=100, world_size=1)
out = fc(output)
print(out.shape)
```

## 添加新模型
1. 编写新模型的结构
模型的基本结构如下所示
```
import mindspore.nn as nn

class NewModel(nn.Cell):
    def __init__(self,arg1, arg2):
        super(NewModel, self).__init__()
        pass
    
    def construct(self, data):
        pass
```
其中__init__函数用初始化模型，包括模型结构搭建和权重初始化，construct执行前向计算

2. 添加到mindface/recognition/models/__init__.py中
```
# 以模型文件名为NewModel为例，在__init__.py中添加引用
from .NewModel import *

```