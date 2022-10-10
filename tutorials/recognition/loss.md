# 使用损失函数

损失函数通过计算前向输出与目标之间的差距，引导模型的训练。MindFace套件目前支持的损失函数包括交叉熵损失函数和Arcface损失函数。

## 添加损失函数

MindFace支持添加新的损失函数，损失函数的添加方式如下：

1. 定义损失函数

损失函数的形式为:

```
class Loss(nn.Cell):
    def __init__(self, arg1, arg2):
        super(Loss, self).__init__()
        pass

    def construct(self, pred, label):
        pass
```

2. 添加到`mindface/recognition/loss/__init__.py`中

```
# 以模型文件名为Loss为例，在__init__.py中添加引用
from .Loss import *
```

### 损失函数与模型结合

在训练中，通常将模型和损失函数封装在一个类中，基本结构如下所示：

```
class MyNetWithLoss(nn.Cell):
    def __init__(self, backbone, num_classes, device_num):
        super(MyNetWithLoss, self).__init__(auto_prefix=False)
        # 添加backbone
        self.backbone = backbone
        # 添加分类层
        self.fc = fc
        # 添加损失函数
        self.loss = loss

    def construct(self, data, label):
        out = self.backbone(data)
        out = self.fc(out)
        loss = self.loss(out, label)

        return loss
```

完整示例如下：

```
import mindspore as ms
import numpy as np
from mindface.recognition.models import *
from mindface.recognition.loss import *
from mindspore import context
import mindspore.nn as nn
from mindspore import dtype as mstype

context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="GPU", save_graphs=False)

imgs = np.random.randn(4,3,112,112)
imgs = ms.Tensor(imgs, dtype=ms.float32)

labels = np.array([0,1,2,3])
labels = ms.Tensor(labels, dtype=ms.int32)


class MyNetWithLoss(nn.Cell):
    """
    WithLossCell
    """

    def __init__(self, backbone, num_classes, device_num):
        super(MyNetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone.to_float(mstype.float32)
        self.fc = PartialFC(num_classes=num_classes,
                                  world_size=device_num).to_float(mstype.float32)
        self.margin_softmax = ArcFace(world_size=device_num)
        self.loss = SoftMaxCE(world_size=device_num)
        # self.L2Norm = ops.L2Normalize(axis=1)

    def construct(self, data, label):
        out = self._backbone(data)
        out_fc = self.fc(out)
        out_fc = self.margin_softmax(out_fc, label)
        loss = self.loss(out_fc, label)

        return loss

backbone = iresnet50()
net_with_loss = MyNetWithLoss(backbone, 100, 1)
out = net_with_loss(imgs, labels)
print(out.shape)
```