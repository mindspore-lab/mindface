# 本文档将介绍如何finetune
在本教程中，您将学会如何使用MIndFace套件搭建RetinaFace模型并进行微调。
在此之前，请先保证您
1. 安装了mindface.
2. 参考`mindface/detection/README.md`下载数据集和预训练模型

## 加载功能包，调用所需函数
在这一部分，我们集中import所需要的功能包，调用之后需要用到的一些函数

```
import argparse
import math
import mindspore
import os
import sys

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net


from mindface.detection.configs.RetinaFace_mobilenet import cfg_mobile025
from mindface.detection.configs.RetinaFace_resnet50 import cfg_res50
from mindface.detection.loss import MultiBoxLoss
from mindface.detection.datasets import create_dataset
from mindface.detection.utils.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr

from mindface.detection.models import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper, resnet50, mobilenet025
```

## 基本设置
1）使用set_seed函数设置随机种子，在set_context函数中，指定模式为`mode=context.PYNATIVE_MODE`动态图模式，选定GPU平台`device_target='GPU'`进行训练。
选择配置文件为cfg_res50，该文件集中包含一些重要参数的配置,具体配置信息其查阅[config](config.md).

2）使用mindface.detection.datasets中的create_dataset函数，可以轻易加载自定义数据集：
    ·选择配置文件为cfg_res50，该文件中包含一些重要参数的配置
    ·为data_dir变量指定自己的WiderFace训练集路径
    ·指定batch_size = 2

```
#set seed
mindspore.common.seed.set_seed(42)


#set mode
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

# create dataset
# set parameters
cfg = cfg_res50
batch_size = 2
data_dir = '/home/d1/czp21/czp/mindspore/retinaface/retinaface_mindinsight/data/WiderFace/train/label.txt' # changde the dataset path of yours
ds_train = create_dataset(data_dir, cfg, batch_size, multiprocessing=True, num_worker=2)
assert ds_train.get_batch_size() == batch_size
```

## 设置学习率
在adjust_learning_rate函数中设置学习率相关参数，学习率类型lr_type1='dynamic_lr'表示选择动态学习率。

```
#set learning rate schedule
steps_per_epoch = math.ceil(ds_train.get_dataset_size())
lr = adjust_learning_rate(0.01, 0.1, (70,90), steps_per_epoch, 100,
                              warmup_epoch=5, lr_type1='dynamic_lr')
```

## 建立模型
训练模型使用retinaface_resnet50，指定backbone为resnet50，网络参数通过先前导入的cfg配置文件进行配置。

```
#build model
backbone_resnet50 = resnet50(1001)
retinaface_resnet50  = RetinaFace(phase='train', backbone = backbone_resnet50, cfg=cfg)
retinaface_resnet50.set_train(True)
```
这一部分代码如果模型构建没有出问题的话，会直接显示RetinaFace的模型结构。

## 加载预训练模型
当接口中的pretrain_model_path参数设置为预训练权重路径时，可以通过load_checkpoint函数从本地加载.ckpt的预训练模型文件，并通过load_param_into_net函数将backbone和预训练模型加载进训练网络。此处的权重路径读者既可以修改pretrain_model_path为自己权重的具体位置，注意使用`res_50`的配置文件一定要加载名为`RetinaFace_ResNet50.ckpt`的权重，mobilenet亦然。

```

# load checkpoint
pretrain_model_path = '/home/d1/czp21/czp/mindspore/retinaface/retinaface_mindinsight/pretrained/RetinaFace_ResNet50.ckpt'
param_dict_retinaface = load_checkpoint(pretrain_model_path)
load_param_into_net(retinaface_resnet50, param_dict_retinaface)
print(f'Resume Model from [{pretrain_model_path}] Done.')
```
正确运行的结果为

`Resume Model from [/home/d1/czp21/czp/mindspore/retinaface/retinaface_mindinsight/pretrained/RetinaFace_ResNet50.ckpt] Done.`

## 设置loss函数参数
在MultiBoxLoss函数中指定类别数num_classes，此处为2，根据配置文件给定的参数设定矩形框num_boxes数量

```
# set loss
multibox_loss = MultiBoxLoss(num_classes = 2, num_boxes = cfg['num_anchor'], neg_pre_positive=7)
```

## 选择优化器
选择优化器为SGD，用变量learning_rate将学习率lr传入优化器，优化器权重衰减weight_decay=5e-4，loss_scale为梯度放大倍数，此处置为1
```
# set optimazer
opt = mindspore.nn.SGD(params=retinaface_resnet50.trainable_params(), learning_rate=lr, momentum=0.9,
                               weight_decay=5e-4, loss_scale=1)
```
## 将loss函数和优化器加入到网络中
loss函数选用multibox_loss，并将loss函数和优化器整合进训练网络
```
# add loss and optimazer  
net = RetinaFaceWithLossCell(retinaface_resnet50, multibox_loss, cfg)
net = TrainingWrapper(net, opt)
```

## 设置预训练权重参数
保存检查点迭代save_checkpoint_steps，预留检查点数量keep_checkpoint_max，指定模型保存路径ckpt_path，之后开始训练

```
finetune_epochs = 10
model = Model(net)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
callback_list = [LossMonitor(), time_cb, ckpoint_cb]

print("============== Starting Training ==============")
model.train(finetune_epochs, ds_train, callbacks=callback_list, dataset_sink_mode=False)
```

整套流程走下来，模型可以开始微调训练啦，权重保存在`cfg['ckpt_path']`中，输出应当类似于：
```
============== Starting Training ==============

```