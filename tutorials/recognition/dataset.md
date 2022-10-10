# 创建数据集

MindFace人脸套件提供了数据集接口 `create_dataset`，方便用户利用接口创建自己的数据集，下边介绍创建数据集的流程

## 调整数据集格式

使用 `create_dataset`创建数据集，需要将label相同的数据放入同一个文件夹中，所有类别的文件夹再放入同一个数据集目录下，数据集的格式如下所示：

```
dataset/
├── 2553
│   ├── Figure_166601.png
│   ├── Figure_166602.png
│   ├── Figure_166603.png
│   ├── Figure_166604.png
|    ...
│   └── Figure_166625.png
├── 2760
│   ├── Figure_178148.png
│   ├── Figure_178149.png
│   ├── Figure_178150.png
│   ├── Figure_178151.png
|    ...
│   └── Figure_178175.png
├── 2968
│   ├── Figure_187283.png
│   ├── Figure_187284.png
│   ├── Figure_187285.png
│   ├── Figure_187286.png
│   ├── Figure_187336.png
│   ├── Figure_187337.png
│   ...
│   └── Figure_187339.png
├── 3174
│   ├── Figure_196740.png
│   ├── Figure_196741.png
│   ├── Figure_196742.png
│   ├── Figure_196743.png
│   ├── Figure_196744.png
│   ├── Figure_196745.png
│   ├── Figure_196746.png
│   │   ...
│   └── Figure_196770.png
└── 3381
    ├── Figure_205344.png
    ├── Figure_205345.png
    ├── Figure_205346.png
    ├── Figure_205347.png
    │   ...
    └── Figure_205370.png
```

`create_dataset`函数会将这种结构的文件夹读取为一个数据集，类别与子文件夹数目相同。

## 配置数据集参数

`create_dataset`的调用格式如下所示，

`create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, augmentation=None, target="Ascend", is_parallel=True)`

该函数能够为训练和测试提供数据，且支持`Ascend`和`GPU`上的单卡和多卡并行工作，本函数内置了训练所用的数据增强方法，可以直接使用。

内置的方法为：
```
# train
trans = [
            C.Decode(),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

# test
trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
]
```
如果需要特殊的数据增强方法，可以使用`augmentation`参数传入。当`augmentation=None`时，使用默认的增强方式。

完整的数据集加载示例如下：

```
from mindspore import context
from mindface.recognition.datasets import *

## Context
context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="GPU", save_graphs=False)

# On single GPU
train_dataset = create_dataset(
                dataset_path="/home/data/dushens/dataset/mindspore/faces_webface_112x112_train", 
                do_train=True, 
                repeat_num=1, 
                batch_size=32, 
                augmentation=None,
                target="GPU", 
                is_parallel=False
            )

print(f"dataset_size: {train_dataset.get_dataset_size()}, batch_size: {train_dataset.get_batch_size()}")

for item in train_dataset.create_dict_iterator():
    print(item.keys(), item['image'].shape, item['label'].shape)
    break
```