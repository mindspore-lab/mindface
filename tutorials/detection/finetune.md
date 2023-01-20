# 本文档将介绍如何finetune
在本教程中，您将学会如何使用MindFace套件搭建RetinaFace模型并进行微调。本文档将分成四个模块（数据准备、模型创建、训练微调、模型评估）详细介绍。

## 准备数据
---

1. 安装mindface

    1.1 从[此处](https://github.com/mindspore-lab/mindface.git)下载mindface仓库并安装mindface

    ```shell 
    git clone https://github.com/mindspore-lab/mindface.git
    cd mindface
    python setup.py install
    ```

    1.2 安装依赖包

    ```shell
    pip install -r requirements.txt
    ```

2. 数据集准备

    2.1. 从[百度云](https://pan.baidu.com/s/1eET2kirYbyM04Bg1s12K5g?pwd=jgcf)或[谷歌云盘](https://drive.google.com/file/d/1pBHUJRWepXZj-X3Brm0O-nVhTchH11YY/view?usp=sharing)下载WIDERFACE数据集和标签。
    
    2.2. 在 mindface/detection/ 目录下存放数据集，结构树如下所示:
    ```text
    data/WiderFace/
        train/
            images/
            label.txt
        val/
            images/
            label.txt
        ground_truth/
            wider_easy_val.mat
            wider_medium_val.mat
            wider_hard_val.mat
            wider_face_val.mat
    ```

3. 下载预训练模型用于微调
从此处下载预训练模型
[RetinaFace-ResNet50](https://download.mindspore.cn/toolkits/mindface/retinaface/RetinaFace_ResNet50.ckpt)
[RetinaFace-MobileNet025](https://download.mindspore.cn/toolkits/mindface/retinaface/RetinaFace_MobileNet025.ckpt)
    
## 构建模型
---
### 加载功能包，调用所需函数

在这一部分，我们集中import所需要的功能包，调用之后需要用到的一些函数。

```python
import argparse
import math
import mindspore

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindface.detection.loss import MultiBoxLoss
from mindface.detection.datasets import create_dataset
from mindface.detection.utils import adjust_learning_rate

from mindface.detection.models import RetinaFace, RetinaFaceWithLossCell, resnet50, mobilenet025
from mindface.detection.runner import read_yaml, TrainingWrapper
```

使用`set_seed`函数设置随机种子，在set_context函数中，指定模式为`mode=context.PYNATIVE_MODE`动态图模式，也可以更改成静态图模式，通过修改`mode=context.GRAPH_MODE`选定GPU平台`device_target='GPU'`进行训练。


```python
#set seed
mindspore.common.seed.set_seed(42)

#set mode
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
```

### 加载配置文件数据集
设置config的地址，用`read_yaml`函数加载config文件，此处选择的是RetinaFace_mobilenet025的配置文件，读者可自行修改路径。
设置数据集路径为数据集准备中下好的数据集路径。然后调用`create_dataset`加载数据集。
```python
# create dataset
cfg = read_yaml(config_cfg) #config_cfg为配置文件的地址
# change the dataset path of yours
data_dir = 'mindface/detection/data/WiderFace/train/label.txt'
ds_train = create_dataset(data_dir, variance=[0.1,0.2], match_thresh=0.35, image_size=640, clip=False, batch_size=8,
                        repeat_num=1, shuffle=True, multiprocessing=True, num_worker=4, is_distribute=False)
print('dataset size is : \n', ds_train.get_dataset_size())

```
正确加载数据集的输出结果为：
```text
dataset size is : 
 1609
```

### 设置学习率
使用`adjust_learning_rate`函数中设置学习率相关参数，`adjust_learning_rate`的参数分别为：`initial_lr`（初始化学习率）, `gamma`, `stepvalues`, `steps_pre_epoch`, `total_epochs`, `warmup_epoch=5`, 学习率类型`lr_type1='dynamic_lr'`表示选择动态学习率。

```python
#set learning rate schedule
steps_per_epoch = math.ceil(ds_train.get_dataset_size())
lr = adjust_learning_rate(0.01, 0.1, (70,90), steps_per_epoch, 100,
                              warmup_epoch=5, lr_type1='dynamic_lr')
```

### 构建retinaface_mobilenet025
训练模型使用以mobilenet0.25为骨干网络的RetinaFace网络模型。

```python
#build model
backbone_mobilenet025 = mobilenet025(1000)
retinaface_mobilenet025  = RetinaFace(phase='train', backbone=backbone_mobilenet025, in_channel=32, out_channel=64)
retinaface_mobilenet025.set_train(True)
```
这一部分代码如果模型构建没有出问题的话，会直接显示RetinaFace的模型结构。

## 训练微调
---
### 加载预训练模型
当我们有了一个预训练权重的时候，可以通过`load_checkpoint`函数从本地加载预训练模型文件，并通过`load_param_into_net`函数将backbone和预训练模型加载进训练网络。此处的权重路径读者可以修改`pretrain_model_path`为自己下载的权重的具体位置，注意使用`RetinaFace_mobilenet025.yaml`的配置文件一定要加载名为`RetinaFace_MobileNet025.ckpt`的权重，使用resnet版本亦然。

```python
# load checkpoint
pretrain_model_path = 'minbdface/detecton/pretrained/RetinaFace_MobileNet025.ckpt'
param_dict_retinaface = load_checkpoint(pretrain_model_path)
load_param_into_net(retinaface_mobilenet025, param_dict_retinaface)
print(f'Resume Model from [{pretrain_model_path}] Done.')
```
正确运行的结果为：
```text
Resume Model from [minbdface/detecton/pretrained/RetinaFace_MobileNet025.ckpt] Done.
```
### 设置loss函数参数
在`MultiBoxLoss`函数中指定类别数`num_classes`，此处为2（只检测人脸），根据配置文件给定的参数设定矩形框`num_boxes`数量。


```python
# set loss
multibox_loss = MultiBoxLoss(num_classes = 2, num_boxes = 16800, neg_pre_positive=7)
```

### 选择优化器
选择优化器为`SGD`，用变量`learning_rate`将学习率`lr`传入优化器，优化器权重衰减`weight_decay=5e-4`，`loss_scale`为梯度放大倍数，此处置为1。
```python
# set optimazer
opt = mindspore.nn.SGD(params=retinaface_resnet50.trainable_params(), learning_rate=lr, momentum=0.9,
                               weight_decay=5e-4, loss_scale=1)
```
### 将loss函数和优化器加入到网络中
loss函数用`multibox_loss`，并将loss函数和优化器整合进训练网络。
```python
# add loss and optimazer  
net = RetinaFaceWithLossCell(retinaface_resnet50, multibox_loss, loc_weight=2.0, class_weight=1.0, landm_weight=1.0)
net = TrainingWrapper(net, opt)
```

### 设置预训练权重参数
保存检查点迭代`save_checkpoint_steps`，预留检查点数量`keep_checkpoint_max`，指定模型保存路径`ckpt_path`，之后开始训练。

```python
finetune_epochs = 10
model = Model(net)
config_ck = CheckpointConfig(save_checkpoint_steps=1000,
                                 keep_checkpoint_max=3)
ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
callback_list = [LossMonitor(), time_cb, ckpoint_cb]

print("============== Starting Training ==============")
model.train(finetune_epochs, ds_train, callbacks=callback_list, dataset_sink_mode=False)
```

整套流程走下来，模型可以开始微调训练啦，权重保存在`cfg['ckpt_path']`中，输出应当类似于：
```text
============== Starting Training ==============
epoch: 1 step: 1, loss is 39.44330978393555
epoch: 1 step: 2, loss is 40.87006378173828
epoch: 1 step: 3, loss is 37.974769592285156
epoch: 1 step: 4, loss is 39.08790588378906
epoch: 1 step: 5, loss is 37.38018035888672
epoch: 1 step: 6, loss is 36.850799560546875
epoch: 1 step: 7, loss is 36.3499755859375
epoch: 1 step: 8, loss is 35.01698303222656
epoch: 1 step: 9, loss is 35.564842224121094
epoch: 1 step: 10, loss is 35.35591125488281
epoch: 1 step: 11, loss is 32.42792510986328
epoch: 1 step: 12, loss is 31.537368774414062
epoch: 1 step: 13, loss is 31.820585250854492
epoch: 1 step: 14, loss is 31.04840850830078
...

```

## 模型评估

### 切换模型为`predict`模式并冻结模型参数
```python
network = RetinaFace(phase='predict', backbone=backbone, in_channel=32, out_channel=64)
backbone.set_train(False)
net.set_train(False)
```

### 加载微调好的权重
```python
cfg['val_model'] = './pretrained/RetinaFace.ckpt'
assert cfg['val_model'] is not None, 'val_model is None.'
param_dict = load_checkpoint(cfg['val_model'])
print('Load trained model done. {}'.format(cfg['val_model']))
network.init_parameters_data()
load_param_into_net(network, param_dict)
```

### 构建验证集
通过设置`val_dataset_folder`为验证集的路径，读取出每张图片位置。
```python
testset_folder = cfg['val_dataset_folder']
testset_label_path = cfg['val_dataset_folder'] + "label.txt"
with open(testset_label_path, 'r') as f:
    _test_dataset = f.readlines()
    test_dataset = []
    for im_path in _test_dataset:
        if im_path.startswith('# '):
            test_dataset.append(im_path[2:-1])

num_images = len(test_dataset)
print(num_images)
```

输出结果为
```text
3226
```

### 对验证集图片做初步处理
如果`cfg['val_origin_size']`为`True`,则根据输入图片大小的不同计算需要使用的prior boxes，设置为`False`则可以直接使用设定好的图片尺寸然后做resize。
```python
# 初始化计时器，forward_time表示网络推理的时间，misc表示后处理的时间。
timers = {'forward_time': Timer(), 'misc': Timer()}

if cfg['val_origin_size']:
    h_max, w_max = 0, 0
    for img_name in test_dataset:
        image_path = os.path.join(testset_folder, 'images', img_name)
        _img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if _img.shape[0] > h_max:
            h_max = _img.shape[0]
        if _img.shape[1] > w_max:
            w_max = _img.shape[1]

    h_max = (int(h_max / 32) + 1) * 32
    w_max = (int(w_max / 32) + 1) * 32

    priors = prior_box(image_sizes=(h_max, w_max),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)
else:
    target_size = 1600
    max_size = 2160
    priors = prior_box(image_sizes=(max_size, max_size),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)
```

### 初始化检测器

```python
from mindface.detection.runner import DetectionEngine, Timer
detection = DetectionEngine(nms_thresh=0.4, conf_thresh=0.02, iou_thresh=0.5, var=[0.1,0.2])
```

### 验证开始
```python
print('Predict box starting')
ave_time = 0
ave_forward_pass_time = 0
ave_misc = 0
for i, img_name in enumerate(test_dataset):
    image_path = os.path.join(testset_folder, 'images', img_name)

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    if cfg['val_origin_size']:
        resize = 1
        assert img.shape[0] <= h_max and img.shape[1] <= w_max
        image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    else:
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        assert img.shape[0] <= max_size and img.shape[1] <= max_size
        image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t

    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = Tensor(img)

    timers['forward_time'].start()
    boxes, confs, _ = network(img)
    timers['forward_time'].end()
    timers['misc'].start()
    detection.eval(boxes, confs, resize, scale, img_name, priors)
    timers['misc'].end()

    ave_time = ave_time + timers['forward_time'].diff + timers['misc'].diff
    ave_forward_pass_time = ave_forward_pass_time + timers['forward_time'].diff
    ave_misc = ave_misc + timers['misc'].diff
    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s sum_time: {:.4f}s'.format(i + 1, num_images,
                                                                                    timers['forward_time'].diff,
                                                                                    timers['misc'].diff,
                                                                                    timers['forward_time'].diff + timers['misc'].diff))
```

正常输出的结果为：
```text
Predict box starting
im_detect: 1/3226 forward_pass_time: 7.0862s misc: 0.0602s sum_time: 7.1464s
im_detect: 2/3226 forward_pass_time: 0.1645s misc: 0.0393s sum_time: 0.2037s
im_detect: 3/3226 forward_pass_time: 0.0638s misc: 0.0522s sum_time: 0.1160s
im_detect: 4/3226 forward_pass_time: 0.0648s misc: 0.0338s sum_time: 0.0986s
im_detect: 5/3226 forward_pass_time: 0.0656s misc: 0.0365s sum_time: 0.1021s
im_detect: 6/3226 forward_pass_time: 0.0648s misc: 0.0424s sum_time: 0.1071s
im_detect: 7/3226 forward_pass_time: 0.0648s misc: 0.0352s sum_time: 0.1000s
im_detect: 8/3226 forward_pass_time: 0.0648s misc: 0.0396s sum_time: 0.1044s
im_detect: 9/3226 forward_pass_time: 0.0648s misc: 0.0315s sum_time: 0.0963s
im_detect: 10/3226 forward_pass_time: 0.0647s misc: 0.0344s sum_time: 0.0991s
...
im_detect: 3226/3226 forward_pass_time: 0.0703s misc: 0.0358s sum_time: 0.1061s
```

### 计算并输出AP
为了让输出结果直观一些，我们调用`get_eval_result`函数计算ap。
```python
detection.get_eval_result()
```
输出结果为：
```text
Easy   Val Ap : 0.8862
Medium Val Ap : 0.8696
Hard   Val Ap : 0.7993
```
