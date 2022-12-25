# 了解模型配置

`mindface`套件可以通过python的`argparse`库和`pyyaml`库解析模型的yaml文件来进行参数的配置，下面我们以MobileNet025模型为例，解释如何配置相应的参数。


## 基础环境

1. 参数说明

- device_target：训练平台，可选CPU平台（CPU）、GPU平台（GPU）、昇腾平台（Ascend）。
- mode：动态图模式（'Pynative'）或者静态图模式('Graph')。
- ngpu：训练的GPU数量。

2. yaml文件样例

```text
'device_target': "GPU"
'mode': 'Graph'
'ngpu': '1'
...
```

3. 对应代码示例

```python
def train(cfg):
    if cfg['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])

    # rank=0
    if cfg['device_target'] == "Ascend":
        device_num = cfg['nnpu']
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
            rank = get_rank()
            print(f"The rank ID of current device is {rank}.")
        else:
            context.set_context(device_id=cfg['device_id'])
    elif cfg['device_target'] == "GPU":
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            rank = get_rank()
            print(f"The rank ID of current device is {rank}.")
    ...
```


## 数据集

1. 参数说明

- variance：方差。
- training_dataset：训练集地址。
- clip：是否对图片做裁剪。
- batch_size：训练批次大小。
- num_workers：数据集加载数据的线程数量。
- num_classes：类别数量。
- image_size：训练图像大小。
- match_thresh：匹配框阈值。

2. yaml文件样例

```text
'variance': [0.1, 0.2]
'training_dataset': 'data/WiderFace/train/label.txt'
'clip': False
'batch_size': 8
'num_workers': 1
'num_classes' : 2
'image_size': 640
'match_thresh': 0.35
...
```

3. 对应的代码示例

```python
def train(cfg):
    ...
    batch_size = cfg['batch_size']
    ...
    training_dataset = cfg['training_dataset']
    num_classes = cfg['num_classes']
    ...
    ds_train = create_dataset(training_dataset, cfg['variance'], cfg['match_thresh'], cfg['image_size'], clip, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())
    ...
```


## 模型

1. 参数说明

- name：backbone名称。
- in_channel：FPN输入通道数。
- out_channel：FPN输出通道数。
- grad_clip：梯度截断。
- pretrain：backbone是否加载预训练模型。
- pretrain_path：backbone预训练权重路径。
- ckpt_path：模型保存路径。
- save_checkpoint_steps：保存检查点迭代。
- keep_checkpoint_max：预留检查点数量。
- resume_net：resume网络，默认为~。

2. yaml文件样例

```text
'name': 'MobileNet025'
'in_channel': 32
'out_channel': 64
'grad_clip': False
'pretrain': False
'pretrain_path': ~
'ckpt_path': './checkpoint/'
'save_checkpoint_steps': 2000
'keep_checkpoint_max': 3
'resume_net': ~
...
```

3. 对应的代码示例

```python
def train(cgf):
    ...
    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    backbone.set_train(True)

    if  cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained= cfg['pretrain_path']
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(backbone, param_dict)
        print(f"Load RetinaFace_{cfg['name']} from [{cfg['pretrain_path']}] done.")

    net = RetinaFace(phase='train', backbone=backbone, in_channel=cfg['in_channel'], out_channel=cfg['out_channel'])
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print(f"Resume Model from [{cfg['resume_net']}] Done.")
    ...
    net = TrainingWrapper(net, opt, grad_clip=cfg['grad_clip'])

    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'], keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)
    ...
```


## 损失函数

1. 参数说明

- num_anchor：anchor数量。
- loc_weight：Bbox回归损失权重。
- class_weight：置信度/类回归损失权重。
- landm_weight：landmark回归损失权重。

2. yaml文件样例

```text
'num_anchor': 16800
'loc_weight': 2.0
'class_weight': 1.0
'landm_weight': 1.0
...
```

3. 对应的代码示例

```python
def train(cfg):
    ...
    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio)
    ...
    net = RetinaFaceWithLossCell(net, multibox_loss, loc_weight = 2.0, class_weight = 1.0, landm_weight = 1.0)
    ...
```


## 学习率策略

1. 参数说明

- epoch：训练轮次。
- decay1：首次权重衰减的轮次数。
- decay2：二次权重衰减的轮次数。
- lr_type：学习率类型。
- initial_lr：学习率。
- warmup_epoch：warmup，-1表示无warmup。
- gamma：学习率衰减比。

2. yaml文件样例

```text
'epoch': 120
'decay1': 70
'decay2': 90
'lr_type': 'dynamic_lr'
'initial_lr': 0.01
'warmup_epoch': 5
'gamma': 0.1
...
```

3. 对应的代码示例

```python
def train(cfg):
    ...
    max_epoch = cfg['epoch']
    ...
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    ...
    stepvalues = (cfg['decay1'], cfg['decay2'])
    ...
    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch, warmup_epoch=cfg['warmup_epoch'], lr_type1=lr_type)
    ...
```


## 优化器

1. 参数说明

- optim：优化器类型。
- momentum：优化器动量。
- weight_decay：优化器权重衰减。 

2. yaml文件样例

```text
'optim': 'sgd'
'momentum': 0.9
'weight_decay': 5e-4
...
```

3. 对应的代码示例

```python
def train(cfg):
    ...
    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum, weight_decay, loss_scale=1)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=weight_decay, loss_scale=1)
    else:
        raise ValueError('optim is not define.')
    ...
```
