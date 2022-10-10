# 训练模型

## 配置训练参数
我们在config文件夹中通过yaml文件配置各项训练参数
```
# Contex
device_memory_capacity: 2147483648.0                    # 设备内存容量(Ascend平台)                  
costmodel_gamma: 0.001                                  # strategy-searching算法参数gamma
costmodel_beta: 280.0                                   # strategy-searching算法参数beta

# Dataset
data_url: "/path/to/dataset"                            # 数据集路径（此路径下的数据集需按照规定格式组织）
num_classes: 10572                                      # 类别总数

# Model
backbone: 'iresnet50'                                   # 骨干网络 （'mobilefacenet', 'iresnet50', 'iresnet100' 等）
method: "arcface"                                       # 损失函数类型
num_features: 512                                       # 提取出的图像特征维数

# Train parameters
epochs: 25                                              # 训练轮数
batch_size: 256                                         # 每批图像数量
learning_rate: 0.02                                     # 学习率
schedule: [10, 16, 21]                                  # 学习率衰减节点
gamma: 0.1                                              # 学习率衰减比例
momentum: 0.9                                           # SGD优化器参数：动量
weight_decay: 0.0001                                    # SGD优化器参数：权重衰减

# Checkpoint
save_checkpoint_steps: 60                               # 每次模型保存间隔的step数
keep_checkpoint_max: 20                                 # 最多保存模型的数量
train_url: '.'                                          # 模型保存位置
resume: False                                           # 预训练模型路径
```

## 执行训练

MindFace可以支持单卡和多卡训练，并且可以在GPU与Ascend平台上运行

### 单卡训练

GPU平台
```
sh scripts/run_standalone_train_gpu.sh  /path/to/configs
```
Ascend平台
```
sh scripts/run_standalone_train.sh  /path/to/configs
```

### 分布式训练

GPU平台
```
sh scripts/run_distribute_train_gpu.sh /path/to/configs rank_size
```
Ascend平台
```
sh scripts/run_distribute_train.sh rank_size /path/to/configs
```