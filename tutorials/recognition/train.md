# 训练模型

## 配置训练参数

```
## 训练参数设置
epochs: 25                # 训练轮数
num_classes: 10572        # 类别总数
batch_size: 256           # 每批图像数量
learning_rate: 0.02       # 学习率
schedule: [10, 16, 21]    # 学习率衰减节点
gamma: 0.1                # 学习率衰减比例
momentum: 0.9             # SGD优化器参数：动量
weight_decay: 0.0001      # SGD优化器参数：权重衰减
backbone: 'iresnet50'     # 骨干网络 （'mobilefacenet', 'iresnet50', 'iresnet100' 等）
train_url: '.'            # 模型保存位置
resume: False             # 预训练模型路径

## 数据集参数
data_url: "/path/to/dataset"   # 数据集路径（此路径下的数据集需按照规定格式组织）
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