# 模型配置

MindFace 套件模型配置文件示例如下，我们在 config 文件夹配置各项训练参数，本教程将介绍算法配置文件每个参数的实际含义。

```python
# Contex
costmodel_gamma: 0.001                     # strategy-searching算法参数gamma
costmodel_beta: 280.0                      # strategy-searching算法参数beta

# Dataset
"data_dir": '/cache/data',
"top_dir_name": "faces_emore_train",       # 数据集路径（此路径下的数据集需按照规定格式组织）
num_classes: 85742                         # 类别总数（需根据数据集的选择调整）

# Model
backbone: 'iresnet50'                      # 骨干网络 （如'mobilefacenet', 'iresnet50', 'iresnet100', 'vit-t' 等）
method: "arcface"                          # 损失函数类型
num_features: 512                          # 提取出的图像特征维数
loss_scale_type: "fixed",                  # loss scale 类型
loss_scale: 8.0,                           # loss放大倍数
amp_level: "O2",                           # 训练精度类型

# Train parameters
epochs: 10                                 # 训练轮数
batch_size: 256                            # 每批图像数量
schedule: [4, 6, 8]                        # 学习率衰减节点
gamma: 0.1                                 # 学习率衰减比例
optimizer: "adamw"                         # 优化器的选择
learning_rate: 0.0001                      # 学习率
weight_decay: 0.025                        # 优化器参数：权重衰减

# Checkpoint
save_checkpoint_steps: 60                  # 每次模型保存间隔的step数
keep_checkpoint_max: 20                    # 最多保存模型的数量
train_url: '.'                             # 模型保存位置
resume: False                              # 加载模型路径
```

在训练中常用来调整训练效果的参数包括学习率的大小、学习率的优化策略、训练轮数及每批图像数量等，针对可能出现的训练不成功或 loss 不收敛的情况，也可以调整训练精度的类型，其他参数可针对训练时的实际情况自行选择和调整。
