# Learn From Config

```
cfg_mobile025 = {
    'name': 'MobileNet025',# 骨干名称
    'device_target': "GPU",# 训练平台
    'mode': 'Graph',# 模式
    'variance': [0.1, 0.2],# 方差
    'clip': False,# 裁剪
    'loc_weight': 2.0,# Bbox回归损失权重
    'class_weight': 1.0,# 置信度/类回归损失权重
    'landm_weight': 1.0,# 地标回归损失权重
    'batch_size': 8, # 训练批次大小
    'num_workers': 1,# 数据集加载数据的线程数量
    'num_anchor': 16800,# 矩形框数量，取决于图片大小
    'ngpu': 1,# 训练的GPU数量
    'image_size': 640,# 训练图像大小
    'in_channel': 32,# DetectionHead输入通道
    'out_channel': 64,# DetectionHead输出通道
    'match_thresh': 0.35,# 匹配框阈值
    'num_classes' : 2,# 类别数量

    # opt
    'optim': 'sgd',# 优化器类型
    'momentum': 0.9,# 优化器动量
    'weight_decay': 5e-4,# 优化器权重衰减

    # seed
    'seed': 1,# 随机种子

    # lr
    'epoch': 120,# 训练轮次
    'decay1': 70,# 首次权重衰减的轮次数
    'decay2': 90,# 二次权重衰减的轮次数
    'lr_type': 'dynamic_lr',# 学习率类型
    'initial_lr': 0.01,# 学习率
    'warmup_epoch': 5,# warmup，-1表示无warmup
    'gamma': 0.1,# 学习率衰减比

    # checkpoint
    'ckpt_path': './checkpoint/',# 模型保存路径
    'save_checkpoint_steps': 2000,# 保存检查点迭代
    'keep_checkpoint_max': 3,# 预留检查点数量
    'resume_net': None,# resume网络，默认为None

    # dataset
    'training_dataset': 'data/WiderFace/train/label.txt',# 训练集地址
    'pretrain': False,# 是否基于预训练骨干进行训练
    'pretrain_path': None,# 是否基于预训练骨干进行训练

    # val
    'val_model': 'pretrained/RetinaFace_MobileNet025.ckpt',# 验证模型路径
    'val_dataset_folder': 'data/WiderFace/val/',# 验证数据集路径
    'val_origin_size': False,# 是否使用全尺寸验证
    'val_confidence_threshold': 0.02,# 验证置信度阈值
    'val_nms_threshold': 0.4,# 验证NMS阈值
    'val_iou_threshold': 0.5,# 验证IOU阈值
    'val_save_result': False,# 是否保存结果
    'val_predict_save_folder': './widerface_result',# 结果保存路径
    'val_gt_dir': 'data/WiderFace/ground_truth',# 验证集ground_truth路径
}
```
