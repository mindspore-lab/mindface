"""
train_config_ms1mv2_r50
"""
from easydict import EasyDict as edict

combination_cfg = edict({
# Contex
"device_memory_capacity": 2147483648.0,
"costmodel_gamma": 0.001,
"costmodel_beta": 280.0,

# Dataset
"data_dir": '/cache/data',
"top_dir_name": "faces_emore_train",
"num_classes": 85742,

# Model
"backbone": 'iresnet50',
"method": "arcface",
"num_features": 512,

# AMP
"amp_level": 'O2',

# dynamic loss scale
"loss_scale_type": 'fixed',
"loss_scale": 128.0,
# scale_factor: 2
# scale_window: 1000

# Train parameters
"epochs": 25,
"schedule": [10, 16, 21],
"gamma": 0.1,

"optimizer": 'sgd',
"learning_rate": 0.08,
"momentum": 0.9,
"weight_decay": 0.0001,
"filter_bias_and_bn": True,
"use_nesterov": False,

# Checkpoint
"save_checkpoint_steps": 60,
"keep_checkpoint_max": 20,
"train_dir": '/cache/output',
"resume": False,

})
