"""
train_config_casia_vit_s
"""
from easydict import EasyDict as edict

combination_cfg = edict({
# Contex
"device_memory_capacity": 2147483648.0,
"costmodel_gamma": 0.001,
"costmodel_beta": 280.0,

# Dataset
"data_dir": '/cache/data',
"top_dir_name": "faces_webface_112x112_train",
"num_classes": 10572,

# Model
"backbone": 'vit_s',
"method": "arcface",
"num_features": 512,

# AMP
"amp_level": "O2",

# dynamic loss scale
"loss_scale_type": "fixed",
"loss_scale": 128.0,
# "scale_factor": 2,
# "scale_window": 1000,

"epochs": 25,
"schedule": [10, 16, 21],
"gamma": 0.1,

"optimizer": "adamw",
"learning_rate": 0.0001,
"weight_decay": 0.025,
"filter_bias_and_bn": True,
"use_nesterov": False,

# Checkpoint
"save_checkpoint_steps": 60,
"keep_checkpoint_max": 5,
"train_dir": '/cache/output',
"resume": False,

})
