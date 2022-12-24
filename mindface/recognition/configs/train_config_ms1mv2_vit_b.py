"""
train_config_ms1mv2_vit_b
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
"backbone": 'vit_b',
"method": "arcface",
"loss_scale_type": "fixed",
"scale_factor": 4,
"scale_window": 3000,
"loss_scale": 32.0,
"amp_level": "O2",
"num_features": 512,

# Train parameters
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
