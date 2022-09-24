"""Config for train and eval."""

cfg_res50 = {
    'name': 'ResNet50',
    'device_target': "GPU",
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'batch_size': 8,
    'num_workers': 1,
    'num_anchor': 29126,
    'ngpu': 1,
    'image_size': 840,
    'in_channel': 256,
    'out_channel': 256,
    'match_thresh': 0.35,
    'num_classes' : 2,

    # opt
    'optim': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,

    # seed
    'seed': 1,

    # lr
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'lr_type': 'dynamic_lr',
    'initial_lr': 0.01,
    'warmup_epoch': 5,
    'gamma': 0.1,

    # checkpoint
    'ckpt_path': './checkpoint/',
    'save_checkpoint_steps': 2000,
    'keep_checkpoint_max': 1,
    'resume_net': None,

    # dataset
    'training_dataset': 'data/WiderFace/train/label.txt',
    'pretrain': True,
    'pretrain_path': 'pretrained/resnet50_ascend_v170_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt',

    # val
    'val_model': 'pretrained/RetinaFace_ResNet50.ckpt',
    'val_dataset_folder': 'data/WiderFace/val/',
    'val_origin_size': False,
    'val_confidence_threshold': 0.02,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': False,
    'val_predict_save_folder': './widerface_result',
    'val_gt_dir': 'data/WiderFace/ground_truth',
}
