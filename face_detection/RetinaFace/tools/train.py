"""Train Retinaface_resnet50ormobilenet0.25."""

import argparse
import math
import mindspore
import os
import sys

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

base_path = os.getcwd()
sys.path.append(base_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from configs.RetinaFace_mobilenet import cfg_mobile025
from configs.RetinaFace_resnet50 import cfg_res50
from utils.loss import MultiBoxLoss
from datasets.dataset import create_dataset
from utils.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr

from model.retinaface import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper 
from backbone.resnet import resnet50
from backbone.mobilenet import mobilenet025


def train(cfg):
    
    mindspore.common.seed.set_seed(cfg['seed'])
   
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg['device_target'])

    # rank=0
    if cfg['device_target'] == "Ascend":
        device_num = cfg['nnpu']
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            # rank = get_rank()
        else:
            context.set_context(device_id=cfg['device_id'])
    elif cfg['device_target'] == "GPU":
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            # rank = get_rank()

    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']

    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    training_dataset = cfg['training_dataset']
    num_classes = cfg['num_classes']
    negative_ratio = 7
    stepvalues = (cfg['decay1'], cfg['decay2'])

    ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    backbone.set_train(True)

    if cfg['name'] == 'ResNet50' and cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained_res50 = cfg['pretrain_path']
        param_dict_res50 = load_checkpoint(pretrained_res50)
        load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_res50))
    elif cfg['name'] == 'MobileNet025' and cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained_mobile025 = cfg['pretrain_path']
        param_dict_mobile025 = load_checkpoint(pretrained_mobile025)
        load_param_into_net(backbone, param_dict_mobile025)
        print('Load mobilenet0.25 from [{}] done.'.format(pretrained_mobile025))

    net = RetinaFace(phase='train', backbone=backbone, cfg=cfg)
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print('Resume Model from [{}] Done.'.format(cfg['resume_net']))

    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)

    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                              warmup_epoch=cfg['warmup_epoch'], lr_type1=lr_type)

    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                               weight_decay=weight_decay, loss_scale=1)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt)

    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--backbone_name', type=str, default='ResNet50',
                        help='backbone name')
    args_opt = parser.parse_args()

    if args_opt.backbone_name == 'ResNet50':
        config = cfg_res50
    elif args_opt.backbone_name == 'MobileNet025':
        config = cfg_mobile025
    train(cfg=config)
