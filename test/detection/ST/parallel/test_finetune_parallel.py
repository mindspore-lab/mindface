import math
import os
import download
import mindspore

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net


from mindface.detection.loss import MultiBoxLoss
from mindface.detection.datasets import create_dataset
from mindface.detection.utils.lr_schedule import adjust_learning_rate 

from mindface.detection.models import RetinaFace, RetinaFaceWithLossCell, resnet50, mobilenet025
from mindface.detection.runner import read_yaml, TrainingWrapper

def test_finetune_parallel(cfg, finetune_epochs):
    """test_finetune"""
    #set seed
    mindspore.common.seed.set_seed(42)

    #set mode
    if cfg['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])

    # create dataset
    # set parameters
    url='https://retinaface.obs.cn-central-231.xckpjs.com/data/WiderFace.zip'
    path='./data'
    if not os.path.exists(path): 
        os.makedirs(path)
    download.download(url, path, kind="zip", replace=True)
    data_dir = './data/WiderFace/train/label.txt'
    batch_size = cfg['batch_size']
    ds_train = create_dataset(data_dir, cfg, batch_size, multiprocessing=True, num_worker=2)
    assert ds_train.get_batch_size() == batch_size

        
    #set learning rate schedule
    steps_per_epoch = math.ceil(ds_train.get_dataset_size())
    lr = adjust_learning_rate(cfg['initial_lr'], cfg['gamma'], (cfg['decay1'], cfg['decay2']), steps_per_epoch, cfg['epoch'],
                                warmup_epoch=cfg['warmup_epoch'], lr_type1='dynamic_lr')

   
    #build model
    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    backbone.set_train(True)
    
    net  = RetinaFace(phase='train', backbone = backbone, cfg=cfg)
    net.set_train(True)

    
    # load checkpoint
    pretrain_model_path = cfg['resume_net']
    param_dict_retinaface = load_checkpoint(pretrain_model_path)
    load_param_into_net(net, param_dict_retinaface)
    print(f'Resume Model from [{pretrain_model_path}] Done.')

    # set loss
    multibox_loss = MultiBoxLoss(num_classes = cfg['num_classes'], num_boxes = cfg['num_anchor'], neg_pre_positive=7)

    # set optimazer
    opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=0.9,
                                weight_decay=5e-4, loss_scale=1)
    
    # add loss and optimazer  
    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
    net = TrainingWrapper(net, opt)

    model = Model(net)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                    keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(finetune_epochs, ds_train, callbacks=callback_list, dataset_sink_mode=False)

if __name__ == '__main__':

    # read the configs
    cfg_res50 = read_yaml('mindface/detection/configs/RetinaFace_resnet50.yaml')
    cfg_mobile025 = read_yaml('mindface/detection/configs/RetinaFace_mobilenet025.yaml')
    finetune_epochs = 10
    
    cfg_res50['ngpu'] = 2
    cfg_mobile025['ngpu'] = 2

    #test retinaface_resnet50_finetune

    #pynative mode
    cfg_res50['mode'] = 'Pynative'
    test_finetune_parallel(cfg_res50, finetune_epochs)
    
    #graph mode
    cfg_res50['mode'] = 'Graph'
    test_finetune_parallel(cfg_res50, finetune_epochs)
    
    #test retinaface_mobilenet025_finetune
    #pynative mode
    cfg_mobile025['mode'] = 'Pynative'
    test_finetune_parallel(cfg_mobile025, finetune_epochs)
    
    #graph mode
    cfg_mobile025['mode'] = 'Graph'
    test_finetune_parallel(cfg_mobile025, finetune_epochs)

