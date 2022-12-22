"""
Training face recognition models.
"""
import os
import argparse
import time

from datasets import create_dataset
from models import iresnet100, iresnet50, get_mbf, PartialFC, vit_t, vit_s, vit_b, vit_l
from loss import ArcFace
from runner import Network, lr_generator
from utils import C2netMultiObsToEnv, EnvToObs
from configs import config_combs
from optim import create_optimizer

import mindspore
from mindspore import context

from mindspore import FixedLossScaleManager, DynamicLossScaleManager
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters
from mindspore.train.serialization import load_checkpoint, load_param_into_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')

    # configs
    parser.add_argument('--config', default='casia_vit_b', type=str, help='output path')
    parser.add_argument('--multi_data_url', default='/cache/data/',
                        type=str, help='path to multi dataset')
    parser.add_argument('--train_url', default= '/cache/output/',
                        type=str, help='output folder to save/load')
    parser.add_argument('--ckpt_url', default= '/cache/checkpoint.ckpt',
                        type=str, help='output folder to save/load')

    # Optimization options
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize (default: 64)')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['GPU', 'Ascend'])
    parser.add_argument('--running_mode', type=str, default='PYNATIVE',
                        choices=['GRAPH', 'PYNATIVE'])

    # Random seed
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()

    mindspore.common.set_seed(args.seed)

    # train_info = read_yaml(os.path.join(os.getcwd(), args.config))
    train_info = config_combs[args.config]

    print(args, train_info)

    os.makedirs(train_info["data_dir"], exist_ok=True)
    os.makedirs(train_info["train_dir"], exist_ok=True)

    if args.running_mode == "GRAPH":
        DATASET_SINK_MODE = True
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target, save_graphs=False)
    elif args.running_mode == "PYNATIVE":
        DATASET_SINK_MODE = False
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=args.device_target, save_graphs=False)
    else:
        raise NotImplementedError

    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    print("device_num: ", device_num, "local_rank: ", local_rank)

    if device_num > 1:
        if args.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True
                )
            cost_model_context.set_cost_model_context(
                    device_memory_capacity = train_info["device_memory_capacity"],
                    costmodel_gamma=train_info["costmodel_gamma"],
                    costmodel_beta=train_info["costmodel_beta"]
                )
            set_algo_parameters(elementwise_op_strategy_follow=True)
            init()
        elif args.device_target == 'GPU':
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              search_mode="recursive_programming")
    else:
        device_id = int(os.getenv('DEVICE_ID'))

    if args.device_target == 'Ascend':
        print("Downloading dataset ...")
        if device_num ==1:
            C2netMultiObsToEnv(args.multi_data_url, train_info['data_dir'])
        else:
            if local_rank % 8==0:
                C2netMultiObsToEnv(args.multi_data_url, train_info['data_dir'])

            while not os.path.exists("/cache/download_input.txt"):
                time.sleep(1)
    else:
        pass

    train_dataset = create_dataset(
        dataset_path=os.path.join(train_info['data_dir'], train_info['top_dir_name']),
        do_train=True,
        repeat_num=1,
        # batch_size=train_info['batch_size'],
        batch_size=args.batch_size,
        target=args.device_target,
        is_parallel=(device_num > 1)
            )

    step = train_dataset.get_dataset_size()
    assert step > 0, "Loading dataset error"

    lr = lr_generator(train_info['learning_rate'], train_info['schedule'],
                     train_info['gamma'], train_info['epochs'], steps_per_epoch=step)

    if train_info['backbone'] == 'mobilefacenet':
        net = get_mbf(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'iresnet50':
        net = iresnet50(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'iresnet100':
        net = iresnet100(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'vit_t':
        net = vit_t(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'vit_s':
        net = vit_s(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'vit_b':
        net = vit_b(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'vit_l':
        net = vit_l(num_features=train_info['num_features'])
    else:
        raise NotImplementedError

    if train_info["resume"]:
        if args.device_target == 'Ascend':
            train_info["resume"] = args.ckpt_url
        else:
            pass
        param_dict = load_checkpoint(train_info["resume"])
        load_param_into_net(net, param_dict)

    head = PartialFC(num_classes=train_info["num_classes"], world_size=device_num)

    train_net = Network(net, head)

    loss_func = ArcFace(world_size=device_num)

    loss_scale_manager = None
    loss_scale_for_opt = 1.0

    if train_info["amp_level"] != "O3": # user for O0 or O2
        if train_info["loss_scale"]>1.0: # adopt loss scale
            if train_info["loss_scale_type"] == "fixed":
                loss_scale_manager = FixedLossScaleManager(
                    loss_scale=train_info["loss_scale"],
                    drop_overflow_update=False
                    )
                # drop_overflow_update=False, loss_scale for opt changes
                loss_scale_for_opt = train_info["loss_scale"]
            else:
                loss_scale_manager = DynamicLossScaleManager(
                    init_loss_scale=train_info["loss_scale"],
                    scale_factor=train_info["scale_factor"],
                    scale_window=train_info["scale_window"]
                    )
        print(train_info["amp_level"], train_info["loss_scale"], train_info["loss_scale_type"])

    optimizer = create_optimizer(train_net.trainable_params(),
                            opt=train_info['optimizer'],
                            lr=lr,
                            weight_decay=train_info['weight_decay'],
                            nesterov=train_info["use_nesterov"],
                            filter_bias_and_bn=train_info["filter_bias_and_bn"],
                            loss_scale = loss_scale_for_opt,
                            )

    if loss_scale_manager != None:
        model = Model(train_net, loss_fn=loss_func, optimizer=optimizer,
                      amp_level=train_info["amp_level"], loss_scale_manager=loss_scale_manager)
    else:
        model = Model(train_net, loss_fn=loss_func, optimizer=optimizer,
                      amp_level=train_info["amp_level"])

    config_ck = CheckpointConfig(save_checkpoint_steps=train_info["save_checkpoint_steps"],
                                keep_checkpoint_max=train_info["keep_checkpoint_max"])

    ckpt_cb = ModelCheckpoint(prefix="_".join([train_info["method"], train_info['backbone']]),
                                config=config_ck, directory=train_info['train_dir'])
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [ckpt_cb, time_cb, loss_cb]

    if device_num == 1:
        model.train(train_info['epochs'], train_dataset,
                    callbacks=cb, dataset_sink_mode=DATASET_SINK_MODE)
    elif device_num > 1 and get_rank() % 8 == 0:
        model.train(train_info['epochs'], train_dataset,
                    callbacks=cb, dataset_sink_mode=DATASET_SINK_MODE)
    else:
        model.train(train_info['epochs'], train_dataset, dataset_sink_mode=DATASET_SINK_MODE)

    if args.device_target == 'Ascend':
        # if local_rank == 0:
        EnvToObs(train_info['train_dir'], args.train_url)
    else:
        pass
