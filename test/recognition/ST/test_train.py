"""
Training face recognition models
"""
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import argparse

from mindface.recognition.datasets import create_dataset
from mindface.recognition.models import iresnet100, iresnet50, get_mbf, PartialFC
from mindface.recognition.loss import ArcFace
from mindface.recognition.runner import NetWithLoss, TrainingWrapper, lr_generator
from mindface.recognition.utils import read_yaml

import mindspore
from mindspore import nn
from mindspore import dtype as mstype
from mindspore import context

from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def test_train(args):
    mindspore.common.set_seed(args.seed)
    train_info = read_yaml(args.config)
    device_id = args.device_id
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device_target, save_graphs=False)

    if args.device_num > 1:
        if args.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True,
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
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              search_mode="recursive_programming")
    else:
        device_id = int(os.getenv('DEVICE_ID'))

    train_dataset = create_dataset(
        dataset_path=train_info['data_url'],
        do_train=True,
        repeat_num=1,
        batch_size=train_info['batch_size'],
        target=args.device_target,
        is_parallel=(args.device_num > 1)
            )

    step = train_dataset.get_dataset_size()
    lr = lr_generator(train_info['learning_rate'], train_info['schedule'],
                     train_info['gamma'], train_info['epochs'], steps_per_epoch=step)

    if train_info['backbone'] == 'mobilefacenet':
        net = get_mbf(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'iresnet50':
        net = iresnet50(num_features=train_info['num_features'])
    elif train_info['backbone'] == 'iresnet100':
        net = iresnet100(num_features=train_info['num_features'])
    else:
        raise NotImplementedError

    if train_info["resume"]:
        param_dict = load_checkpoint(train_info["resume"])
        load_param_into_net(net, param_dict)

    head = PartialFC(num_classes=train_info["num_classes"], world_size=args.device_num)

    loss_func = ArcFace(world_size=args.device_num)

    train_net = NetWithLoss(net.to_float(mstype.float16), head.to_float(mstype.float32), loss_func)
    optimizer = nn.SGD(params=train_net.trainable_params(), learning_rate=lr,
                       momentum=train_info['momentum'], weight_decay=train_info['weight_decay'])

    train_net = TrainingWrapper(train_net, optimizer)

    model = Model(train_net)

    config_ck = CheckpointConfig(save_checkpoint_steps=train_info["save_checkpoint_steps"], 
                                keep_checkpoint_max=train_info["keep_checkpoint_max"])

    ckpt_cb = ModelCheckpoint(prefix="_".join([train_info["method"], train_info['backbone']]), 
                                config=config_ck, directory=train_info['train_url'])
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [ckpt_cb, time_cb, loss_cb]

    if args.device_num == 1:
        model.train(train_info['epochs'], train_dataset,
                    callbacks=cb, dataset_sink_mode=False)
    elif args.device_num > 1 and get_rank() % 8 == 0:
        model.train(train_info['epochs'], train_dataset,
                    callbacks=cb, dataset_sink_mode=False)
    else:
        model.train(train_info['epochs'], train_dataset, dataset_sink_mode=False)
    print("============== Starting Training ==============")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training')
    # configs
    parser.add_argument('--config', default='configs/train_config_ms1m.yaml', type=str, help='output path')
    # Optimization options
    parser.add_argument('--device_target', type=str, default='GPU', choices=['GPU', 'Ascend'])
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--device_id', type=int, default=0)
    # Random seed
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    test_train(args)