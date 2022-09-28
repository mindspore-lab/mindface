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
import numpy as np
import yaml
import argparse

import mindspore
import mindspore.nn as nn

import mindspore.ops as ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindspore import context, Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

from mindspore.train.model import Model, ParallelMode
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from models.iresnet import iresnet100, iresnet50
from models.mobilefacenet import get_mbf
from models.partialFC import PartialFC
from datasets.face_dataset import create_dataset
from loss.arcface_loss import ArcFace
from loss.ce_loss import SoftMaxCE

mindspore.common.set_seed(2022)

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def lr_generator(lr_init, schedule, gamma, total_epochs, steps_per_epoch):
    """lr_generator
    """
    lr_each_step = []
    for i in range(total_epochs):
        if i in schedule:
            lr_init *= gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)


class MyNetWithLoss(nn.Cell):
    """
    WithLossCell
    """

    def __init__(self, backbone, num_classes, device_num):
        super(MyNetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone.to_float(mstype.float16)
        self.fc = PartialFC(num_classes=num_classes,
                                  world_size=device_num).to_float(mstype.float32)
        self.margin_softmax = ArcFace(world_size=device_num)
        self.loss = SoftMaxCE(world_size=device_num)
        # self.L2Norm = ops.L2Normalize(axis=1)

    def construct(self, data, label):
        out = self._backbone(data)
        out_fc = self.fc(out)
        out_fc = self.margin_softmax(out_fc, label)
        loss = self.loss(out_fc, label)

        return loss


# clip_grad
GRADIENT_CLIP_TYPE = 1
GRANDIENT_CLIP_VALUE = 0.7

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        class_list = [mindspore.context.ParallelMode.DATA_PARALLEL,
                      mindspore.context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                optimizer.parameters, mean, degree)
        self.hyper_map = mindspore.ops.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)

        grads = self.hyper_map(
            F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRANDIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')

    # configs
    parser.add_argument('--config', default='configs\train_config_ms1m.yaml', type=str,
                        help='output path')
    
    # Optimization options
    parser.add_argument('--device_target', type=str,
                    default='GPU', choices=['GPU', 'Ascend'])
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--modelarts', action="store_true", help="using modelarts")
    args = parser.parse_args()

    train_info = read_yaml(args.config)

    device_id = args.device_id

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device_target, save_graphs=False)

    if args.device_num > 1:
        if args.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              )
            cost_model_context.set_cost_model_context(device_memory_capacity=2.0 * 1024.0 * 1024.0 * 1024.0,
                                                      costmodel_gamma=0.001,
                                                      costmodel_beta=280.0)
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
    lr = lr_generator(train_info['learning_rate'], train_info['schedule'], train_info['gamma'], train_info['epochs'], steps_per_epoch=step)

    if train_info['backbone'] == 'mobilefacenet':
        net = get_mbf(False, 512)
    elif train_info['backbone'] == 'iresnet50':
        net = iresnet50()
    elif train_info['backbone'] == 'iresnet100':
        net = iresnet100()
    else:
        raise NotImplementedError

    train_net = MyNetWithLoss(net, train_info['num_classes'], args.device_num)
    optimizer = nn.SGD(params=train_net.trainable_params(), learning_rate=lr,
                       momentum=train_info['momentum'], weight_decay=train_info['weight_decay'])

    if train_info["resume"]:
        param_dict = load_checkpoint(train_info["resume"])
        load_param_into_net(train_net, param_dict)

    train_net = TrainingWrapper(train_net, optimizer)

    model = Model(train_net)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=60, keep_checkpoint_max=20)

    ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
                              directory=train_info['train_url'])
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
