'''
Pipeline: 本部分针对Retinaface检测模型和arcface识别模型, 利用前五部分设计的内容, 构建训练、验证、测试的流程
'''

import argparse
import os
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor

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

from dataset import create_dataset
from backbone_head import iresnet100, get_mbf, get_mbf_large, PartialFC
from loss import ArcFace, SoftMaxCE


def lr_generator(lr_init, total_epochs, steps_per_epoch):
    lr_each_step = []
    for i in range(total_epochs):
        if i in args.schedule:
            lr_init *= args.gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)

# clip_grad
GRADIENT_CLIP_TYPE = 1
GRANDIENT_CLIP_VALUE = 1.0

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


class ArcFaceWithLossCell(nn.Cell):
    def __init__(self, backbone, cfg):
        super(ArcFaceWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone.to_float(mstype.float16)
        self.head = PartialFC(
            num_classes=cfg.num_classes, world_size=cfg.device_num).to_float(mstype.float32)
        self.margin_softmax = ArcFace(world_size=cfg.device_num)
        self.loss = SoftMaxCE(world_size=cfg.device_num)

    def construct(self, data, label):
        out = self._backbone(data)
        out = self.head(out)
        out = self.margin_softmax(out, label)
        loss = self.loss(out, label)
        return loss


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


def train(args):
    train_epoch = args.epochs
    target = args.device_target

    ## Dynamic Graph
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=target, save_graphs=False)
    
    ## Static Graph
    # context.set_context(mode=context.GRAPH_MODE,
    #                     device_target=target, save_graphs=False)

    device_id = args.device_id

    if args.device_num > 1:
        if target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, )
            cost_model_context.set_cost_model_context(
                device_memory_capacity=32.0 * 1024.0 * 1024.0 * 1024.0, costmodel_gamma=0.001, costmodel_beta=280.0)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            init()
        elif target == 'GPU':
            init()
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
    else:
        device_id = int(os.getenv('DEVICE_ID'))

    train_dataset = create_dataset(
        dataset_dir=args.data_url, do_train=True, repeat_num=1, batch_size=args.batch_size, target=target)

    step = train_dataset.get_dataset_size()
    lr = lr_generator(args.lr, train_epoch, steps_per_epoch=step)
    
    if args.backbone == 'iresnet100':
        net = iresnet100()
    elif args.backbone == 'mobilenet-small':
        net = get_mbf(False, 512)
    elif args.backbone == 'mobilenet-large':
        net = get_mbf_large(False, 512)

    train_net = ArcFaceWithLossCell(net, args)
    optimizer = nn.SGD(params=train_net.trainable_params(), learning_rate=lr,
                       momentum=args.momentum, weight_decay=args.weight_decay)

    train_net = TrainingWrapper(train_net, optimizer)

    model = Model(train_net)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=60, keep_checkpoint_max=20)
    ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
                              directory=args.train_url)
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [ckpt_cb, time_cb, loss_cb]
    
    ## Dynamic Graph
    if args.device_num == 1:
        model.train(train_epoch, train_dataset,
                    callbacks=cb, dataset_sink_mode=False)
    elif args.device_num > 1 and get_rank() % 8 == 0:
        model.train(train_epoch, train_dataset,
                    callbacks=cb, dataset_sink_mode=False)
    else:
        model.train(train_epoch, train_dataset, dataset_sink_mode=False)
    
    ## Static Graph
    # if args.device_num == 1:
    #     model.train(train_epoch, train_dataset,
    #                 callbacks=cb, dataset_sink_mode=True)
    # elif args.device_num > 1 and get_rank() % 8 == 0:
    #     model.train(train_epoch, train_dataset,
    #                 callbacks=cb, dataset_sink_mode=True)
    # else:
    #     model.train(train_epoch, train_dataset, dataset_sink_mode=True)


if __name__ == '__main__':
    mindspore.common.set_seed(2022)

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--train_url', default='.', type=str, help='output path')
    parser.add_argument('--data_url', default='data path', type=str)

    parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
    # parser.add_argument('--num_classes', default=10572, type=int, metavar='N', help='num of classes')
    parser.add_argument('--num_classes', default=85742, type=int, metavar='N', help='num of classes')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='train batchsize (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 16, 21], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.02, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--backbone', type=str, default='iresnet100', choices=['mobilenet-small', 'mobilenet-large'])

    parser.add_argument('--device_target', type=str, default='GPU', choices=['GPU', 'Ascend'])
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--modelarts', action="store_true", help="using modelarts")

    args = parser.parse_args()

    train(args=args)
