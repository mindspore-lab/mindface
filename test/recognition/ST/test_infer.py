"""
Training face recognition models.
"""
import os
import argparse

from mindface.recognition.datasets import create_dataset
from mindface.recognition.models import iresnet100, iresnet50, get_mbf, PartialFC, vit_t, vit_s, vit_b, vit_l
from mindface.recognition.loss import ArcFace
from mindface.recognition.runner import Network, lr_generator
from mindface.recognition.utils import read_yaml
from mindcv import create_optimizer

import mindspore as ms
from mindspore import context
import numpy as np
from mindspore import FixedLossScaleManager, DynamicLossScaleManager
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l

def main():
    config = '../../../configs/train_config_casia_vit_t.yaml'
    batch_size = 128
    device_target = 'Ascend'
    running_mode = 'GRAPH'
    seed = 2022
    model_name = iresnet50
    num_features = 512
    num_classes = 10572
    ms.common.set_seed(seed)

    train_info = read_yaml(os.path.join(os.getcwd(), config))

    if running_mode == "GRAPH":
        DATASET_SINK_MODE = True
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=device_target, save_graphs=False)
    elif running_mode == "PYNATIVE":
        DATASET_SINK_MODE = False
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=device_target, save_graphs=False)
    else:
        raise NotImplementedError

    device_num = 1

    if device_num > 1:
        if device_target == 'Ascend':
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
        elif device_target == 'GPU':
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              search_mode="recursive_programming")
    else:
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') is not None else 0
    if model_name == 'iresnet50':
        model = iresnet50(num_features=num_features)
        print("Finish loading iresnet50")
    elif model_name == 'iresnet100':
        model = iresnet100(num_features=num_features)
        print("Finish loading iresnet100")
    elif model_name == 'mobilefacenet':
        model = get_mbf(num_features=num_features)
        print("Finish loading mobilefacenet")
    elif model_name == 'vit_t':
        model = vit_t(num_features=num_features)
        print("Finish loading vit_t")
    elif model_name == 'vit_s':
        model = vit_s(num_features=num_features)
        print("Finish loading vit_s")
    elif model_name == 'vit_b':
        model = vit_b(num_features=num_features)
        print("Finish loading vit_b")
    elif model_name == 'vit_l':
        model = vit_l(num_features=num_features)
        print("Finish loading vit_l")
    else:
        raise NotImplementedError
    bs = 256
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = model(x)
    assert output.shape[0] == bs, 'output shape not match'

