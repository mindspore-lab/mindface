"""
Training face recognition models.
"""
import os
import argparse
import sys
sys.path.append('.')

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

import pytest

@pytest.mark.parametrize('model_name', ['iresnet50', 'iresnet100', 'mobilefacenet', 'vit_t', 'vit_s', 'vit_b', 'vit_l'])
@pytest.mark.parametrize('num_classes', [10572, 85742])
def infer_test(model_name, num_classes):
    batch_size = 128
    device_target = 'Ascend'
    running_mode = 'GRAPH'
    seed = 2022
    num_features = 512
    ms.common.set_seed(seed)
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

