import os
import sys
# sys.path.append(os.getcwd())
# sys.path.append(os.path.join(os.getcwd(),'mindface/recognition'))
sys.path.append('.')
import mindspore.numpy as np
import mindspore as ms
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore import context, Tensor
from mindface.recognition.models.mobilefacenet import get_mbf
import pytest


@pytest.mark.parametrize("mode",['PYNATIVE_MODE','GRAPH_MODE'])
# @pytest.mark.parametrize("target",['CPU'])

def test_model(mode):
    if mode == 'PYNATIVE_MODE':
        context.set_context(mode=context.PYNATIVE_MODE,device_target='CPU', save_graphs=False)
    else:
        context.set_context(mode=context.GRAPH_MODE,device_target='CPU', save_graphs=False)
    bs = 256
    net = get_mbf(False,512)
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'
    print('unit test OK')
# test_model(mode='PYNATIVE_MODE', target='GPU')