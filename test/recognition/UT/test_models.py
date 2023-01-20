# import packages
import sys
sys.path.append('.')
import mindspore as ms
from mindspore import Tensor
import numpy as np
import pytest
from mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l

@pytest.mark.parametrize('model_name', ['iresnet50', 'iresnet100', 'mobilefacenet', 'vit_t', 'vit_s', 'vit_b', 'vit_l'])
def test_model(model_name):
    num_features = 512
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
    bs = 128
    x = ms.Tensor(np.ones([128, 3, 112, 112]), ms.float32)
    output = model(x)
    assert output.shape[0] == bs, 'output shape not match'
