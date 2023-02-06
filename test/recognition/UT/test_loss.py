# import packages
import sys
import os 
sys.path.append('.')
from mindface.recognition.runner import Network
from mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l, PartialFC
from mindface.recognition.loss import ArcFace

import pytest

@pytest.mark.parametrize('model_name', ['iresnet50', 'iresnet100', 'mobilefacenet', 'vit_t', 'vit_s', 'vit_b', 'vit_l'])
@pytest.mark.parametrize('num_classes', [10572, 85742])

def test_loss(model_name, num_classes):  
    num_features = 512
    device_num = 1
    if model_name == 'iresnet50':
        model = iresnet50()
        print("Finish loading iresnet50")
    elif model_name == 'iresnet100':
        model = iresnet100()
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

    head = PartialFC(num_classes = num_classes, world_size=device_num)

    train_net = Network(model, head)

    loss_func = ArcFace(world_size=device_num)
    assert num_features > 0, 'Invalid Loss'