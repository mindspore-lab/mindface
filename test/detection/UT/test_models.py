# import packages
import sys
sys.path.append('.')
import mindspore
from mindspore import Tensor
import numpy as np
from mindface.detection.models import resnet50, mobilenet025, RetinaFace
import pytest

@pytest.mark.parametrize('backbone_name', ['mobilenet025', 'resnet50'])
def test_models(backbone_name):
    dummy_input = Tensor(np.random.rand(8, 3, 224, 224), dtype=mindspore.float32)
    if backbone_name == 'resnet50':
        backbone = resnet50(1001)
        y1, y2, y3 = backbone(dummy_input)
        assert y3.shape==(8, 2048,7,7), 'output shape not match'
        assert y2.shape==(8, 1024,14,14), 'output shape not match'
        assert y1.shape==(8, 512,28,28), 'output shape not match'
        
        retinaface = RetinaFace(phase='train', backbone=backbone, in_channel=256, out_channel=256)
        y = retinaface(dummy_input)

        assert y[0].shape==(8, 2058,4), 'BBoxHead output shape not match'
        assert y[1].shape==(8, 2058,2), 'ClassHead output shape not match'
        assert y[2].shape==(8, 2058,10), 'LanmarkHead output shape not match'        
    elif backbone_name == 'mobilenet025':
        backbone = mobilenet025(1000)
        y4, y5, y6 = backbone(dummy_input)
        assert y6.shape==(8, 256,7,7), 'output shape not match'
        assert y5.shape==(8, 128,14,14), 'output shape not match'
        assert y4.shape==(8, 64,28,28), 'output shape not match'

        retinaface = RetinaFace(phase='train', backbone = backbone, in_channel=32, out_channel=64)
        y = retinaface(dummy_input)

        assert y[0].shape==(8, 2058,4), 'BBoxHead output shape not match'
        assert y[1].shape==(8, 2058,2), 'ClassHead output shape not match'
        assert y[2].shape==(8, 2058,10), 'LanmarkHead output shape not match'