# import packages
import sys
sys.path.append('.')
import mindspore
from mindspore import Tensor
import numpy as np
from mindface.detection.models import resnet50 

def test_resnet50():
    """test resnet50"""
    backbone_resnet50 = resnet50(1001)
    batchsize = 8 
    dummy_input = Tensor(np.random.rand(batchsize, 3, 224, 224), dtype=mindspore.float32)
    y1, y2, y3 = backbone_resnet50(dummy_input)

    assert y3.shape==(batchsize, 2048,7,7), 'output shape not match'
    assert y2.shape==(batchsize, 1024,14,14), 'output shape not match'
    assert y1.shape==(batchsize, 512,28,28), 'output shape not match'

from mindface.detection.models import mobilenet025 

def test_mobilenet025():
    """test mobilenet025"""
    backbone_mobilenet025 = mobilenet025(1000)
    batchsize = 8 
    dummy_input = Tensor(np.random.rand(batchsize, 3, 224, 224), dtype=mindspore.float32)
    y4, y5, y6 = backbone_mobilenet025(dummy_input)

    assert y6.shape==(batchsize, 256,7,7), 'output shape not match'
    assert y5.shape==(batchsize, 128,14,14), 'output shape not match'
    assert y4.shape==(batchsize, 64,28,28), 'output shape not match'

from mindface.detection.models import RetinaFace

def test_retinaface_mobilenet025():
    """test retinaface_mobilenet025"""
    batchsize = 8 
    backbone_mobilenet025 = mobilenet025(1000)
    cfg = {'in_channel':32,'out_channel':64}
    retinaface_mobilenet025 = RetinaFace(phase='train', backbone = backbone_mobilenet025, cfg=cfg)
    dummy_input = Tensor(np.random.rand(batchsize, 3, 224, 224), dtype=mindspore.float32)
    y = retinaface_mobilenet025(dummy_input)

    assert y[0].shape==(batchsize, 2058,4), 'BBoxHead output shape not match'
    assert y[1].shape==(batchsize, 2058,2), 'ClassHead output shape not match'
    assert y[2].shape==(batchsize, 2058,10), 'LanmarkHead output shape not match'

def test_retinaface_resnet50():
    """test retinaface_resnet50"""
    batchsize = 8 
    backbone_resnet50 = resnet50(1001)
    cfg = {'in_channel':256,'out_channel':256}
    retinaface_resnet50 = RetinaFace(phase='train', backbone = backbone_resnet50, cfg=cfg)
    dummy_input = Tensor(np.random.rand(batchsize, 3, 224, 224), dtype=mindspore.float32)
    y = retinaface_resnet50(dummy_input)

    assert y[0].shape==(batchsize, 2058,4), 'BBoxHead output shape not match'
    assert y[1].shape==(batchsize, 2058,2), 'ClassHead output shape not match'
    assert y[2].shape==(batchsize, 2058,10), 'LanmarkHead output shape not match'