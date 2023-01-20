# import packages
import sys
sys.path.append('.')
import mindspore as ms
from mindspore import Tensor
import numpy as np
from mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l

def test_resnet50():
    """test resnet50"""
    bs = 256
    net = iresnet50()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_resnet100():
    """test resnet100"""
    bs = 256
    net = iresnet100()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_mobilefacenet():
    """test mobilefacenet"""
    bs = 256
    net = get_mbf()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_vit_t():
    """test vit_t"""
    bs = 256
    net = vit_t()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_vit_s():
    """test vit_t"""
    bs = 256
    net = vit_s()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_vit_b():
    """test vit_b"""
    bs = 256
    net = vit_b()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'

def test_vit_l():
    """test vit_b"""
    bs = 256
    net = vit_l()
    x = ms.Tensor(np.ones([bs, 3, 112, 112]), ms.float32)
    output = net(x)
    assert output.shape[0] == bs, 'output shape not match'