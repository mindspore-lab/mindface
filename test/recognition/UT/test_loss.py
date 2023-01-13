# import packages
import sys
sys.path.append('.')
import mindspore as ms
from mindspore import Tensor
import numpy as np
from mindface.detection.loss import MultiBoxLoss
import pytest

@pytest.mark.parametrize('batchsize', [8, 16])
@pytest.mark.parametrize('num_classes', [2])
@pytest.mark.parametrize('num_anchor', [16800, 29196])
@pytest.mark.parametrize('negative_ratio', [7])


def test_multiboxloss(batchsize, num_classes, num_anchor, negative_ratio):
    ms.set_seed(1)
    np.random.seed(1)
    multibox_loss = MultiBoxLoss(num_classes, num_anchor, negative_ratio)
    loc_data = ms.Tensor(np.random.randn(batchsize, num_anchor,4), ms.float32)
    loc_t = ms.Tensor(np.random.randn(batchsize, num_anchor,4), ms.float32)
    conf_data = ms.Tensor(np.random.randn(batchsize, num_anchor,2), ms.float32)
    conf_t = ms.Tensor(np.random.randn(batchsize, num_anchor), ms.float32)
    landm_data = ms.Tensor(np.random.randn(batchsize, num_anchor,10), ms.float32)
    landm_t = ms.Tensor(np.random.randn(batchsize, num_anchor,10), ms.float32)
    loss_bbox, loss_class, loss_landm = multibox_loss(loc_data, loc_t, conf_data, conf_t, landm_data, landm_t)

    assert loss_bbox > 0, 'Invalid Loss'
    assert loss_class > 0, 'Invalid Loss'
    assert loss_landm > 0, 'Invalid Loss'
