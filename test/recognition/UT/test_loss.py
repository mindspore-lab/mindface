# import packages
import sys
sys.path.append('.')
from mindface.detection.loss import MultiBoxLoss

def test_multiboxloss():
    batch_size = 8
    negative_ratio = 7
    num_classes = 2
    num_anchor = 16800
    multibox_loss = MultiBoxLoss(num_classes, num_anchor, negative_ratio)