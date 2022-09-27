# import packages
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'mindface/detection'))

from mindface.detection.loss import MultiBoxLoss

def test_multiboxloss():
    batch_size = 8
    negative_ratio = 7
    num_classes = 2
    num_anchor = 16800
    multibox_loss = MultiBoxLoss(num_classes, num_anchor, negative_ratio, batch_size)