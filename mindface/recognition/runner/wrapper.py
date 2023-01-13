"""
Wrapper.
"""
import numpy as np

from mindspore import nn
from mindspore import Tensor

__all__ = ["Network", "lr_generator"]


def lr_generator(lr_init, schedule, gamma, total_epochs, steps_per_epoch):
    """
    lr_generator.
    """
    lr_each_step = []
    for i in range(total_epochs):
        if i in schedule:
            lr_init *= gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)


class Network(nn.Cell):
    """
    WithLossCell.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self._backbone = backbone
        self.fc = head

    def construct(self, data):
        """
        construct.
        """
        out = self._backbone(data)
        logits = self.fc(out)

        return logits
