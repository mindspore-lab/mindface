"""Network."""
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor

# MobileNet0.25
def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride,
                  pad_mode='pad', padding=1, has_bias=False),
        nn.BatchNorm2d(num_features=oup, momentum=0.9),
        nn.LeakyReLU(alpha=leaky)  # ms official: nn.get_activation('relu6')
    ])

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=stride,
                  pad_mode='pad', padding=1, group=inp, has_bias=False),
        nn.BatchNorm2d(num_features=inp, momentum=0.9),
        nn.LeakyReLU(alpha=leaky),  # ms official: nn.get_activation('relu6')

        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1,
                  pad_mode='pad', padding=0, has_bias=False),
        nn.BatchNorm2d(num_features=oup, momentum=0.9),
        nn.LeakyReLU(alpha=leaky),  # ms official: nn.get_activation('relu6')
    ])


class MobileNetV1(nn.Cell):
    """MobileNetV1"""
    def __init__(self, num_classes):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.SequentialCell([
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        ])
        self.stage2 = nn.SequentialCell([
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        ])
        self.stage3 = nn.SequentialCell([
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        ])
        self.avg = P.ReduceMean()
        self.fc = nn.Dense(in_channels=256, out_channels=num_classes)

    def construct(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        out = self.avg(x3, (2, 3))
        out = self.fc(out)
        return x1, x2, x3


def mobilenet025(class_num=1000):
    return MobileNetV1(class_num)