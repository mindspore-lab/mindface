import mindspore.nn as nn
from mindspore import Parameter
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer

__all__=["PartialFC"]

class PartialFC(nn.Cell):
    '''
    partialFC
    '''
    def __init__(self, num_classes, world_size):
        super(PartialFC, self).__init__()
        self.L2Norm = ops.L2Normalize(axis=1)
        self.weight = Parameter(initializer(
            "normal", (num_classes, 512)), name="mp_weight")
        self.sub_weight = self.weight
        self.linear = ops.MatMul(transpose_b=True).shard(
            ((1, 1), (world_size, 1)))

    def construct(self, features):
        total_features = self.L2Norm(features)
        norm_weight = self.L2Norm(self.sub_weight)
        logits = self.forward(total_features, norm_weight)
        return logits

    def forward(self, total_features, norm_weight):
        logits = self.linear(F.cast(total_features, mstype.float16), F.cast(
            norm_weight, mstype.float16))
        return F.cast(logits, mstype.float32)

