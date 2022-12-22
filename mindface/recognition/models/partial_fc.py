"""
partail_fc.
"""
import mindspore.nn as nn
from mindspore import Parameter
import mindspore.ops as ops
from mindspore.common.initializer import initializer


__all__=["PartialFC"]


class PartialFC(nn.Cell):
    """
    Build the arcface model without loss function.

    Args:
        num_classes (Int): The num of classes.
        world_size (Int): Number of processes involved in this work.

    Examples:
        >>> net=PartialFC(num_classes=num_classes, world_size=device_num)
    """
    def __init__(self, num_classes, world_size):
        super().__init__()
        self.l2_norm = ops.L2Normalize(axis=1)
        self.weight = Parameter(initializer(
            "normal", (num_classes, 512)), name="mp_weight")
        self.sub_weight = self.weight
        self.linear = ops.MatMul(transpose_b=True).shard(
            ((1, 1), (world_size, 1)))

    def construct(self, features):
        """
        construct.
        """
        total_features = self.l2_norm(features)
        norm_weight = self.l2_norm(self.sub_weight)
        logits = self.forward(total_features, norm_weight)
        return logits

    def forward(self, total_features, norm_weight):
        """
        forward.
        """
        logits = self.linear(total_features, norm_weight)
        return logits
