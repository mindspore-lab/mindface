from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import functional as F


class SoftMaxCE(nn.Cell):
    """
    Softmax cross entrophy of arcface.

    Args:
        world_size (Int): Number of processes involved in this work.

    Examples:
        >>> loss = SoftMaxCE(world_size=world_size)
    """
    def __init__(self, world_size):
        super(SoftMaxCE, self).__init__()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        self.onvalue = Tensor(1.0, mstype.float32)
        self.offvalue = Tensor(0.0, mstype.float32)
        self.eps = Tensor(1e-30, mstype.float32)

    def construct(self, logits, total_label):
        '''construct
        '''
        max_fc = F.amax(logits, 1, keep_dims=True)

        logits_exp = F.exp(logits - max_fc)
        logits_sum_exp = self.sum(logits_exp, 1)

        logits_exp = F.div(logits_exp, logits_sum_exp)

        label = self.onehot(total_label, F.shape(
            logits)[1], self.onvalue, self.offvalue)

        softmax_result_log = F.log(logits_exp + self.eps)
        loss = self.sum((F.mul(softmax_result_log, label)), -1)
        loss = F.mul(ops.scalar_to_array(-1.0), loss)
        loss_v = F.mean(loss, 0, keep_dims=False)

        return loss_v
