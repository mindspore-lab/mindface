'''
Loss: 本部分设计所需要的使用的损失函数
'''

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer



class ArcFace(nn.Cell):
    def __init__(self, world_size, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.shape = ops.Shape()
        self.mul = ops.Mul()
        self.cos = ops.Cos()
        self.acos = ops.ACos()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        self.on_value = Tensor(m, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, cosine, label):
        m_hot = self.onehot(label, self.shape(
            cosine)[1], self.on_value, self.off_value)
        cosine = self.acos(cosine)
        cosine += m_hot
        cosine = self.cos(cosine)
        cosine = self.mul(cosine, self.s)
        return cosine


class SoftMaxCE(nn.Cell):
    def __init__(self, world_size):
        super(SoftMaxCE, self).__init__()
        self.max = ops.ReduceMax(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.exp = ops.Exp()
        self.div = ops.Div()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        self.mul = ops.Mul()
        self.log = ops.Log()
        self.onvalue = Tensor(1.0, mstype.float32)
        self.offvalue = Tensor(0.0, mstype.float32)
        self.eps = Tensor(1e-30, mstype.float32)

    def construct(self, logits, total_label):
        max_fc = self.max(logits, 1)
        logits_exp = self.exp(logits - max_fc)
        logits_sum_exp = self.sum(logits_exp, 1)
        logits_exp = self.div(logits_exp, logits_sum_exp)
        label = self.onehot(total_label, F.shape(
            logits)[1], self.onvalue, self.offvalue)

        softmax_result_log = self.log(logits_exp + self.eps)
        loss = self.sum((self.mul(softmax_result_log, label)), -1)
        loss = self.mul(ops.scalar_to_array(-1.0), loss)
        loss_v = self.mean(loss, 0)
        return loss_v


