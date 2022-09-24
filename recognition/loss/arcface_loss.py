from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype


class ArcFace(nn.Cell):
    '''
    Arcface loss
    '''
    def __init__(self, world_size, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.shape = ops.Shape()
        self.mul = ops.Mul()
        self.cos = ops.Cos()
        self.acos = ops.ACos()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        # self.tile = ops.Tile().shard(((8, 1),))
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
