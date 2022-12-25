"""
helper.
"""
import math
import warnings
import numpy as np
from scipy import special

import mindspore as ms
from mindspore import nn, ops

__all__ = ['DropPath', 'trunc_normal_']

class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, keep_prob=None, seed=0):
        super().__init__()
        self.keep_prob = keep_prob
        seed = min(seed, 0)
        self.rand = ops.UniformReal(seed = seed)
        self.shape = ops.Shape()
        self.floor = ops.Floor()

    def construct(self, sample_x):
        """
        construct.
        """
        if not self.training or self.keep_prob == 1:
            return sample_x

        x_shape = self.shape(sample_x)
        shape = (sample_x.shape[0],) + (1,) * (len(x_shape) - 1)
        random_tensor = self.rand(shape)
        random_tensor = random_tensor + self.keep_prob
        random_tensor = self.floor(random_tensor)
        sample_x = sample_x / self.keep_prob
        sample_x = sample_x * random_tensor

        return sample_x

def _trunc_normal_(tensor, mean, std, a, b):
    """
    trunc_normal
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor = np.random.uniform(2 * l - 1, 2 * u - 1, tensor.shape)

    tensor = special.erfinv(tensor)

    tensor = tensor * std * math.sqrt(2.)
    tensor = ms.Tensor(tensor + mean, ms.float32)

    a = ms.Tensor(a, ms.float32)
    b = ms.Tensor(b, ms.float32)

    tensor = ops.clip_by_value(tensor, a, b)

    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Trunc_normal.

    Args:
        tensor(Object): An n-dimensional `torch.Tensor`.
        mean(Int): The mean of the normal distribution. Default: 0.
        std(Int): The standard deviation of the normal distribution. Default: 1.
        a(Int): The minimum cutoff value. Default: -2.
        b(Int): The maximum cutoff value. Default: 2.
    """
    return _trunc_normal_(tensor, mean, std, a, b)
