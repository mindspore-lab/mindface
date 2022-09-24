import pytest
from src.iresnet import iresnet18, iresnet34, iresnet50, iresnet100

# iresnet
__all__ = ['iresnet50', 'iresnet100']



if __name__ == "__main__":
    import mindspore.numpy as np
    import mindspore as ms

    from mindspore.parallel import _cost_model_context as cost_model_context
    from mindspore import context, Tensor

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target='GPU', save_graphs=False)
    def test_model():
        
        net = iresnet50()
        x = ms.Tensor(np.ones([4, 3, 112, 112]), ms.float32)
        print(x.shape)
        output = net(x)
        print(output.shape)