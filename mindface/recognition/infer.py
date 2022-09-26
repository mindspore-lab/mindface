import numpy as np
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

import sys
sys.path.append("recognition/")

from models import iresnet100, iresnet50, get_mbf


def infer(img, backbone="iresnet50", pretrained=False):
    '''
    img: numpy, ms.Tensor
    backbone: iresnet50, iresnet100, mobilefacenet
    '''

    assert (img.shape[-1] == 112 and img.shape[-2] == 112)
    img = ((img / 255) - 0.5) / 0.5
    img = ms.Tensor(img, ms.float32)
    if len(img.shape) == 4:
        pass
    elif len(img.shape) == 3:
        img = img.expand_dims(axis=0)

    if backbone == "iresnet50":
        model = iresnet50()
    elif backbone == "iresnet100":
        model = iresnet100()
    elif backbone == "mobilefacenet":
        model = get_mbf(False, 512)
    else:
        raise NotImplementedError

    if pretrained:
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(model, param_dict)

    net_out = model(img)
    embeddings = net_out.asnumpy()

    return embeddings


if __name__ == '__main__':

    context.set_context(
        device_id=0, mode=context.GRAPH_MODE, device_target="GPU")

    data = np.random.randn(3, 112, 112)
    print(data.shape)
    # assert 1==0
    # out1 = infer(data, backbone="iresnet50",
    #              pretrained="train_parallel_iresnet50_gradclip/ArcFace--1_1200.ckpt")
    # print(out1.shape)
    # out2 = infer(data, backbone="iresnet100",
    #              pretrained="train_parallel_iresnet100_gradclip/ArcFace--1_840.ckpt")
    # print(out2.shape)
    # out3 = infer(data, backbone="mobilefacenet",
    #              pretrained="train_parallel_small_gradclip_casia/ArcFace--25_958.ckpt")
    # print(out3.shape)

    # data = np.random.randn(4, 3, 112, 112)
    # out1 = infer(data, backbone="iresnet50",
    #              pretrained="train_parallel_iresnet50_gradclip/ArcFace--1_1200.ckpt")
    # print(out1.shape)
    # out2 = infer(data, backbone="iresnet100",
    #              pretrained="train_parallel_iresnet100_gradclip/ArcFace--1_840.ckpt")
    # print(out2.shape)
    # out3 = infer(data, backbone="mobilefacenet",
    #              pretrained="train_parallel_small_gradclip_casia/ArcFace--25_958.ckpt")
    # print(out3.shape)
