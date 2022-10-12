import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'mindface/recognition'))
import numpy as np
import mindspore as ms
import argparse
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
from mindface.recognition.models import iresnet100, iresnet50
from mindface.recognition.models import get_mbf


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
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--device_target', type=str,
                    default='GPU', choices=['GPU', 'Ascend'])
    parser.add_argument('--backbone', type=str,
                    default='iresnet50', choices=['iresnet50', 'iresnet100', 'mobilefacenet'])
    args = parser.parse_args()
    if args.device_target == 'GPU':
        context.set_context(
            device_id=0, mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(
            device_id=0, mode=context.GRAPH_MODE, device_target="Ascend")


    data = np.random.randn(3, 112, 112)
    out1 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out1.shape)
    out2 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out2.shape)
    out3 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out3.shape)

    data = np.random.randn(4, 3, 112, 112)
    out1 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out1.shape)
    out2 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out2.shape)
    out3 = infer(data, backbone=args.backbone,
                 pretrained="")
    print(out3.shape)
