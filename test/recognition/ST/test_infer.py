"""
inference of face recognition models.
"""
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l

def test_infer(img, backbone="iresnet50", num_features=512, pretrained=False):
    assert (img.shape[-1] == 112 and img.shape[-2] == 112)
    img = ((img / 255) - 0.5) / 0.5
    img = ms.Tensor(img, ms.float32)
    if len(img.shape) == 4:
        pass
    elif len(img.shape) == 3:
        img = img.expand_dims(axis=0)

    if backbone == 'iresnet50':
        model = iresnet50(num_features=num_features)
        print("Finish loading iresnet50")
    elif backbone == 'iresnet100':
        model = iresnet100(num_features=num_features)
        print("Finish loading iresnet100")
    elif backbone == 'mobilefacenet':
        model = get_mbf(num_features=num_features)
        print("Finish loading mobilefacenet")
    elif backbone == 'vit_t':
        model = vit_t(num_features=num_features)
        print("Finish loading vit_t")
    elif backbone == 'vit_s':
        model = vit_s(num_features=num_features)
        print("Finish loading vit_s")
    elif backbone == 'vit_b':
        model = vit_b(num_features=num_features)
        print("Finish loading vit_b")
    elif backbone == 'vit_l':
        model = vit_l(num_features=num_features)
        print("Finish loading vit_l")
    else:
        raise NotImplementedError

    if pretrained:
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(model, param_dict)

    net_out = model(img)
    embeddings = net_out.asnumpy()

    return embeddings