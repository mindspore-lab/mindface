# import packages
import sys
import os 
sys.path.append('.')
from mindface.recognition.runner import Network
from mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l, PartialFC
from mindface.recognition.loss import ArcFace

def test_multiboxloss():
    model_name = 'iresnet50'
    num_features = 512
    num_classes = 10572
    device_num = int(os.getenv('RANK_SIZE'))
    if model_name == 'iresnet50':
        model = iresnet50(num_features=num_features)
        print("Finish loading iresnet50")
    elif model_name == 'iresnet100':
        model = iresnet100(num_features=num_features)
        print("Finish loading iresnet100")
    elif model_name == 'mobilefacenet':
        model = get_mbf(num_features=num_features)
        print("Finish loading mobilefacenet")
    elif model_name == 'vit_t':
        model = vit_t(num_features=num_features)
        print("Finish loading vit_t")
    elif model_name == 'vit_s':
        model = vit_s(num_features=num_features)
        print("Finish loading vit_s")
    elif model_name == 'vit_b':
        model = vit_b(num_features=num_features)
        print("Finish loading vit_b")
    elif model_name == 'vit_l':
        model = vit_l(num_features=num_features)
        print("Finish loading vit_l")
    else:
        raise NotImplementedError


    head = PartialFC(num_classes = num_classes, world_size=device_num)

    train_net = Network(model, head)

    loss_func = ArcFace(world_size=device_num)