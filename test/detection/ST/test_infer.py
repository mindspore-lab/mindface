"""Eval Retinaface_resnet50_or_mobilenet0.25."""
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindface.detection.utils import prior_box

from mindface.detection.models import RetinaFace, resnet50, mobilenet025
from mindface.detection.runner import DetectionEngine, read_yaml

def test_infer(cfg):
    """test one image"""
    if cfg['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])

    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    network = RetinaFace(phase='predict', backbone=backbone, cfg=cfg)
    backbone.set_train(False)
    network.set_train(False)

    # load checkpoint
    assert cfg['val_model'] is not None, 'val_model is None.'
    param_dict = load_checkpoint(cfg['val_model'])
    print('Load trained model done. {}'.format(cfg['val_model']))
    network.init_parameters_data()
    load_param_into_net(network, param_dict)

    # testing image

    conf_test = cfg['conf']
    test_origin_size = False
    image_path = cfg['image_path']

    if test_origin_size:
        h_max, w_max = 0, 0

        _img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if _img.shape[0] > h_max:
            h_max = _img.shape[0]
        if _img.shape[1] > w_max:
            w_max = _img.shape[1]

        h_max = (int(h_max / 32) + 1) * 32
        w_max = (int(w_max / 32) + 1) * 32

        priors = prior_box(image_sizes=(h_max, w_max),
                           min_sizes=[[16, 32], [64, 128], [256, 512]],
                           steps=[8, 16, 32],
                           clip=False)
    else:
        target_size = 1600
        max_size = 2176
        priors = prior_box(image_sizes=(max_size, max_size),
                            min_sizes=[[16, 32], [64, 128], [256, 512]],
                            steps=[8, 16, 32],
                            clip=False)
    detection = DetectionEngine(cfg)
    # testing begin
    print('Predict box starting')

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    #testing scale
    if test_origin_size:
        resize = 1
        assert img.shape[0] <= h_max and img.shape[1] <= w_max
        image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    else:
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        assert img.shape[0] <= max_size and img.shape[1] <= max_size
        image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t


    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = Tensor(img)

    boxes, confs, _ = network(img)
    boxes = detection.infer(boxes, confs, resize, scale, image_path, priors)
    _img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for box in boxes:
        if box[4] > conf_test:
            cv2.rectangle(_img,(int(box[0]),int(box[1])),
                (int(box[0])+int(box[2]),int(box[1])+int(box[3])),color=(0,0,255))
            cv2.putText(_img,str(round(box[4],5)),(int(box[0]),int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    save_path = image_path.split('.')[0]+'_pred.jpg'
    cv2.imwrite(save_path,_img)
    print(f'Result saving: {save_path}')

if __name__ == '__main__':

    # read the configs
    cfg_res50 = read_yaml('mindface/detection/configs/RetinaFace_resnet50.yaml')
    cfg_mobile025 = read_yaml('mindface/detection/configs/RetinaFace_mobilenet025.yaml')
    test_infer(cfg=cfg_res50)
    test_infer(cfg=cfg_mobile025)
