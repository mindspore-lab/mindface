"""Eval Retinaface_resnet50_or_mobilenet0.25."""
import os
import numpy as np
import cv2
import sys
sys.path.append('.')

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindface.detection.utils import prior_box

from mindface.detection.models import RetinaFace,resnet50,mobilenet025
from mindface.detection.runner import DetectionEngine, Timer, read_yaml

def test_val(cfg):
    """test eval"""
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

    # testing dataset
    testset_folder = cfg['val_dataset_folder']
    testset_label_path = cfg['val_dataset_folder'] + "label.txt"
    with open(testset_label_path, 'r', encoding='utf-8') as f:
        _test_dataset = f.readlines()
        test_dataset = []
        for im_path in _test_dataset:
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:-1])  # delete '# ...\n'

    num_images = len(test_dataset)

    timers = {'forward_time': Timer(), 'misc': Timer()}

    if cfg['val_origin_size']:
        h_max, w_max = 0, 0
        for img_name in test_dataset:
            image_path = os.path.join(testset_folder, 'images', img_name)
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
        max_size = 2160
        priors = prior_box(image_sizes=(max_size, max_size),
                           min_sizes=[[16, 32], [64, 128], [256, 512]],
                           steps=[8, 16, 32],
                           clip=False)

    # init detection engine
    detection = DetectionEngine(cfg)

    # testing begin
    print('Predict box starting')
    ave_time = 0
    ave_forward_pass_time = 0
    ave_misc = 0
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, 'images', img_name)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        if cfg['val_origin_size']:
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

        timers['forward_time'].start()
        boxes, confs, _ = network(img)
        timers['forward_time'].end()
        timers['misc'].start()
        detection.detect(boxes, confs, resize, scale, img_name, priors)
        timers['misc'].end()

        ave_time = ave_time + timers['forward_time'].diff + timers['misc'].diff
        ave_forward_pass_time = ave_forward_pass_time + timers['forward_time'].diff
        ave_misc = ave_misc + timers['misc'].diff
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s sum_time: {:.4f}s'.format(i + 1, num_images,
                                                                                     timers['forward_time'].diff,
                                                                                     timers['misc'].diff,
                                                                                     timers['forward_time'].diff + timers['misc'].diff))
    print("ave_time: {:.4f}s".format(ave_time/(i+1)))
    print("ave_forward_pass_time: {:.4f}s".format(ave_forward_pass_time/(i+1)))
    print("ave_misc: {:.4f}s".format(ave_misc/(i+1)))
    print('Predict box done.')
    print('Eval starting')

    if cfg['val_save_result']:
        # Save the predict result if you want.
        predict_result_path = detection.write_result()
        print('predict result path is {}'.format(predict_result_path))

    detection.get_eval_result()
    print('Eval done.')

if __name__ == '__main__':

    # read the configs
    cfg_res50 = read_yaml('mindface/detection/configs/RetinaFace_resnet50.yaml')
    cfg_mobile025 = read_yaml('mindface/detection/configs/RetinaFace_mobilenet025.yaml')
    test_val(cfg=cfg_res50)
    test_val(cfg=cfg_mobile025)
