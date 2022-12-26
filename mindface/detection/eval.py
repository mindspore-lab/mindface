# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Eval Retinaface_resnet50_or_mobilenet0.25."""
import argparse
import os
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import ops

from utils import prior_box
from models import RetinaFace, resnet50, mobilenet025
from runner import DetectionEngine, Timer, read_yaml

def val(cfg):
    """val"""
    if cfg['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])


    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    network = RetinaFace(phase='predict',backbone=backbone,in_channel=cfg['in_channel'],out_channel=cfg['out_channel'])
    backbone.set_train(False)
    network.set_train(False)

    # load checkpoint
    assert cfg['val_model'] is not None, 'val_model is None.'
    param_dict = load_checkpoint(cfg['val_model'])
    print(f"Load trained model done. {cfg['val_model']}")
    network.init_parameters_data()
    load_param_into_net(network, param_dict)

    # testing dataset
    testset_folder = cfg['val_dataset_folder']
    testset_label_path = cfg['val_dataset_folder'] + "label.txt"
    with open(testset_label_path, 'r', encoding = 'utf-8') as file:
        all_test_dataset = file.readlines()
        test_dataset = []
        for im_path in all_test_dataset:
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:-1])  # delete '# ...\n'

    num_images = len(test_dataset)

    timers = {'forward_time': Timer(), 'misc': Timer()}

    if cfg['val_origin_size']:
        h_max, w_max = 0, 0
        for img_name in test_dataset:
            image_path = os.path.join(testset_folder, 'images', img_name)
            img_each = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_each.shape[0] > h_max:
                h_max = img_each.shape[0]
            if img_each.shape[1] > w_max:
                w_max = img_each.shape[1]

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
    detection = DetectionEngine(nms_thresh=cfg['val_nms_threshold'], conf_thresh=cfg['val_confidence_threshold'],
        iou_thresh=cfg['val_iou_threshold'], var=cfg['variance'],
        save_prefix=cfg['val_predict_save_folder'], gt_dir=cfg['val_gt_dir'])


    # testing begin
    print('Predict box starting')
    ave_time = 0
    ave_forward_pass_time = 0
    ave_misc = 0
    i = 0
    for i, img_name in enumerate(test_dataset):
        test_scales = [500, 800, 1100, 1400, 1700]
        timers['forward_time'].start()
        for idx, test_scale in enumerate(test_scales):
            target_size = test_scale
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

            boxes, confs, _ = network(img)
            if idx ==0:
                boxes_all = boxes
                confs_all =  confs
                resize_all = [resize]
            else:
                boxes_all = ops.concat((boxes_all,boxes))
                confs_all = ops.concat((confs_all,confs),axis=1)
                resize_all.append(resize)
        timers['forward_time'].end()

        boxes = boxes_all
        confs = confs_all
        resize = resize_all

        timers['misc'].start()
        detection.eval(boxes, confs, resize, scale, img_name, priors)
        timers['misc'].end()

        ave_time = ave_time + timers['forward_time'].diff + timers['misc'].diff
        ave_forward_pass_time = ave_forward_pass_time + timers['forward_time'].diff
        ave_misc = ave_misc + timers['misc'].diff

        forward_time = timers['forward_time'].diff
        sum_time = timers['forward_time'].diff + timers['misc'].diff
        print(f"im_detect: {i + 1}/{num_images} forward_pass_time: {forward_time:.4f}s",end=' ')
        print(f"misc: {timers['misc'].diff:.4f}s sum_time: {sum_time:.4f}s")

    print(f"ave_time: {(ave_time/(i+1)):.4f}s")
    print(f"ave_forward_pass_time: {(ave_forward_pass_time/(i+1)):.4f}s")
    print(f"ave_misc: {(ave_misc/(i+1)):.4f}s")
    print('Predict box done.')
    print('Eval starting')

    if cfg['val_save_result']:
        # Save the predict result if you want.
        predict_result_path = detection.write_result()
        print(f'predict result path is {predict_result_path}')

    detection.get_eval_result()
    print('Eval done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='val')
    # configs
    parser.add_argument('--config', default='mindface/detection/configs/RetinaFace_mobilenet025.yaml', type=str,
                        help='configs path')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='checpoint path')
    args = parser.parse_args()

    config = read_yaml(args.config)

    if args.checkpoint:
        config['val_model'] = args.checkpoint
    val(cfg=config)
