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
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from configs.RetinaFace_mobilenet import cfg_mobile025
from configs.RetinaFace_resnet50 import cfg_res50
from utils import prior_box

from models import RetinaFace, resnet50, mobilenet025
from eval import DetectionEngine

def infer(cfg):
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
    # image_path = 'imgs/0_Parade_marchingband_1_1004.jpg'
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
    boxes = detection.detect(boxes, confs, resize, scale, image_path, priors,phase='test')
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
    parser = argparse.ArgumentParser(description='val')
    parser.add_argument('--backbone_name', type=str, default='ResNet50',
                        help='backbone name')
    parser.add_argument('--checkpoint', type=str, default='pretrained/RetinaFace_ResNet50.ckpt',
                        help='checpoint path')                      
    parser.add_argument('--image_path', type=str, default='imgs/0000.jpg',
                        help='image path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='image path')
    args = parser.parse_args()
    if args.backbone_name == 'ResNet50':
        config = cfg_res50
    elif args.backbone_name == 'MobileNet025':
        config = cfg_mobile025
    if args.image_path:
        config['image_path'] = args.image_path
    if args.conf:
        config['conf'] = args.conf
    if args.checkpoint:
        config['val_model'] = args.checkpoint
    infer(cfg=config)
