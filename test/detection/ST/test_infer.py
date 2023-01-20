# import packages
import sys
sys.path.append('.')

import numpy as np
import pytest
import cv2

from mindspore import Tensor, context
from mindface.detection.models import RetinaFace, mobilenet025, resnet50
from mindface.detection.runner import DetectionEngine
from mindface.detection.utils import prior_box

@pytest.mark.parametrize('backbone_name', ['mobilenet025', 'resnet50'])
@pytest.mark.parametrize('target_size', [1200, 1600])
def test_detect(backbone_name, target_size):
    """The test api of eval and infer."""
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    if backbone_name == 'resnet50':
        backbone = resnet50(1001)
        network = RetinaFace(phase='predict', backbone = backbone, in_channel=256, out_channel=256)     
    elif backbone_name == 'mobilenet025':
        backbone = mobilenet025(1000)
        network = RetinaFace(phase='predict', backbone = backbone, in_channel=32, out_channel=64)

    backbone.set_train(False)
    network.set_train(False)

    detector = DetectionEngine()
    target_size = target_size
    max_size = int(target_size*1.2)
    priors = prior_box(image_sizes=(max_size, max_size),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)
    image_path = 'test/detection/imgs/0000.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    img = np.float32(img)

    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
    image_t[:, :] = (104.0, 117.0, 123.0)
    image_t[0:img.shape[0], 0:img.shape[1]] = img
    img = image_t

    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = Tensor(img)

    if resize is not list:
        resize = [resize]
    boxes, confs, _ = network(img)
    boxes_infer = detector.infer(boxes, confs, resize, scale, priors)
    assert len(boxes_infer) >0, 'Can not detect the faces'
    assert len(boxes_infer[0])==5, 'Not a BBox'
    detector.eval(boxes, confs, resize, scale, image_path, priors)
    results = detector.write_result()
    assert results is not None, 'Saved Nothing!'

if __name__ == '__main__':
    test_detect('mobilenet025', 1600)