"""engine"""
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

import datetime
import time
import os
import json
import numpy as np
import yaml
from scipy.io import loadmat

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

def read_yaml(path):
    """read_yaml"""
    with open (path, 'r', encoding='utf-8') as file :
        string = file.read()
        dict_yaml = yaml.safe_load(string)

    return dict_yaml

def decode_bbox(bbox, priors, var):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        bbox (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        var: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, 0:2] + bbox[:, 0:2] * var[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(bbox[:, 2:4] * var[1])), axis=1)  # (xc, yc, w, h)
    boxes[:, :2] -= boxes[:, 2:] / 2    # (x0, y0, w, h)
    boxes[:, 2:] += boxes[:, :2]        # (x0, y0, x1, y1)
    return boxes

class Timer():
    """Timer"""
    def __init__(self):
        self.start_time = 0.
        self.diff = 0.

    def start(self):
        """start"""
        self.start_time = time.time()

    def end(self):
        """end"""
        self.diff = time.time() - self.start_time

class DetectionEngine:
    """
    DetectionEngine, a detector

    Args:
        nms_thresh (Float): The threshold of nms method. Default: 0.4
        conf_thresh (Float): The threshold of confidence. DeFault: 0.02
        iou_thresh (Float): The threshold of iou. DeFault: 0.5
        var (List): Variances of priorboxes. Default: [0.1, 0.2]
        gt_dir (String): The path of ground truth.
    Examples:
        >>> detection = DetectionEngine(cfg)
    """
    def __init__(self, nms_thresh=0.4, conf_thresh=0.02, iou_thresh=0.5, var=None,
                        gt_dir='data/WiderFace/ground_truth'):
        self.results = {}
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.var = var or [0.1,0.2]
        self.gt_dir = gt_dir

    def _iou(self, a, b):
        """_iou"""
        a_shape0 = a.shape[0]
        b_shape0 = b.shape[0]
        max_xy = np.minimum(
            np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [a_shape0, b_shape0, 2]),
            np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [a_shape0, b_shape0, 2]))
        min_xy = np.maximum(
            np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [a_shape0, b_shape0, 2]),
            np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [a_shape0, b_shape0, 2]))
        inter = np.maximum((max_xy - min_xy + 1), np.zeros_like(max_xy - min_xy))
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_a = np.broadcast_to(
            np.expand_dims(
                (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1),
            np.shape(inter))
        area_b = np.broadcast_to(
            np.expand_dims(
                (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), 0),
            np.shape(inter))
        union = area_a + area_b - inter
        return inter / union

    def _nms(self, boxes, threshold=0.5):
        """_nms"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]

        return reserved_boxes

    def write_result(self, save_path  = None):
        """write_result"""
        # save result to file.
        if not save_path:
            return self.results
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_path = save_path + '/predict' + t + '.json'
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(self.results, file)
                print(f"The results were saved in {file_path}.")
            file.close()
            return self.results
        except IOError as err:
            raise RuntimeError(f"Unable to open json file to dump. What(): {err}") from err

    def eval(self, boxes, confs, resize, scale, image_path, priors):
        """eval
        Args:
            boxes: The boxes predicted by network.
            confs: The confidence of boxes.
            resize: The image scaling factor.
            scale: The origin image size.
            image_path: The image path of the image.
            priors: The prior boxes.
        """

        if boxes.shape[0] == 0:
            # add to result
            event_name = image_path.split('/')[-2]
            img_name = image_path.split('/')[-1]
            self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                       'bboxes': []}
            return

        for i, size in enumerate(resize):
            boxes_1 = boxes[i].asnumpy()
            boxes_1 = decode_bbox(boxes_1, priors, self.var)
            boxes_1 = boxes_1* scale / size
            if i==0:
                boxes_all = boxes_1
            else:
                boxes_all = np.append(boxes_all,boxes_1,axis=0)

        boxes = boxes_all

        scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]

        dets[:, 2:4] = (dets[:, 2:4].astype(np.int32) - dets[:, 0:2].astype(np.int32)).astype(np.float32) # int
        dets[:, 0:4] = dets[:, 0:4].astype(np.int32).astype(np.float32)                                 # int


        # add to result
        event_name = image_path.split('/')[-2]
        img_name = image_path.split('/')[-1]
        event_names = self.results.keys()
        if event_name not in event_names:
            self.results[event_name] = {}
        self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                   'bboxes': dets[:, :5].astype(np.float32).tolist()}

    def infer(self, boxes, confs, resize, scale, priors):
        """infer
        Args:
            boxes: The boxes predicted by network.
            confs: The confidence of boxes.
            resize: The image scaling factor.
            scale: The origin image size.
            priors: The prior boxes.
        """

        if boxes.shape[0] == 0:
            return None

        boxes = decode_bbox(np.squeeze(boxes.asnumpy(), 0), priors, self.var)
        boxes = boxes * scale / resize

        scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]

        dets[:, 2:4] = (dets[:, 2:4].astype(np.int32) - dets[:, 0:2].astype(np.int32)).astype(np.float32) # int
        dets[:, 0:4] = dets[:, 0:4].astype(np.int32).astype(np.float32)                                 # int


        # return boxes
        return dets[:, :5].astype(np.float32).tolist()

    def _get_gt_boxes(self):
        """_get_gt_boxes"""
        gt = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        hard = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        medium = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        easy = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))

        faceboxes = gt['face_bbx_list']
        events = gt['event_list']
        files = gt['file_list']

        hard_gt_list = hard['gt_list']
        medium_gt_list = medium['gt_list']
        easy_gt_list = easy['gt_list']

        return faceboxes, events, files, hard_gt_list, medium_gt_list, easy_gt_list

    def _norm_pre_score(self):
        """_norm_pre_score"""
        max_score = 0
        min_score = 1

        for event, event_boxes in self.results.items():
            for name in event_boxes:
                bbox = np.array(event_boxes[name]['bboxes']).astype(np.float32)
                if bbox.shape[0] <= 0:
                    continue
                max_score = max(max_score, np.max(bbox[:, -1]))
                min_score = min(min_score, np.min(bbox[:, -1]))

        length = max_score - min_score
        for event, event_boxes in self.results.items():
            for name in event_boxes:
                bbox = np.array(event_boxes[name]['bboxes']).astype(np.float32)
                if bbox.shape[0] <= 0:
                    continue
                bbox[:, -1] -= min_score
                bbox[:, -1] /= length
                self.results[event][name]['bboxes'] = bbox.tolist()

    def _image_eval(self, predict, gt, keep, iou_thresh, section_num):
        """_image_eval"""
        copy_predict = predict.copy()
        copy_gt = gt.copy()

        image_p_right = np.zeros(copy_predict.shape[0])
        image_gt_right = np.zeros(copy_gt.shape[0])
        proposal = np.ones(copy_predict.shape[0])

        # x1y1wh -> x1y1x2y2
        copy_predict[:, 2:4] = copy_predict[:, 0:2] + copy_predict[:, 2:4]
        copy_gt[:, 2:4] = copy_gt[:, 0:2] + copy_gt[:, 2:4]

        ious = self._iou(copy_predict[:, 0:4], copy_gt[:, 0:4])
        for i in range(copy_predict.shape[0]):
            gt_ious = ious[i, :]
            max_iou, max_index = gt_ious.max(), gt_ious.argmax()
            if max_iou >= iou_thresh:
                if keep[max_index] == 0:
                    image_gt_right[max_index] = -1
                    proposal[i] = -1
                elif image_gt_right[max_index] == 0:
                    image_gt_right[max_index] = 1

            right_index = np.where(image_gt_right == 1)[0]
            image_p_right[i] = len(right_index)



        image_pr = np.zeros((section_num, 2), dtype=np.float32)
        for section in range(section_num):
            score_thresh = 1 - (section + 1)/section_num
            over_score_index = np.where(predict[:, 4] >= score_thresh)[0]
            if over_score_index.shape[0] <= 0:
                image_pr[section, 0] = 0
                image_pr[section, 1] = 0
            else:
                index = over_score_index[-1]
                p_num = len(np.where(proposal[0:(index+1)] == 1)[0])
                image_pr[section, 0] = p_num
                image_pr[section, 1] = image_p_right[index]

        return image_pr


    def get_eval_result(self):
        """get_eval_result"""
        self._norm_pre_score()
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self._get_gt_boxes()
        section_num = 1000
        sets = ['easy', 'medium', 'hard']
        set_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        ap_key_dict = {0: "Easy   Val AP : ", 1: "Medium Val AP : ", 2: "Hard   Val AP : ",}
        ap_dict = {}
        for index_set in range(len(sets)):
            gt_list = set_gts[index_set]
            count_gt = 0
            pr_curve = np.zeros((section_num, 2), dtype=np.float32)
            for i, _ in enumerate(event_list):
                event = str(event_list[i][0][0])
                image_list = file_list[i][0]
                event_predict_dict = self.results[event]
                event_gt_index_list = gt_list[i][0]
                event_gt_box_list = facebox_list[i][0]

                for j, _ in enumerate(image_list):
                    predict = np.array(event_predict_dict[str(image_list[j][0][0])]['bboxes']).astype(np.float32)
                    gt_boxes = event_gt_box_list[j][0].astype('float')
                    keep_index = event_gt_index_list[j][0]
                    count_gt += len(keep_index)

                    if gt_boxes.shape[0] <= 0 or predict.shape[0] <= 0:
                        continue
                    keep = np.zeros(gt_boxes.shape[0])
                    if keep_index.shape[0] > 0:
                        keep[keep_index-1] = 1

                    image_pr = self._image_eval(predict, gt_boxes, keep,
                                                iou_thresh=self.iou_thresh,
                                                section_num=section_num)
                    pr_curve += image_pr

            precision = pr_curve[:, 1] / pr_curve[:, 0]
            recall = pr_curve[:, 1] / count_gt

            precision = np.concatenate((np.array([0.]), precision, np.array([0.])))
            recall = np.concatenate((np.array([0.]), recall, np.array([1.])))
            for i in range(precision.shape[0]-1, 0, -1):
                precision[i-1] = np.maximum(precision[i-1], precision[i])
            index = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])


            print(ap_key_dict[index_set] + f'{ap:.4f}')

        return ap_dict



GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """_clip_grad"""
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainingWrapper(nn.Cell):
    """TrainingWrapper

    Args:
        network (Object): The network.
        optimizer (Object): The optimizer.
        grad_clip (Bool): Whether to clip the gradient.
    """
    def __init__(self, network, optimizer, sens=1.0,grad_clip=True):
        super().__init__(auto_prefix=False)
        self.clip = grad_clip
        self.network = network
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.shape = ops.Shape()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        class_list = [mindspore.context.ParallelMode.DATA_PARALLEL, mindspore.context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = mindspore.ops.HyperMap()

    def construct(self, *args):
        """construct"""
        weights = self.weights
        loss = self.network(*args)
        sens = self.fill(self.dtype(loss), self.shape(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.clip:
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
