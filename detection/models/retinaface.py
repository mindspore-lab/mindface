"""Network."""
import math
from functools import reduce
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size


# RetinaFace
def Init_KaimingUniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    """Init_KaimingUniform"""
    def _calculate_in_and_out(arr_shape):
        dim = len(arr_shape)
        if dim < 2:
            raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

        n_in = arr_shape[1]
        n_out = arr_shape[0]

        if dim > 2:

            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def calculate_gain(nonlinearity, a=None):
        linear_fans = ['linear', 'conv1d', 'conv2d', 'conv3d',
                       'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fans or nonlinearity == 'sigmoid':
            return 1
        if nonlinearity == 'tanh':
            return 5.0 / 3
        if nonlinearity == 'relu':
            return math.sqrt(2.0)
        if nonlinearity == 'leaky_relu':
            if a is None:
                negative_slope = 0.01
            elif not isinstance(a, bool) and isinstance(a, int) or isinstance(a, float):
                negative_slope = a
            else:
                raise ValueError("negative_slope {} not a valid number".format(a))
            return math.sqrt(2.0 / (1 + negative_slope ** 2))

        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    fan_in, _ = _calculate_in_and_out(arr_shape)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weight = np.random.uniform(-bound, bound, arr_shape).astype(np.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(fan_in)
        bias = np.random.uniform(-bound_bias, bound_bias, arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return Tensor(weight), bias

class ConvBNReLU(nn.SequentialCell):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer, leaky=0):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = Init_KaimingUniform(weight_shape, a=math.sqrt(5))

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
            nn.LeakyReLU(alpha=leaky)
        )

class ConvBN(nn.SequentialCell):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = Init_KaimingUniform(weight_shape, a=math.sqrt(5))

        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
        )

class SSH(nn.Cell):
    """SSH"""
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1

        norm_layer = nn.BatchNorm2d
        self.conv3X3 = ConvBN(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, groups=1,
                              norm_layer=norm_layer)

        self.conv5X5_1 = ConvBNReLU(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv5X5_2 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.conv7X7_2 = ConvBNReLU(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv7X7_3 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.cat = P.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7X7_3(conv7X7_2)

        out = self.cat((conv3X3, conv5X5, conv7X7))
        out = self.relu(out)

        return out

class FPN(nn.Cell):
    """FPN"""
    def __init__(self, cfg):
        super(FPN, self).__init__()
        out_channels = cfg['out_channel']
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        norm_layer = nn.BatchNorm2d
        self.output1 = ConvBNReLU(cfg['in_channel'] * 2, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)
        self.output2 = ConvBNReLU(cfg['in_channel'] * 4, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)
        self.output3 = ConvBNReLU(cfg['in_channel'] * 8, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)

        self.merge1 = ConvBNReLU(cfg['out_channel'], cfg['out_channel'], kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)
        self.merge2 = ConvBNReLU(cfg['out_channel'], cfg['out_channel'], kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)

    def construct(self, input1, input2, input3):
        """construct"""
        output1 = self.output1(input1)
        output2 = self.output2(input2)
        output3 = self.output3(input3)

        up3 = P.ResizeNearestNeighbor([P.Shape()(output2)[2], P.Shape()(output2)[3]])(output3)
        output2 = up3 + output2
        output2 = self.merge2(output2)

        up2 = P.ResizeNearestNeighbor([P.Shape()(output1)[2], P.Shape()(output1)[3]])(output2)
        output1 = up2 + output1
        output1 = self.merge1(output1)

        return output1, output2, output3

class ClassHead(nn.Cell):
    """ClassHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors

        weight_shape = (self.num_anchors * 2, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0,
                                 has_bias=True, weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 2))

class BboxHead(nn.Cell):
    """BboxHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()

        weight_shape = (num_anchors * 4, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 4))

class LandmarkHead(nn.Cell):
    """LandmarkHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()

        weight_shape = (num_anchors * 10, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 10))

class RetinaFace(nn.Cell):
    """RetinaFace"""
    def __init__(self, phase='train', backbone=None, cfg=None):

        super(RetinaFace, self).__init__()
        self.phase = phase

        self.base = backbone

        self.fpn = FPN(cfg)

        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                      cfg['out_channel']], anchor_num=[2, 2, 2])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                    cfg['out_channel']], anchor_num=[2, 2, 2])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=[cfg['out_channel'],
                                                                            cfg['out_channel'],
                                                                            cfg['out_channel']],
                                                     anchor_num=[2, 2, 2])

        self.cat = P.Concat(axis=1)

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        classhead = nn.CellList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels[i], anchor_num[i]))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        bboxhead = nn.CellList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels[i], anchor_num[i]))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num):
        landmarkhead = nn.CellList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels[i], anchor_num[i]))
        return landmarkhead

    def construct(self, inputs):
        """construct"""
        f1, f2, f3 = self.base(inputs)
        f1, f2, f3 = self.fpn(f1, f2, f3)

        # SSH
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        features = [f1, f2, f3]

        bbox = ()
        for i, feature in enumerate(features):
            bbox = bbox + (self.BboxHead[i](feature),)
        bbox_regressions = self.cat(bbox)

        cls = ()
        for i, feature in enumerate(features):
            cls = cls + (self.ClassHead[i](feature),)
        classifications = self.cat(cls)

        landm = ()
        for i, feature in enumerate(features):
            landm = landm + (self.LandmarkHead[i](feature),)
        ldm_regressions = self.cat(landm)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, P.Softmax(-1)(classifications), ldm_regressions)

        return output

class RetinaFaceWithLossCell(nn.Cell):
    """RetinaFaceWithLossCell"""
    def __init__(self, network, multibox_loss, config):
        super(RetinaFaceWithLossCell, self).__init__()
        self.network = network
        self.loc_weight = config['loc_weight']
        self.class_weight = config['class_weight']
        self.landm_weight = config['landm_weight']
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t, landm_t):
        pred_loc, pre_conf, pre_landm = self.network(img)
        loss_loc, loss_conf, loss_landm = self.multibox_loss(pred_loc, loc_t, pre_conf, conf_t, pre_landm, landm_t)

        return loss_loc * self.loc_weight + loss_conf * self.class_weight + loss_landm * self.landm_weight


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
    """TrainingWrapper"""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
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
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
